"""Control-room web server: REST API, WebSocket push, live MJPEG, incident review & report support."""
from __future__ import annotations

import asyncio
import json
import threading
import time
import uuid
from contextlib import asynccontextmanager
from datetime import datetime
from pathlib import Path
from typing import Optional

from fastapi import Body, FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, JSONResponse, PlainTextResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles

from ..config import ROOT, load_policy, load_site
from ..incidents.manager import IncidentManager
from ..incidents.report import build_report, template_summary
from ..risk.policy import Policy
from ..store import Store

STATIC = Path(__file__).parent / "static"


def _now() -> str:
    return datetime.now().astimezone().isoformat(timespec="milliseconds")


def _secs(a: Optional[str], b: Optional[str]) -> Optional[float]:
    if not a or not b:
        return None
    return round((datetime.fromisoformat(b) - datetime.fromisoformat(a)).total_seconds(), 2)


class Runtime:
    def __init__(self, site_path=None, policy_path=None, data_dir: Path = ROOT / "runs" / "live",
                 eval_dir: Path = ROOT / "runs" / "eval", blur_heads: bool = False):
        self.site_path, self.policy_path = site_path, policy_path
        self.site = load_site(site_path)
        self.policy_cfg = load_policy(policy_path)
        self.policy = Policy(self.policy_cfg)
        self.data_dir = Path(data_dir)
        self.eval_dir = Path(eval_dir)
        self.blur_heads = blur_heads
        self.store = Store(self.data_dir / "incidents.db")
        self.manager = IncidentManager(self.site, self.policy, self.data_dir)
        for doc in self.store.load_all():  # history from previous sessions (read-only)
            doc.setdefault("_last_active", 0)
            doc.setdefault("_popup", True)
            if doc.get("status") == "OPEN":
                doc["status"] = "CLOSED"
            self.manager.incidents[doc["id"]] = doc
        self.manager.listeners.append(self._on_incident)
        self.pipelines: dict = {}
        self.jobs: dict[str, dict] = {}
        self.clients: set[WebSocket] = set()
        self.loop: Optional[asyncio.AbstractEventLoop] = None
        self._last_push: dict[str, float] = {}
        self._last_save: dict[str, float] = {}
        self.lock = threading.RLock()

    # ------------------------------------------------------------ events
    def _on_incident(self, kind: str, inc: dict) -> None:
        now = time.monotonic()
        important = kind != "incident_updated"
        if important or now - self._last_save.get(inc["id"], 0) > 1.0:
            inc["summary"] = template_summary(self.manager.public(inc))
            self.store.save(inc)
            self._last_save[inc["id"]] = now
        if important or now - self._last_push.get(inc["id"], 0) > 0.5:
            self._last_push[inc["id"]] = now
            self.broadcast({"kind": kind, "incident": self.card(inc)})

    def broadcast(self, msg: dict) -> None:
        if not self.loop or not self.clients:
            return
        data = json.dumps(msg, ensure_ascii=False)
        for ws in list(self.clients):
            asyncio.run_coroutine_threadsafe(self._send(ws, data), self.loop)

    async def _send(self, ws: WebSocket, data: str) -> None:
        try:
            await ws.send_text(data)
        except Exception:
            self.clients.discard(ws)

    def card(self, inc: dict) -> dict:
        keys = ["id", "status", "review", "type", "type_ko", "level", "peak_level", "peak_score", "start_time",
                "primary_camera", "cameras", "location", "objects"]
        c = {k: inc.get(k) for k in keys}
        c["popup"] = bool(inc.get("_popup"))
        c["n_actors"] = len(inc.get("actors", {}))
        return c

    # ------------------------------------------------------------ jobs
    def start_scenario(self, name: str, speed: float = 1.0, seed: int = 0) -> dict:
        from ..pipeline import run_scenario
        from ..sim import library

        scn = library.get(name)
        stop = threading.Event()
        jid = uuid.uuid4().hex[:8]
        job = {"id": jid, "kind": "scenario", "name": name, "title": scn.title, "cameras": scn.cameras,
               "status": "running", "started": _now(), "stop": stop, "error": None}

        def run():
            try:
                run_scenario(scn, self.site, self.policy, self.manager, seed=seed, realtime=True, speed=speed,
                             stop=stop, pipelines_out=self.pipelines, blur_heads=self.blur_heads)
                self.manager.finish()
                job["status"] = "stopped" if stop.is_set() else "finished"
            except Exception as e:  # surface errors to the UI
                job["status"], job["error"] = "error", repr(e)
            self.broadcast({"kind": "job", "job": self.job_public(job)})

        job["thread"] = threading.Thread(target=run, daemon=True)
        self.jobs[jid] = job
        job["thread"].start()
        return self.job_public(job)

    def start_source(self, camera_id: str, source: str, backend: str = "yolo", realtime: Optional[bool] = None,
                     analysis_fps: float = 10.0, max_seconds: Optional[float] = None) -> dict:
        from ..pipeline import run_source

        cam = self.site.camera(camera_id)
        stop = threading.Event()
        jid = uuid.uuid4().hex[:8]
        job = {"id": jid, "kind": "source", "name": source, "title": f"{camera_id} ← {source}",
               "cameras": [camera_id], "status": "starting", "started": _now(), "stop": stop, "error": None}

        def run():
            try:
                if backend == "yolo":
                    from ..perception.yolo import YoloPerception
                    perception = YoloPerception()
                else:
                    raise ValueError(f"unknown backend {backend}")
                job["status"] = "running"
                run_source(source, cam, self.policy, self.manager, perception, analysis_fps=analysis_fps,
                           realtime=realtime, stop=stop, pipelines_out=self.pipelines, max_seconds=max_seconds,
                           blur_heads=self.blur_heads, on_status=lambda s: job.__setitem__("stream", s))
                self.manager.finish()
                job["status"] = "stopped" if stop.is_set() else "finished"
            except Exception as e:
                job["status"], job["error"] = "error", repr(e)
            self.broadcast({"kind": "job", "job": self.job_public(job)})

        job["thread"] = threading.Thread(target=run, daemon=True)
        self.jobs[jid] = job
        job["thread"].start()
        return self.job_public(job)

    @staticmethod
    def job_public(job: dict) -> dict:
        return {k: v for k, v in job.items() if k not in ("thread", "stop")}

    # ------------------------------------------------------------ incident helpers
    def get(self, iid: str) -> dict:
        inc = self.manager.incidents.get(iid)
        if inc is None:
            raise HTTPException(404, "incident not found")
        return inc

    def mark(self, inc: dict, key: str) -> None:
        inc.setdefault("kpi", {})
        if not inc["kpi"].get(key):
            inc["kpi"][key] = _now()

    def kpi_row(self, inc: dict) -> dict:
        k = inc.get("kpi", {})
        alert = k.get("alert_wall") or k.get("detected_wall")
        rep = inc.get("report") or {}
        return {
            "id": inc["id"], "type": inc["type_ko"], "peak_level": inc["peak_level"],
            "review": inc.get("review"), "feedback": (inc.get("feedback") or {}).get("label"),
            "detect_latency_s": _secs(k.get("event_start_wall"), k.get("detected_wall")),
            "alert_latency_s": _secs(k.get("event_start_wall"), k.get("alert_wall")),
            "recognition_s": _secs(alert, k.get("first_viewed_wall")),          # 사건 인지 시간
            "understanding_s": _secs(k.get("first_viewed_wall"), k.get("acked_wall")),  # 상황 파악 시간
            "report_prep_s": _secs(k.get("acked_wall") or k.get("first_viewed_wall"), k.get("report_ready_wall")),
            "total_response_s": _secs(k.get("event_start_wall"), k.get("reported_wall") or k.get("report_ready_wall")),
            "report_status": rep.get("status"),
        }


def create_app(runtime: Optional[Runtime] = None, **kw) -> FastAPI:
    rt = runtime or Runtime(**kw)

    @asynccontextmanager
    async def lifespan(_app):
        rt.loop = asyncio.get_running_loop()
        yield
        for j in rt.jobs.values():
            j["stop"].set()

    app = FastAPI(title="SafeWatch 관제 지원", version="0.1.0", lifespan=lifespan)
    app.state.rt = rt
    rt.data_dir.mkdir(parents=True, exist_ok=True)
    app.mount("/media", StaticFiles(directory=str(rt.data_dir)), name="media")
    app.mount("/static", StaticFiles(directory=str(STATIC)), name="static")

    @app.get("/")
    def index():
        return FileResponse(STATIC / "index.html")

    # ---------------------------------------------------------------- site & jobs
    @app.get("/api/site")
    def site():
        cams = []
        for c in rt.site.cameras.values():
            cams.append({"id": c.id, "name": c.name, "address": c.address, "lat": c.lat, "lon": c.lon,
                         "place_type": c.place_type, "neighbors": list(c.neighbors),
                         "zones": [{"id": z.id, "name": z.name, "type": z.type} for z in c.zones],
                         "live": c.id in rt.pipelines})
        return {"name": rt.site.name, "cameras": cams,
                "policy": {"provenance": rt.policy_cfg.get("provenance"), "levels": rt.policy_cfg.get("levels"),
                           "popup_min_level": rt.policy.popup_min_level.name}}

    @app.get("/api/scenarios")
    def scenarios():
        from ..sim import library
        out = []
        for name, fn in library.ALL.items():
            s = fn()
            out.append({"name": name, "title": s.title, "duration": s.duration, "cameras": s.cameras,
                        "hard_negative": s.hard_negative, "events": [e.type for e in s.events]})
        return out

    @app.get("/api/status")
    def status():
        return {"jobs": [rt.job_public(j) for j in rt.jobs.values()],
                "pipelines": {k: p.stats.as_dict() for k, p in rt.pipelines.items()},
                "pending_encodes": rt.manager.pending_encodes}

    @app.post("/api/jobs")
    def start_job(body: dict = Body(...)):
        kind = body.get("kind", "scenario")
        if kind == "scenario":
            return rt.start_scenario(body["name"], float(body.get("speed", 1.0)), int(body.get("seed", 0)))
        if kind == "source":
            if not body.get("source") or not body.get("camera_id"):
                raise HTTPException(400, "camera_id and source are required")
            return rt.start_source(body["camera_id"], body["source"], body.get("backend", "yolo"),
                                   body.get("realtime"), float(body.get("analysis_fps", 10)),
                                   body.get("max_seconds"))
        raise HTTPException(400, "unknown job kind")

    @app.post("/api/jobs/{jid}/stop")
    def stop_job(jid: str):
        job = rt.jobs.get(jid)
        if not job:
            raise HTTPException(404)
        job["stop"].set()
        return rt.job_public(job)

    # ---------------------------------------------------------------- live video
    @app.get("/api/cameras/{cid}/snapshot.jpg")
    def snapshot(cid: str):
        p = rt.pipelines.get(cid)
        if not p or not p.latest_jpeg:
            raise HTTPException(404, "no live frame")
        from fastapi.responses import Response
        return Response(p.latest_jpeg, media_type="image/jpeg")

    @app.get("/api/cameras/{cid}/live.mjpg")
    async def live(cid: str):
        async def gen():
            last = None
            idle = 0
            while idle < 600:
                p = rt.pipelines.get(cid)
                frame = p.latest_jpeg if p else None
                if frame is not None and frame is not last:
                    last = frame
                    idle = 0
                    yield b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + frame + b"\r\n"
                else:
                    idle += 1
                await asyncio.sleep(0.1)
        return StreamingResponse(gen(), media_type="multipart/x-mixed-replace; boundary=frame")

    # ---------------------------------------------------------------- incidents
    @app.get("/api/incidents")
    def incidents(limit: int = 200):
        items = sorted(rt.manager.incidents.values(), key=lambda i: i["kpi"].get("detected_wall", ""), reverse=True)
        return [rt.card(i) for i in items[:limit]]

    @app.get("/api/incidents/{iid}")
    def incident(iid: str):
        inc = rt.get(iid)
        pub = rt.manager.public(inc)
        pub["summary"] = template_summary(pub)
        pub["audit"] = rt.store.audit_log(iid)
        pub["kpi_derived"] = rt.kpi_row(inc)
        return pub

    def _review(inc, state, action, actor="", detail=None):
        order = ["NEW", "VIEWED", "ACKED", "REPORT_READY", "REPORTED"]
        if state in order and inc.get("review") in order and order.index(state) <= order.index(inc["review"]):
            pass
        else:
            inc["review"] = state
        rt.store.audit(inc["id"], action, actor, detail)
        rt.store.save(inc)
        rt.broadcast({"kind": "incident_updated", "incident": rt.card(inc)})

    @app.post("/api/incidents/{iid}/view")
    def view(iid: str, body: dict = Body(default={})):
        inc = rt.get(iid)
        rt.mark(inc, "first_viewed_wall")
        _review(inc, "VIEWED", "view", body.get("operator", ""))
        return rt.kpi_row(inc)

    @app.post("/api/incidents/{iid}/ack")
    def ack(iid: str, body: dict = Body(default={})):
        inc = rt.get(iid)
        rt.mark(inc, "first_viewed_wall")
        rt.mark(inc, "acked_wall")
        _review(inc, "ACKED", "ack", body.get("operator", ""))
        return rt.kpi_row(inc)

    @app.post("/api/incidents/{iid}/feedback")
    def feedback(iid: str, body: dict = Body(...)):
        inc = rt.get(iid)
        label = body.get("label")
        if label not in ("TP", "FP", "UNSURE"):
            raise HTTPException(400, "label must be TP, FP or UNSURE")
        inc["feedback"] = {"label": label, "note": body.get("note", ""), "by": body.get("operator", ""), "at": _now()}
        rt.store.audit(iid, "feedback", body.get("operator", ""), inc["feedback"])
        rt.store.save(inc)
        rt.broadcast({"kind": "incident_updated", "incident": rt.card(inc)})
        return inc["feedback"]

    @app.post("/api/incidents/{iid}/dismiss")
    def dismiss(iid: str, body: dict = Body(default={})):
        inc = rt.get(iid)
        inc["review"] = "DISMISSED"
        rt.store.audit(iid, "dismiss", body.get("operator", ""), {"reason": body.get("reason", "")})
        rt.store.save(inc)
        rt.broadcast({"kind": "incident_updated", "incident": rt.card(inc)})
        return {"ok": True}

    @app.get("/api/incidents/{iid}/report")
    def get_report(iid: str):
        inc = rt.get(iid)
        if not inc.get("report"):
            inc["report"] = build_report(rt.manager.public(inc), rt.policy_cfg.get("provenance"))
            rt.mark(inc, "report_opened_wall")
            rt.store.audit(iid, "report_draft")
            rt.store.save(inc)
        else:
            # refresh AI-derived parts (clips may have finished encoding) but keep operator edits
            fresh = build_report(rt.manager.public(inc), rt.policy_cfg.get("provenance"))
            if inc["report"]["status"] == "DRAFT":
                fresh["fields"].update({k: v for k, v in inc["report"].get("edited", {}).items()})
                fresh["operator"] = inc["report"]["operator"]
                fresh["edited"] = inc["report"].get("edited", {})
                fresh["call_script"] = inc["report"].get("call_script_edited") or fresh["call_script"]
                fresh["call_script_edited"] = inc["report"].get("call_script_edited")
                inc["report"] = fresh
        return inc["report"]

    @app.put("/api/incidents/{iid}/report")
    def edit_report(iid: str, body: dict = Body(...)):
        inc = rt.get(iid)
        rep = inc.get("report") or get_report(iid)
        if rep["status"] == "REPORTED":
            raise HTTPException(409, "already reported")
        for k, v in (body.get("fields") or {}).items():
            rep["fields"][k] = v
            rep.setdefault("edited", {})[k] = v
        if "call_script" in body:
            rep["call_script"] = rep["call_script_edited"] = body["call_script"]
        rep["operator"].update({k: v for k, v in (body.get("operator") or {}).items()
                                if k in ("name", "memo", "agency", "verified", "receipt_no")})
        rt.store.audit(iid, "report_edit", rep["operator"].get("name", ""), body)
        rt.store.save(inc)
        return rep

    @app.post("/api/incidents/{iid}/report/confirm")
    def confirm_report(iid: str, body: dict = Body(default={})):
        inc = rt.get(iid)
        rep = inc.get("report") or get_report(iid)
        op = rep["operator"]
        op.update({k: v for k, v in body.items() if k in ("name", "memo", "agency", "verified")})
        if not op.get("verified"):
            raise HTTPException(400, "관제자가 영상으로 직접 확인했다는 체크(verified)가 필요합니다")
        if not op.get("name"):
            raise HTTPException(400, "확인자 이름이 필요합니다")
        rep["status"] = "CONFIRMED"
        op["confirmed_at"] = _now()
        rt.mark(inc, "report_ready_wall")
        _review(inc, "REPORT_READY", "report_confirm", op["name"], {"agency": op.get("agency")})
        return rep

    @app.post("/api/incidents/{iid}/report/reported")
    def reported(iid: str, body: dict = Body(default={})):
        """Records that a *human operator* made the report (the system never calls 112/119 itself)."""
        inc = rt.get(iid)
        rep = inc.get("report")
        if not rep or rep["status"] not in ("CONFIRMED", "REPORTED"):
            raise HTTPException(400, "신고 정보 확정(confirm) 후에 신고 완료를 기록할 수 있습니다")
        rep["status"] = "REPORTED"
        rep["operator"].update({k: v for k, v in body.items() if k in ("agency", "receipt_no", "memo")})
        rep["operator"]["reported_at"] = _now()
        rt.mark(inc, "reported_wall")
        _review(inc, "REPORTED", "reported", rep["operator"].get("name", ""),
                {"agency": rep["operator"].get("agency"), "receipt_no": rep["operator"].get("receipt_no")})
        return rep

    @app.get("/api/incidents/{iid}/report.txt", response_class=PlainTextResponse)
    def report_txt(iid: str):
        rep = get_report(iid)
        f = rep["fields"]
        lines = [f"[신고 지원 정보] {rep['incident_id']}  상태: {rep['status']}", "",
                 f"발생시간: {f['occurred_at']}", f"위치: {f['location']['name']} / {f['location']['address']} "
                 f"({f['location']['lat']}, {f['location']['lon']})",
                 f"사건 유형: {f['incident_type']} (위험도 {f['risk_level']})",
                 f"위험물: {', '.join(f['dangerous_objects'])}",
                 f"관련 인원: {f['persons']['count']}명 — " + "; ".join(f["persons"]["descriptions"]),
                 f"행동: {', '.join(f['behaviors'])}", f"이동방향: {', '.join(f['movement_direction'])}",
                 "관련 CCTV: " + ", ".join(f"{c['camera']}({c['name']})" for c in f["related_cctv"]),
                 "사건 영상: " + ", ".join(v["url"] for v in f["video"]),
                 f"권장 기관: {', '.join(f['recommended_agency'])}", "", "[AI 분석]", f["ai_analysis"]["summary"], "",
                 "[신고 문안]", rep["call_script"], "", f"확인자: {rep['operator'].get('name')}  "
                 f"확정: {rep['operator'].get('confirmed_at')}  신고: {rep['operator'].get('reported_at')} "
                 f"{rep['operator'].get('agency')} {rep['operator'].get('receipt_no')}", "", rep["disclaimer"]]
        return "\n".join(lines)

    # ---------------------------------------------------------------- KPI & evaluation
    @app.get("/api/kpi")
    def kpi():
        rows = [rt.kpi_row(i) for i in rt.manager.incidents.values()]

        def avg(key):
            v = [r[key] for r in rows if r[key] is not None]
            return {"mean": round(sum(v) / len(v), 2), "n": len(v)} if v else {"mean": None, "n": 0}

        fb = [r["feedback"] for r in rows if r["feedback"]]
        return {"rows": rows, "summary": {k: avg(k) for k in ["detect_latency_s", "alert_latency_s", "recognition_s",
                                                                "understanding_s", "report_prep_s", "total_response_s"]},
                "feedback": {"TP": fb.count("TP"), "FP": fb.count("FP"), "UNSURE": fb.count("UNSURE"),
                             "operator_precision": round(fb.count("TP") / (fb.count("TP") + fb.count("FP")), 3)
                             if (fb.count("TP") + fb.count("FP")) else None}}

    @app.get("/api/eval/latest")
    def eval_latest():
        p = rt.eval_dir / "report.json"
        if not p.exists():
            return JSONResponse({"available": False, "hint": "safewatch eval 명령으로 평가 리포트를 생성하세요"})
        r = json.loads(p.read_text(encoding="utf-8"))
        r["available"] = True
        return r

    # ---------------------------------------------------------------- websocket
    @app.websocket("/ws")
    async def ws(socket: WebSocket):
        await socket.accept()
        rt.clients.add(socket)
        try:
            while True:
                await socket.receive_text()
        except WebSocketDisconnect:
            pass
        finally:
            rt.clients.discard(socket)

    return app
