"""SafeWatch command line.

  safewatch demo                    대시보드 실행 + 데모 시나리오 자동 재생
  safewatch serve                   대시보드만 실행 (UI에서 시나리오/영상/RTSP 시작)
  safewatch scenarios               내장 시나리오 목록
  safewatch run SCENARIO            시나리오를 헤드리스로 분석 → 사건/클립/신고패키지 저장
  safewatch analyze --source X      영상 파일 또는 RTSP를 YOLO 백엔드로 헤드리스 분석
  safewatch eval                    사건 단위 평가 (P/R/F1, PR, FP/cam-h, 지연) [--fit-policy]
  safewatch export-sim SCENARIO     합성 시나리오를 mp4 + 정답(JSON)으로 내보내기
"""
from __future__ import annotations

import argparse
import json
import sys
import threading
import time
from datetime import datetime
from pathlib import Path

from .config import ROOT, load_policy, load_site


def _policy(args):
    from .risk.policy import Policy
    return Policy(load_policy(args.policy))


def cmd_scenarios(args):
    from .sim import library
    for name, fn in library.ALL.items():
        s = fn()
        tag = "[정상-유사]" if s.hard_negative else "[위험]   "
        ev = ", ".join(f"{e.type}({e.start:.0f}-{e.end:.0f}s)" for e in s.events) or "-"
        print(f"{tag} {name:22s} {s.duration:5.0f}s  {','.join(s.cameras):18s} {s.title}  GT: {ev}")


def _dump_incidents(mgr, out: Path, policy_cfg):
    from .incidents.report import build_report, template_summary
    # wait for background encoders
    for _ in range(600):
        if mgr.pending_encodes <= 0:
            break
        time.sleep(0.1)
    docs = []
    for inc in mgr.incidents.values():
        pub = mgr.public(inc)
        pub["summary"] = template_summary(pub)
        pub["report"] = build_report(pub, policy_cfg.get("provenance"))
        docs.append(pub)
    (out / "incidents.json").write_text(json.dumps(docs, ensure_ascii=False, indent=1), encoding="utf-8")
    for d in docs:
        print(f"\n=== {d['id']}  {d['type_ko']}  최고 {d['peak_level']} ({d['peak_score']:.2f})  카메라 {d['cameras']}")
        print(d["summary"])
        print("클립:", {c: v["path"] for c, v in d["clips"].items()})
    print(f"\n저장: {out / 'incidents.json'} (사건 {len(docs)}건)")


def cmd_run(args):
    from .incidents.manager import IncidentManager
    from .pipeline import run_scenario
    from .sim import library
    site, pol = load_site(args.site), _policy(args)
    scn = library.get(args.scenario)
    out = Path(args.out or ROOT / "runs" / f"{args.scenario}-{datetime.now():%Y%m%d-%H%M%S}")
    mgr = IncidentManager(site, pol, out, encode_background=False)
    t0 = time.time()
    pipes = run_scenario(scn, site, pol, mgr, seed=args.seed, blur_heads=args.blur_heads)
    mgr.finish()
    print(f"[run] {scn.name}: {scn.duration:.0f}s 영상, 처리 {time.time() - t0:.1f}s, "
          + ", ".join(f"{k} {p.stats.as_dict()['proc_ms_avg']}ms/f" for k, p in pipes.items()))
    _dump_incidents(mgr, out, pol.cfg)


def cmd_analyze(args):
    from .incidents.manager import IncidentManager
    from .perception.yolo import YoloPerception
    from .pipeline import run_source
    site, pol = load_site(args.site), _policy(args)
    cam = site.camera(args.camera)
    out = Path(args.out or ROOT / "runs" / f"analyze-{datetime.now():%Y%m%d-%H%M%S}")
    mgr = IncidentManager(site, pol, out, encode_background=False)
    perception = YoloPerception(pose_model=args.pose_model, det_model=args.det_model, imgsz=args.imgsz,
                                device=args.device)
    t0 = time.time()
    pipe = run_source(args.source, cam, pol, mgr, perception, analysis_fps=args.fps, realtime=False,
                      max_seconds=args.max_seconds, blur_heads=args.blur_heads,
                      on_status=lambda s: print(f"[source] {s}"))
    mgr.finish()
    st = pipe.stats.as_dict()
    print(f"[analyze] frames={st['frames']} media={st['last_ts']}s wall={time.time() - t0:.1f}s "
          f"proc={st['proc_ms_avg']}ms/f (max {st['proc_ms_max']}ms)")
    _dump_incidents(mgr, out, pol.cfg)


def cmd_eval(args):
    from .eval.runner import run_eval
    from .sim import library
    scen = args.scenarios or list(library.ALL)
    run_eval(scen, args.seeds, Path(args.out), site_path=args.site, policy_path=args.policy, noise=args.noise,
             workers=args.workers, fit_policy=Path(args.fit_policy) if args.fit_policy else None,
             calibrate=args.calibrate)


def cmd_export_sim(args):
    import cv2
    import imageio_ffmpeg
    from dataclasses import asdict
    from .sim import library
    from .sim.render import iter_camera
    from .sim.scenario import SimCamera
    site = load_site(args.site)
    scn = library.get(args.scenario)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    for cid in scn.cameras:
        c = site.camera(cid)
        cam = SimCamera(cid, tuple(c.sim["world"]))
        path = out / f"{scn.name}_{cid}.mp4"
        gen = imageio_ffmpeg.write_frames(str(path), (cam.width, cam.height), fps=args.fps, codec="libx264",
                                          pix_fmt_out="yuv420p", macro_block_size=2)
        gen.send(None)
        for f in iter_camera(scn, cam, fps=args.fps):
            gen.send(f.image[:, :, ::-1].copy())
        gen.close()
        print("wrote", path)
    gt = {"scenario": scn.name, "start_clock": scn.start_clock, "cameras": scn.cameras,
          "events": [asdict(e) for e in scn.events]}
    (out / f"{scn.name}_gt.json").write_text(json.dumps(gt, ensure_ascii=False, indent=1), encoding="utf-8")
    print("wrote", out / f"{scn.name}_gt.json")


def _serve(args, autoplay: list[str] | None = None):
    import uvicorn
    from .server.app import Runtime, create_app
    rt = Runtime(site_path=args.site, policy_path=args.policy,
                 data_dir=Path(args.data_dir) if args.data_dir else ROOT / "runs" / "live",
                 eval_dir=Path(args.eval_dir) if args.eval_dir else ROOT / "runs" / "eval",
                 blur_heads=args.blur_heads)
    app = create_app(rt)
    if autoplay:
        def play():
            time.sleep(2.0)
            for name in autoplay:
                job = rt.start_scenario(name, speed=args.speed)
                while rt.jobs[job["id"]]["status"] == "running":
                    time.sleep(0.5)
                time.sleep(3.0)
        threading.Thread(target=play, daemon=True).start()
    print(f"\n  SafeWatch 대시보드:  http://{args.host if args.host != '0.0.0.0' else 'localhost'}:{args.port}\n")
    uvicorn.run(app, host=args.host, port=args.port, log_level="warning")


def cmd_serve(args):
    _serve(args)


def cmd_demo(args):
    _serve(args, autoplay=args.play or ["knife_threat_chase", "assault_fight", "jogging_pair", "collapse_alone"])


def main(argv=None):
    ap = argparse.ArgumentParser(prog="safewatch", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--site", default=None, help="사이트(카메라·구역) 설정 YAML")
    ap.add_argument("--policy", default=None, help="위험도 정책 YAML")
    sub = ap.add_subparsers(dest="cmd", required=True)

    def server_opts(p):
        p.add_argument("--host", default="127.0.0.1")
        p.add_argument("--port", type=int, default=8000)
        p.add_argument("--data-dir", default=None)
        p.add_argument("--eval-dir", default=None)
        p.add_argument("--blur-heads", action="store_true", help="영상 내 머리 영역 블러(개인정보 보호)")
        p.add_argument("--speed", type=float, default=1.0)

    p = sub.add_parser("demo", help="대시보드 + 데모 시나리오 자동 재생")
    server_opts(p)
    p.add_argument("--play", nargs="*", help="자동 재생할 시나리오 이름들")
    p.set_defaults(fn=cmd_demo)
    p = sub.add_parser("serve", help="대시보드 서버")
    server_opts(p)
    p.set_defaults(fn=cmd_serve)
    p = sub.add_parser("scenarios", help="시나리오 목록")
    p.set_defaults(fn=cmd_scenarios)
    p = sub.add_parser("run", help="시나리오 헤드리스 분석")
    p.add_argument("scenario")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out")
    p.add_argument("--blur-heads", action="store_true")
    p.set_defaults(fn=cmd_run)
    p = sub.add_parser("analyze", help="영상 파일/RTSP 분석 (YOLO 백엔드)")
    p.add_argument("--source", required=True, help="영상 파일 경로 또는 rtsp:// URL")
    p.add_argument("--camera", default="cam01", help="사이트 설정의 카메라 ID (구역/위치 정보 사용)")
    p.add_argument("--fps", type=float, default=5.0, help="분석 FPS")
    p.add_argument("--max-seconds", type=float, default=None)
    p.add_argument("--pose-model", default="yolo11n-pose.pt")
    p.add_argument("--det-model", default="yolo11n.pt")
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--device", default="cpu")
    p.add_argument("--out")
    p.add_argument("--blur-heads", action="store_true")
    p.set_defaults(fn=cmd_analyze)
    p = sub.add_parser("eval", help="사건 단위 성능 평가")
    p.add_argument("--scenarios", nargs="*")
    p.add_argument("--seeds", nargs="*", type=int, default=[1, 2, 3])
    p.add_argument("--noise", choices=["default", "hard"], default="default")
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--out", default=str(ROOT / "runs" / "eval"))
    p.add_argument("--fit-policy", default=None, help="PR 운영점으로 임계값을 적합한 새 정책 파일 경로")
    p.add_argument("--calibrate", action="store_true", help="isotonic 보정 테이블 산출")
    p.set_defaults(fn=cmd_eval)
    p = sub.add_parser("export-sim", help="합성 시나리오 → mp4 + GT JSON")
    p.add_argument("scenario")
    p.add_argument("--out", default=str(ROOT / "runs" / "sim-export"))
    p.add_argument("--fps", type=float, default=10.0)
    p.set_defaults(fn=cmd_export_sim)
    args = ap.parse_args(argv)
    args.fn(args)


if __name__ == "__main__":
    sys.exit(main())
