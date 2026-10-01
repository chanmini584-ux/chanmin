"""AI incident summary and report-support package (관제자 확인 후 사람이 신고).

The summary is template-based and built only from recorded evidence, so every sentence can be
traced back to a measurement. A VLM/LLM summarizer can be plugged in later via `Summarizer`,
but its output must still be cross-checked against this evidence (see docs).
"""
from __future__ import annotations

from datetime import datetime
from typing import Protocol

from ..types import Level, SituationType as ST

ROLE_KO = {"actor": "행위자(추정)", "target": "상대방", "subject": "당사자"}
AGENCY = {
    ST.ASSAULT: ["112"], ST.WEAPON_THREAT: ["112"], ST.CHASE: ["112"],
    ST.FALL: ["119"], ST.INTRUSION: ["112", "시설관리자"], ST.LOITERING: ["현장 순찰 요청"],
}
DISCLAIMER = "※ AI 분석 결과이며 사실 확인 전입니다. 관제자가 영상으로 확인한 뒤 신고 여부를 판단합니다."


class Summarizer(Protocol):
    def __call__(self, inc: dict) -> str: ...


def _hms(iso: str) -> str:
    try:
        return datetime.fromisoformat(iso).strftime("%H:%M:%S")
    except Exception:
        return iso


def _actor_desc(a: dict) -> str:
    parts = [ROLE_KO.get(a.get("role"), "")]
    if a.get("color"):
        parts.append(f"상의 {a['color']} 계열")
    if a.get("weapon"):
        parts.append(a["weapon"])
    if a.get("direction") and a["direction"] != "정지":
        parts.append(f"{a['direction']} 방향 이동")
    if a.get("linked_from"):
        parts.append(f"{a['linked_from']}와 동일인 후보(확인 필요)")
    return f"인물 #{a['track_id']}({a['camera']}): " + ", ".join(p for p in parts if p)


def _persons(inc: dict) -> list[dict]:
    """Group actor tracks; cross-camera links are shown under the same person entry."""
    groups, seen = [], set()
    for k, a in inc["actors"].items():
        if k in seen:
            continue
        chain = [a]
        seen.add(k)
        for k2, b in inc["actors"].items():
            if b.get("linked_from") == k and k2 not in seen:
                chain.append(b)
                seen.add(k2)
        if a.get("linked_from") and a["linked_from"] in inc["actors"]:
            continue  # will be listed under its source
        groups.append({"tracks": chain, "role": a.get("role")})
    return groups


def template_summary(inc: dict) -> str:
    loc = inc["location"]
    lines = [f"[AI 사건 요약] {inc['start_time'].replace('T', ' ')[:19]}경 {loc['name']}({loc['camera']})에서 "
             f"'{inc['type_ko']}' 상황이 감지되었습니다. 최고 위험도 {inc['peak_level']} "
             f"(보정 점수 {inc['peak_score']:.2f})."]
    persons = _persons(inc)
    if persons:
        lines.append(f"관련 인원 {len(persons)}명:")
        for g in persons:
            lines.append("  - " + " → ".join(_actor_desc(a) for a in g["tracks"]))
    if inc["objects"]:
        lines.append("위험물: " + ", ".join(inc["objects"]))
    if inc["timeline"]:
        lines.append("경과:")
        for e in inc["timeline"][:12]:
            lines.append(f"  - {_hms(e['time'])} [{e['camera']}] {e['text']}")
    linked = [r for r in inc["related_cameras"] if r["status"] in ("linked", "predicted")]
    if linked:
        lines.append("이동/주변 CCTV: " + "; ".join(f"{r['camera']}({r['name']}) - {r['reason']}" for r in linked))
    top = max(inc["situations"].values(), key=lambda r: r["peak_score"], default=None)
    if top:
        ev = [e["text"] for e in top["evidence"] if (e["weight"] or 0) >= 0]
        neg = [e["text"] for e in top["evidence"] if (e["weight"] or 0) < 0]
        lines.append("판단 근거: " + " / ".join(ev))
        if neg:
            lines.append("반대 근거: " + " / ".join(neg))
    lines.append(DISCLAIMER)
    return "\n".join(lines)


def build_report(inc: dict, policy_provenance: dict | None = None, base_url: str = "") -> dict:
    """Structured report-support package. Operator fields stay empty until a human fills them."""
    loc = inc["location"]
    persons = _persons(inc)
    best: dict[str, dict] = {}
    for r in inc["situations"].values():
        if Level[r["peak_level"]] >= Level.ATTENTION and (
                r["type"] not in best or r["peak_score"] > best[r["type"]]["peak_score"]):
            best[r["type"]] = r
    behaviors = [f"{r['type_ko']} ({r['peak_level']})"
                 for r in sorted(best.values(), key=lambda r: -r["peak_score"])]
    directions = sorted({a["direction"] for a in inc["actors"].values()
                         if a.get("direction") and a["direction"] != "정지"})
    agencies = []
    for r in inc["situations"].values():
        if Level[r["peak_level"]] >= Level.WARNING:
            for a in AGENCY.get(r["type"], []):
                if a not in agencies:
                    agencies.append(a)
    person_txt = []
    for g in persons:
        a = g["tracks"][0]
        d = [ROLE_KO.get(a.get("role"), "")]
        if a.get("color"):
            d.append(f"상의 {a['color']} 계열")
        if a.get("weapon"):
            d.append(a["weapon"])
        person_txt.append(", ".join(x for x in d if x))
    when = datetime.fromisoformat(inc["start_time"])
    script = (f"CCTV 관제센터입니다. {when:%H시 %M분}경 {loc['address'] or loc['name']} "
              f"({loc['name']})에서 {inc['type_ko']} 상황이 CCTV로 확인되었습니다. "
              + (f"관련 인원 {len(persons)}명" + (f"({'; '.join(person_txt)})" if person_txt else "") + ". ")
              + (f"위험물: {', '.join(inc['objects'])}. " if inc["objects"] else "")
              + (f"이동 방향: {', '.join(directions)}. " if directions else "")
              + "영상 확보되어 있습니다.")
    return {
        "incident_id": inc["id"],
        "status": "DRAFT",
        "generated_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "fields": {
            "occurred_at": inc["start_time"],
            "location": {"name": loc["name"], "address": loc["address"], "lat": loc["lat"],
                         "lon": loc["lon"], "camera": loc["camera"]},
            "incident_type": inc["type_ko"],
            "risk_level": inc["peak_level"],
            "dangerous_objects": inc["objects"] or ["확인되지 않음"],
            "persons": {"count": len(persons), "descriptions": person_txt},
            "behaviors": behaviors,
            "movement_direction": directions or ["확인되지 않음"],
            "related_cctv": [{"camera": c, "name": inc["location"]["name"] if c == loc["camera"] else
                              next((r["name"] for r in inc["related_cameras"] if r["camera"] == c), c)}
                             for c in inc["cameras"]]
                            + [{"camera": r["camera"], "name": r["name"], "note": r["reason"]}
                               for r in inc["related_cameras"]
                               if r["camera"] not in inc["cameras"] and r["status"] != "nearby"],
            "video": [{"camera": c, "url": f"{base_url}/media/{v['path']}", "status": v["status"],
                       "range_s": [v["start_ts"], v["end_ts"]]} for c, v in inc["clips"].items()],
            "ai_analysis": {
                "summary": inc.get("summary") or template_summary(inc),
                "peak_score": inc["peak_score"],
                "situations": [{"type": r["type_ko"], "camera": r["camera"], "peak_level": r["peak_level"],
                                "peak_score": r["peak_score"],
                                "evidence": [e["text"] for e in r["evidence"]]}
                               for r in inc["situations"].values()],
                "policy": policy_provenance or {},
            },
            "recommended_agency": agencies or ["관제자 판단"],
        },
        "call_script": script,
        "operator": {"name": "", "verified": False, "memo": "", "agency": "", "receipt_no": "",
                     "confirmed_at": None, "reported_at": None},
        "disclaimer": "본 시스템은 자동 신고하지 않습니다. 관제자가 내용을 확인·수정한 뒤 직접 신고합니다.",
    }
