from pathlib import Path

import cv2
import imageio_ffmpeg
import numpy as np
import pytest
from fastapi.testclient import TestClient

from safewatch.ingest.source import VideoSource
from safewatch.pipeline import run_scenario
from safewatch.server.app import Runtime, create_app
from safewatch.sim import library


@pytest.fixture(scope="module")
def client(tmp_path_factory):
    d = tmp_path_factory.mktemp("live")
    rt = Runtime(data_dir=d, eval_dir=d / "eval")
    rt.manager.encode_background = False
    run_scenario(library.get("collapse_alone"), rt.site, rt.policy, rt.manager, seed=1)
    rt.manager.finish()
    with TestClient(create_app(rt)) as c:
        yield c


def test_site_and_scenarios(client):
    s = client.get("/api/site").json()
    assert {c["id"] for c in s["cameras"]} >= {"cam01", "cam02", "cam03"}
    assert s["policy"]["provenance"]["source"] in ("hypothesis", "fitted")
    names = {x["name"] for x in client.get("/api/scenarios").json()}
    assert "knife_threat_chase" in names


def test_incident_review_and_report_flow(client):
    incs = client.get("/api/incidents").json()
    assert incs, "scenario should have produced an incident"
    iid = incs[0]["id"]
    d = client.get(f"/api/incidents/{iid}").json()
    assert d["type"] == "FALL" and "AI 사건 요약" in d["summary"]
    assert client.post(f"/api/incidents/{iid}/view", json={}).status_code == 200
    k = client.post(f"/api/incidents/{iid}/ack", json={"operator": "tester"}).json()
    assert k["understanding_s"] is not None
    rep = client.get(f"/api/incidents/{iid}/report").json()
    assert rep["status"] == "DRAFT" and rep["fields"]["incident_type"] == "쓰러짐"
    assert "119" in rep["fields"]["recommended_agency"]
    # recording a report before confirmation is refused
    assert client.post(f"/api/incidents/{iid}/report/reported", json={}).status_code == 400
    # confirmation requires the operator's explicit video verification
    assert client.post(f"/api/incidents/{iid}/report/confirm", json={"name": "tester"}).status_code == 400
    client.put(f"/api/incidents/{iid}/report", json={"operator": {"memo": "확인함"}, "call_script": "수정 문안"})
    r = client.post(f"/api/incidents/{iid}/report/confirm", json={"name": "tester", "verified": True}).json()
    assert r["status"] == "CONFIRMED" and r["call_script"] == "수정 문안"
    r = client.post(f"/api/incidents/{iid}/report/reported", json={"agency": "119", "receipt_no": "A-1"}).json()
    assert r["status"] == "REPORTED"
    txt = client.get(f"/api/incidents/{iid}/report.txt").text
    assert "신고 지원 정보" in txt and "자동 신고하지 않습니다" in txt
    assert client.post(f"/api/incidents/{iid}/feedback", json={"label": "TP"}).status_code == 200
    kpi = client.get("/api/kpi").json()
    row = next(r for r in kpi["rows"] if r["id"] == iid)
    assert row["report_status"] == "REPORTED" and row["total_response_s"] is not None
    assert kpi["feedback"]["operator_precision"] == 1.0
    audit = client.get(f"/api/incidents/{iid}").json()["audit"]
    assert [a["action"] for a in audit][:2] == ["view", "ack"]


def test_media_and_eval_endpoints(client):
    iid = client.get("/api/incidents").json()[0]["id"]
    d = client.get(f"/api/incidents/{iid}").json()
    clip = d["clips"]["cam01"]
    assert clip["status"] == "ready"
    r = client.get(f"/media/{clip['path']}")
    assert r.status_code == 200 and r.headers["content-type"] == "video/mp4"
    assert client.get("/api/eval/latest").json()["available"] is False
    assert client.get("/").status_code == 200


def test_video_source_subsamples_file(tmp_path):
    path = tmp_path / "v.mp4"
    gen = imageio_ffmpeg.write_frames(str(path), (64, 48), fps=25, codec="libx264", pix_fmt_out="yuv420p",
                                      macro_block_size=2)
    gen.send(None)
    for i in range(50):
        gen.send(np.full((48, 64, 3), i * 5, np.uint8))
    gen.close()
    frames = list(VideoSource(str(path), analysis_fps=5, realtime=False).frames())
    assert 9 <= len(frames) <= 11
    assert frames[1][1] == pytest.approx(0.2, abs=0.01)
