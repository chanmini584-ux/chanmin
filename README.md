# SafeWatch — AI 기반 위험상황 탐지 및 CCTV 관제 지원 시스템 (프로토타입)

기존 CCTV 영상에서 **객체 + 행동 + 시간 흐름 + 공간 맥락 + 사람 간 관계 + 복수 CCTV**를 종합해
"지금 무슨 상황이 일어나고 있는가"를 판단하고, 관제자가 상황을 이해하고 신고를 준비하는 시간을 줄이도록 돕는 시스템입니다.

> AI는 신고하지 않습니다. 위험상황을 탐지하고 근거·영상·신고 정보를 정리하며, **최종 판단과 신고는 관제자가 합니다.**
> 얼굴인식·신원식별·"위험인물" 예측은 하지 않습니다.

```
CCTV/NVR/RTSP ─▶ 인식(탐지·추적·자세) ─▶ 특징(개인·쌍·구역) ─▶ Risk Engine(증거 누적·반대증거·정책 단계)
   ─▶ 사건 관리(병합·타임라인·전후 영상·주변 CCTV·동일인 후보) ─▶ 관제 대시보드(알림·영상·근거)
   ─▶ AI 사건 요약 ─▶ 신고 지원 패키지 ─▶ 관제자 확인·확정 ─▶ 사람이 신고 ─▶ KPI·피드백 기록
```

## 빠른 시작

```bash
python3 -m venv .venv && . .venv/bin/activate
pip install -e ".[dev]"            # 핵심 기능 (합성 시나리오, 대시보드, 평가) — GPU·모델 불필요
safewatch demo                     # http://127.0.0.1:8000 대시보드 + 데모 시나리오 자동 재생
```

| 명령 | 용도 |
|---|---|
| `safewatch demo [--play 시나리오...]` | 대시보드 실행 + 시나리오 자동 재생 (흉기 위협→추격, 폭행→쓰러짐, 정상 조깅, 단독 쓰러짐) |
| `safewatch serve` | 대시보드만 실행. 화면에서 시나리오 재생 / 영상 파일·RTSP 연결 |
| `safewatch scenarios` | 내장 시나리오 11종 (위험 5 + 정상-유사 6) |
| `safewatch run knife_threat_chase` | 헤드리스 분석 → `runs/…/incidents.json`, 사건 영상(mp4), 신고 패키지 |
| `safewatch eval --seeds 1 2 3 [--noise hard] [--fit-policy out.yaml]` | 사건 단위 P/R/F1, PR·AP, FP/카메라·시간, 탐지 지연 → `runs/eval/report.md` |
| `safewatch analyze --source video.mp4 --camera cam01` | **실제 영상** 분석 (YOLO pose + ByteTrack + COCO 흉기/배트) |
| `safewatch analyze --source rtsp://user:pass@ip:554/... --camera cam01` | **IP 카메라/NVR RTSP** 실시간 분석 (자동 재연결) |
| `safewatch export-sim knife_threat_chase` | 합성 시나리오를 mp4 + 정답 JSON으로 내보내기 |

실제 영상 분석에는 선택 의존성이 필요합니다: `pip install -e ".[vision]"` (Ultralytics, AGPL-3.0 — [기술 결정](docs/decisions/0001-tech-stack.md) 참조).
처음 실행 시 YOLO11n 가중치(약 6MB)를 GitHub에서 내려받습니다. CPU 4코어 기준 약 5 fps/카메라.

## 대시보드에서 할 수 있는 것

- **알림 큐**: 위험도·미확인 우선 정렬, WARNING 이상 팝업/알림음, 진행 중 사건 표시
- **CCTV 화면**: 카메라별 실시간 분석 영상(위험도 색상, 트랙, 위험물 박스, 구역)
- **사건 상세**: AI 요약 · 사건 전후 영상(사건 전 8초 버퍼 포함) · 관련 인원(역할·인상착의·이동방향·위험물) ·
  주변 CCTV 지도(이동 예상 / 동일인 후보 — 관제자 확인 필요) · 타임라인 · 처리 이력
- **판단 근거**: 상황별 관문 증거 / 가중 증거 / 반대 증거(예: "함께 이동하는 일행")와 강도
- **신고 지원**: 발생시간·위치·유형·위험물·인원·행동·이동방향·관련 CCTV·영상·AI 결과 자동 정리 → 관제자 수정 →
  "영상으로 직접 확인" 체크 후 확정 → 관제자가 직접 신고 후 기관·접수번호 기록 (자동 신고 없음)
- **대응 KPI**: 탐지 지연 · 알림 지연 · 사건 인지 · 상황 파악 · 신고 준비 · 전체 대응 시간, 관제자 정탐/오탐 피드백
- **AI 성능 평가**: 최신 평가 리포트(운영점별 지표, PR 곡선, 정상-유사 상황 점수)

## 저장소 구성

```
configs/site.demo.yaml       카메라·위치·구역(출입통제/배회감시)·카메라 토폴로지
configs/policy.default.yaml  Risk Engine 정책 (모든 수치 = 가설값, provenance 기록)
src/safewatch/
  sim/          합성 다중 카메라 CCTV 시뮬레이터 + 시나리오(정답 포함) + 인식 오류 모델
  perception/   실제 인식 백엔드 (YOLO pose + tracker + COCO objects)
  ingest/       영상 파일 / RTSP 입력 (서브샘플링, 재연결)
  features/     트랙별 속도·자세·쓰러짐·타격동작·위험물 보유·구역 체류
  risk/         Risk Engine (상황 평가, 증거, 반대증거, 감쇠) + 정책(보정·단계)
  incidents/    사건 관리, 클립 녹화, 다중 카메라 연계, 인상착의, 요약·신고 패키지
  eval/         사건 단위 평가, PR 운영점 적합, isotonic 보정
  server/       FastAPI + WebSocket + MJPEG + 대시보드(정적 파일)
tests/          단위·시나리오 회귀·API 테스트 (33개)
docs/research/  사전 조사 v1 (선례, Gap, 타당성, 프로젝트 정의)
docs/prototype/ 프로토타입 설계·검증 결과·한계
docs/decisions/ 기술 결정 기록 (버전·선택 이유·대안·호환성)
```

자세한 설계와 검증 결과, 한계는 [docs/prototype/README.md](docs/prototype/README.md)를 참고하세요.
