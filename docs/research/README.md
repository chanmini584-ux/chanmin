# 사전 조사 v1 — AI 기반 위험상황 탐지 및 CCTV 관제 지원

| 문서 | 내용 |
|---|---|
| [01_prior_art.md](01_prior_art.md) | 정부·지자체 사업, 국내외 상용 제품, 연구, 특허, 규제 |
| [02_gap_analysis.md](02_gap_analysis.md) | 해결된 기능 / 어려운 문제 / 차별화 가설 D1~D6 |
| [03_feasibility.md](03_feasibility.md) | CCTV 연동, 모델 후보, 데이터, GPU, 위험 요소 |
| [04_project_definition_v1.md](04_project_definition_v1.md) | 범위, Risk Engine 평가 구조, MVP, 아키텍처, 단계별 Gate |
| [sources.md](sources.md) | 출처 (원문 재확인 필요 표시) |

## 결정 기록 (Decision Log)

| # | 결정 | 근거 | 상태 |
|---|---|---|---|
| 001 | "객체+행동+맥락 종합" 개념 자체는 차별점으로 주장하지 않음 | Ambient.ai Context Graph, 서울시 VLM 관제 선례 | 확정 |
| 002 | 얼굴인식·신원식별·감정인식·오디오·자동신고 제외 | 개인정보보호법 §25, EU AI Act, 프로젝트 원칙 | 확정 |
| 003 | 흉기 탐지는 단독 판정 근거로 쓰지 않음 | 소형객체 성능 한계, 유사물체 오탐 | 확정 |
| 004 | 주 평가지표 = 이벤트 수준 P/R/F1 + FP/카메라·시간 + 탐지지연 | KISA 개편·감사원 오탐 지적 | 확정 |
| 005 | 위험 단계 경계는 PR 운영점 기반 정책 설정값, 코드 하드코딩 금지 | 사용자 요구, 검증 가능성 | 확정 |
| 006 | VLM은 상위 이벤트 검증·요약에만 사용 | 비용·환각, Cerberus 캐스케이드 근거 | 가설(Phase 5 검증) |
| 007 | 라이브러리 정확한 버전은 Phase 0에서 환경 확인 후 고정 | 근거 없는 고정 방지 | 대기 |
| 008 | 탐지기 라이선스(Apache vs AGPL) | 프로토타입은 Ultralytics(AGPL)를 어댑터 1개 파일로 격리해 사용 ([0001](../decisions/0001-tech-stack.md)). 상용화 여부 결정 시 재결정 | 잠정 (사용자 결정 필요) |
