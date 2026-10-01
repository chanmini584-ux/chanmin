# 0001. 프로토타입 기술 스택 (2026-10-01)

검증 환경: Linux x86_64, Python 3.11.15, CPU 4코어 / 15GB RAM, **GPU 없음**.
네트워크: PyPI·GitHub 릴리스 접근 가능, HuggingFace·OpenMMLab 다운로드 서버 차단 → 모델 선택에 영향.

| 구성요소 | 버전 | 선택 이유 | 대안과 비교 | 호환성 / 이 버전인 이유 |
|---|---|---|---|---|
| Python | 3.11.15 | 환경 기본, 주요 ML 패키지 휠 제공 | 3.12/3.13 | `requires-python >=3.10` |
| numpy | 2.2.6 | 수치 연산 | – | opencv-python 4.12가 numpy<2.3 범위로 해석되어 pip가 2.2.6 선택. `numpy>=2.0,<2.3`으로 고정 |
| opencv-python | 4.12.0.88 | 영상 디코딩(FFmpeg 백엔드: 파일·RTSP), 그리기, 색상 히스토그램 | 5.0.0.93(최신, 메이저 변경) / headless | 5.0은 메이저 업데이트라 Ultralytics 호환 미검증. Ultralytics가 `opencv-python`(GUI판)을 요구하므로 headless와 혼용 시 `cv2` 충돌 → GUI판으로 통일. 4.13.0.90은 Ultralytics가 제외 |
| FastAPI | 0.142.2 (Starlette 1.7.0) | REST + WebSocket + 정적 파일을 한 프로세스에서, 타입 기반 API | Flask(+Socket.IO), Django | lifespan API 사용(`on_event` 대체). TestClient 경고(httpx2 권장)는 기능 영향 없음 |
| uvicorn[standard] | 0.54.0 | ASGI 서버, WebSocket 지원 | hypercorn | – |
| PyYAML | 6.0.3 | 사이트·정책 설정 | TOML | 비개발자도 정책 파일 수정 가능 |
| imageio-ffmpeg | 0.6.0 (내장 ffmpeg 7.0.2-static) | 사건 클립을 **H.264 MP4**로 인코딩(브라우저 재생). 시스템 ffmpeg 불필요 | OpenCV VideoWriter(mp4v는 브라우저 재생 불가), PyAV | `-movflags +faststart`, yuv420p, 짝수 해상도(macro_block_size=2) |
| SQLite (stdlib) | 3.x | 사건 문서·감사 로그 저장, 설치 불필요 | PostgreSQL | 다중 관제석·대규모 운영 시 PostgreSQL로 교체 |
| 대시보드 | 빌드 없는 HTML/CSS/JS | 의존성 0, 프로토타입 수정 속도 | React/Vue | 화면이 커지면 컴포넌트 프레임워크로 이전 |
| 실시간 영상 | MJPEG (multipart) | 브라우저 기본 지원, 분석 결과 오버레이 프레임 그대로 전송 | WebRTC, HLS | 다수 카메라·원격 관제 시 WebRTC(예: MediaMTX 연계)로 교체 |
| **인식 백엔드 (선택)** | ultralytics 8.4.171, torch 2.14.1, torchvision 0.29.1, 가중치 YOLO11n / YOLO11n-pose (assets v8.3.0) | pose + 추적(ByteTrack/BoT-SORT 내장) + COCO 객체(knife, baseball bat, scissors)를 한 패키지로 → 가장 빠르게 **실제 영상 end-to-end** 검증. 가중치를 GitHub에서 받을 수 있음 | RF-DETR(Apache-2.0, HF 호스팅 → 이 환경에서 다운로드 불가), RTMPose(Apache-2.0, OpenMMLab 서버 차단), YOLOX(Apache-2.0, pose 없음) | **AGPL-3.0**: 네트워크 서비스로 배포 시 소스 공개 의무. 그래서 `perception/yolo.py` 한 파일에만 import하고 선택 의존성(`[vision]`)으로 분리. 상용화 결정 시 RF-DETR + RTMPose로 교체(인터페이스 동일). PyTorch CPU 전용 인덱스가 차단되어 PyPI 기본 빌드(CUDA 포함) 사용 — CPU에서 정상 동작 |
| pytest / httpx | 9.1.1 / 0.28.1 | 테스트, API TestClient | – | – |
| MediaMTX (테스트 전용) | v1.15.0 | RTSP 서버 역할로 IP 카메라 모사 → RTSP 입력 검증 | ffmpeg `-rtsp_flags listen`(입력 전용이라 불가) | 저장소에 포함하지 않음 |

## 결정 사항
- 탐지기 라이선스(조사 문서 결정 008)는 **프로토타입 한정으로 Ultralytics 사용, 어댑터로 격리**. 상용화 여부가 정해지면 다시 결정한다.
- 시뮬레이션 백엔드를 핵심 경로로 둔다 — GPU·모델·데이터 없이도 Risk Engine·사건 관리·대시보드·평가를 결정적으로 시험할 수 있고, 정답(GT)이 정확하다.
