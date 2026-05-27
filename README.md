# 🍼 AI 아기 수면 안전 모니터링 시스템

> 실시간 카메라와 마이크를 활용해 아기의 수면 중 위험 상황을 자동으로 감지하는 AI 모니터링 시스템

---

## 📌 프로젝트 개요

아기 수면 중 발생할 수 있는 다양한 위험 상황을 AI로 실시간 감지하여 부모에게 즉각 알림을 제공합니다.

- **개발 기간**: 4일
- **개발 환경**: Python 3.11, Windows
- **사용 기술**: OpenCV, MediaPipe, YAMNet, TensorFlow

---

## 🔍 감지 기능 5가지

| 기능 | 방법 | 위험 판단 기준 |
|------|------|--------------|
| 엎드림 감지 | MediaPipe Pose 키포인트 | 어깨/코 가시성 낮음 + z축 반전 |
| 얼굴 가려짐 감지 | MediaPipe Face Detection | 3초 이상 얼굴 미감지 |
| 움직임 없음 감지 | 프레임 차분 (Frame Diff) | 20초 이상 미동 |
| 낙상 감지 | MediaPipe Pose 소실 감지 | 감지되던 아기가 2초 이상 사라짐 |
| 울음소리 감지 | YAMNet 오디오 AI 모델 | 2초 이상 울음 지속 |

---

## 🛠️ 기술 스택

```
Python 3.11
├── OpenCV            → 웹캠 입력 + 화면 출력 + 움직임 감지
├── MediaPipe Pose    → 신체 33개 키포인트 추출 (엎드림/낙상)
├── MediaPipe Face    → 얼굴 감지 (이불 가려짐)
├── YAMNet            → 구글 오디오 AI 모델 (울음소리 분류)
├── TensorFlow Hub    → YAMNet 모델 로드
└── sounddevice       → 마이크 입력
```

---

## 💻 설치 방법

### 1. Python 3.11 설치
- [python.org](https://www.python.org/downloads/release/python-3119/) 에서 다운로드
- 설치 시 **"Add to PATH"** 체크 필수

### 2. 가상환경 생성
```powershell
python -m venv venv
venv\Scripts\Activate.ps1
```

### 3. 라이브러리 설치
```powershell
pip install opencv-python mediapipe numpy tensorflow tensorflow-hub sounddevice
```

---

## 🚀 실행 방법

```powershell
python baby_monitor.py
---

## 📁 프로젝트 구조

```
baby_monitor/
├── venv/                    # 가상환경
├── baby_monitor.py          # 메인 코드
├── pose_landmarker.task     # MediaPipe Pose 모델 (자동 다운로드)
├── face_detector.tflite     # MediaPipe Face 모델 (자동 다운로드)
└── README.md                # 프로젝트 설명
```

---

## ⚙️ 설정값 조정

`baby_monitor.py` 상단에서 민감도 조정 가능:

```python
FACE_MISSING_THRESHOLD = 3.0   # 얼굴 미감지 위험 판단 시간 (초)
NO_MOTION_THRESHOLD   = 20.0   # 움직임 없음 위험 판단 시간 (초)
MOTION_SENSITIVITY    = 800    # 움직임 감도 (낮을수록 민감)
PRONE_Z_THRESHOLD     = 0.1    # 엎드림 판단 민감도
FALL_THRESHOLD        = 2.0    # 낙상 판단 시간 (초)
CRY_THRESHOLD         = 3.0    # 울음 지속 위험 판단 시간 (초)
```

---

## 🖥️ 화면 구성

```
┌─────────────────────────────────────┐
│  LIVE  AI Baby Monitor   timer...   │  ← 상태바
├─────────────────────────────────────┤
│  🔴 PRONE DETECTED                  │  ← 위험 알림
│  🔴 FACE COVERED                    │
│                                     │
│     [ 웹캠 화면 + 키포인트 ]         │
│                                     │
└─────────────────────────────────────┘
```

---

## 🤖 AI 모델 설명

### MediaPipe Pose
- 구글이 만든 신체 자세 감지 모델
- 몸에서 **33개 키포인트** 추출
- 코/어깨 위치로 엎드림 판단
- 라이선스: Apache 2.0 (무료)

### MediaPipe Face Detection
- 구글이 만든 얼굴 감지 모델
- 얼굴 미감지 시간으로 이불 가려짐 판단
- 라이선스: Apache 2.0 (무료)

### YAMNet
- 구글이 만든 오디오 분류 AI 모델
- **521가지 소리** 분류 가능
- 18번 인덱스: Baby cry (아기 울음)
- 라이선스: Apache 2.0 (무료)

---

## ⚠️ 한계점

- 실제 배포용이 아닌 **프로토타입 수준**
- 야간/저조도 환경에서 감지율 저하
- 투명한 이불은 감지 어려움
- 마이크 주변 소음에 영향받을 수 있음

---

## 🔮 향후 계획

- [ ] 위험 구역 진입 감지 (주방, 베란다 ROI 설정)
- [ ] 모바일 푸시 알림 연동
- [ ] 야간 모드 (IR 카메라 지원)
- [ ] 온도/습도 센서 연동
- [ ] 감지 로그 저장 (CSV)
- [ ] 대시보드 UI 개발
