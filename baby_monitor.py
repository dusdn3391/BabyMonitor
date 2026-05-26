import cv2
import numpy as np
import time
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision
from mediapipe.tasks.python.components import containers
import urllib.request
import os
import sounddevice as sd
import tensorflow_hub as hub
import threading  

# ── 모델 다운로드 ──────────────────────────────────────
POSE_MODEL_URL  = "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/1/pose_landmarker_lite.task"
FACE_MODEL_URL  = "https://storage.googleapis.com/mediapipe-models/face_detector/blaze_face_short_range/float16/1/blaze_face_short_range.tflite"
POSE_MODEL_PATH = "pose_landmarker.task"
FACE_MODEL_PATH = "face_detector.tflite"

def download_model(url, path):
    if not os.path.exists(path):
        print(f"모델 다운로드 중: {path}")
        urllib.request.urlretrieve(url, path)
        print(f"완료: {path}")

download_model(POSE_MODEL_URL, POSE_MODEL_PATH)
download_model(FACE_MODEL_URL, FACE_MODEL_PATH)

# ── 설정값 ─────────────────────────────────────────────
FACE_MISSING_THRESHOLD = 3.0
NO_MOTION_THRESHOLD   = 20.0
MOTION_SENSITIVITY    = 800
PRONE_Z_THRESHOLD     = 0.1
FALL_THRESHOLD        = 2.0
CRY_THRESHOLD         = 2.0

# ── 모델 초기화 ────────────────────────────────────────
BaseOptions = mp_python.BaseOptions
PoseLandmarker = vision.PoseLandmarker
PoseLandmarkerOptions = vision.PoseLandmarkerOptions
FaceDetector = vision.FaceDetector
FaceDetectorOptions = vision.FaceDetectorOptions
VisionRunningMode = vision.RunningMode

yamnet_model = hub.load("https://tfhub.dev/google/yamnet/1")


pose_options = PoseLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=POSE_MODEL_PATH),
    running_mode=VisionRunningMode.IMAGE
)
face_options = FaceDetectorOptions(
    base_options=BaseOptions(model_asset_path=FACE_MODEL_PATH),
    running_mode=VisionRunningMode.IMAGE
)

pose_landmarker = PoseLandmarker.create_from_options(pose_options)
face_detector   = FaceDetector.create_from_options(face_options)

# ── 상태 변수 ──────────────────────────────────────────
class BabyState:
    def __init__(self):
        self.face_last_seen     = time.time()
        self.last_motion_time   = time.time()
        self.prev_gray          = None
        self.face_missing_sec   = 0.0
        self.no_motion_sec      = 0.0
        self.baby_last_seen     = time.time()
        self.baby_ever_detected = False
        self.fall_missing_sec   = 0.0
        self.baby_cry_start     = time.time()
        self.baby_crying_seen   = False
        self.baby_crying_sec    = 0.0

state = BabyState()

cry_detected_flag = False

# ✅ 울음 감지 스레드 함수
def cry_detection_loop():
    global cry_detected_flag
    
    BASELINE_SAMPLES = []
    
    while True:
        audio_data = sd.rec(int(1.0 * 16000), samplerate=16000, channels=1, dtype='float32')
        sd.wait()
        
        volume = np.abs(audio_data).mean()
        
        # ── 1단계: 기준치 측정 (처음 10번) ──
        if len(BASELINE_SAMPLES) < 10:
            BASELINE_SAMPLES.append(volume)
            cry_detected_flag = False
            continue
        
        baseline = np.mean(BASELINE_SAMPLES)
        ratio = volume / (baseline + 1e-9)
        
        # ── 2단계: 볼륨이 작으면 YAMNet 스킵 ──
        if ratio < 1.8:
            cry_detected_flag = False
            continue
        
        # ── 3단계: 볼륨이 크면 YAMNet으로 정밀 분석 ──
        audio_input = audio_data.flatten()
        scores, _, _ = yamnet_model(audio_input)
        scores_np = scores.numpy()
        mean_scores = np.mean(scores_np, axis=0)
        
        # 19번(Crying) + 20번(Baby cry) 둘 다 체크
        cry_score = max(mean_scores[19], mean_scores[20])
        
        cry_detected_flag = cry_score > 0.15
        
# ── 1. 엎드림 감지 ─────────────────────────────────────
def detect_prone(landmarks):
    nose           = landmarks[0]
    left_shoulder  = landmarks[11]
    right_shoulder = landmarks[12]

    shoulder_vis = (left_shoulder.visibility + right_shoulder.visibility) / 2
    if shoulder_vis < 0.4 and nose.visibility < 0.4:
        return True
    avg_shoulder_z = (left_shoulder.z + right_shoulder.z) / 2
    if nose.z - avg_shoulder_z > 0.1:
        return True
    return False

# ── 2. 얼굴 가려짐 감지 ───────────────────────────────
def detect_face_covered(face_results, current_time):
    if face_results.detections:
        state.face_last_seen   = current_time
        state.face_missing_sec = 0.0
        return False
    else:
        state.face_missing_sec = current_time - state.face_last_seen
        return state.face_missing_sec >= FACE_MISSING_THRESHOLD

# ── 3. 움직임 없음 감지 ───────────────────────────────
def detect_no_motion(gray_frame, current_time):
    if state.prev_gray is None:
        state.prev_gray = gray_frame
        return False
    diff      = cv2.absdiff(state.prev_gray, gray_frame)
    _, thresh = cv2.threshold(diff, 25, 255, cv2.THRESH_BINARY)
    motion_px = np.sum(thresh > 0)
    state.prev_gray = gray_frame
    if motion_px > MOTION_SENSITIVITY:
        state.last_motion_time = current_time
        state.no_motion_sec    = 0.0
        return False
    else:
        state.no_motion_sec = current_time - state.last_motion_time
        return state.no_motion_sec >= NO_MOTION_THRESHOLD

# ── 4. 떨어짐 감지 ────────────────────────────────────
def detect_fall(pose_result, current_time):
    if pose_result.pose_landmarks:
        state.baby_last_seen   = current_time
        state.fall_missing_sec = 0.0
        state.baby_ever_detected = True
        return False
    else:
        if state.baby_ever_detected:
            state.fall_missing_sec = current_time - state.baby_last_seen
            return state.fall_missing_sec >= FALL_THRESHOLD
        return False

# ── 알림 오버레이 ──────────────────────────────────────
def draw_alerts(frame, alerts, face_missing_sec, no_motion_sec):
    h, w = frame.shape[:2]
    cv2.rectangle(frame, (0, 0), (w, 48), (30, 30, 30), -1)
    status_text  = "  LIVE  AI Baby Monitor"
    status_color = (0, 220, 100)
    if alerts:
        status_text  = "  WARNING!"
        status_color = (0, 0, 220)
    cv2.putText(frame, status_text, (12, 32),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, status_color, 2)
    timer = f"face missing: {face_missing_sec:.1f}s  |  no motion: {no_motion_sec:.1f}s"
    cv2.putText(frame, timer, (w - 430, 32),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180, 180, 180), 1)
    colors = {
        "PRONE DETECTED": (0, 0, 220),
        "FACE COVERED":   (0, 0, 220),
        "NO MOTION":      (0, 180, 220),
        "FALL":           (0, 0, 220),
        "BABY CRYING":    (0, 140, 255),
    }
    for i, alert in enumerate(alerts):
        color = colors.get(alert, (0, 0, 200))
        y = 80 + i * 50
        cv2.rectangle(frame, (10, y - 28), (360, y + 10), color, -1)
        cv2.putText(frame, alert, (18, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2)
    if alerts:
        cv2.rectangle(frame, (0, 0), (w - 1, h - 1), (0, 0, 220), 4)
    return frame

# ── 키포인트 그리기 ────────────────────────────────────
def draw_landmarks(frame, landmarks, w, h):
    CONNECTIONS = [
        (0,1),(1,2),(2,3),(3,7),(0,4),(4,5),(5,6),(6,8),
        (11,12),(11,13),(13,15),(12,14),(14,16),
        (11,23),(12,24),(23,24),(23,25),(24,26),(25,27),(26,28)
    ]
    pts = {}
    for i, lm in enumerate(landmarks):
        cx, cy = int(lm.x * w), int(lm.y * h)
        pts[i] = (cx, cy)
        cv2.circle(frame, (cx, cy), 4, (0, 255, 0), -1)
    for a, b in CONNECTIONS:
        if a in pts and b in pts:
            cv2.line(frame, pts[a], pts[b], (0, 200, 200), 2)

# ── 메인 루프 ──────────────────────────────────────────
def main():
    # ✅ 울음 감지 스레드 시작
    cry_thread = threading.Thread(target=cry_detection_loop, daemon=True)
    cry_thread.start()

    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    print("=" * 50)
    print("  AI 아기 수면 안전 모니터링 시작")
    print("  종료: q 키")
    print("=" * 50)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        current_time = time.time()
        h, w = frame.shape[:2]

        rgb_frame  = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray_frame = cv2.GaussianBlur(gray_frame, (21, 21), 0)

        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)

        pose_result = pose_landmarker.detect(mp_image)
        face_result = face_detector.detect(mp_image)

        alerts = []

        # 1. 자세 감지
        if pose_result.pose_landmarks:
            landmarks = pose_result.pose_landmarks[0]
            if detect_prone(landmarks):
                alerts.append("PRONE DETECTED")
            draw_landmarks(frame, landmarks, w, h)

        # 2. 얼굴 가려짐
        if detect_face_covered(face_result, current_time):
            alerts.append("FACE COVERED")

        # 얼굴 바운딩박스
        if face_result.detections:
            for det in face_result.detections:
                bb = det.bounding_box
                cv2.rectangle(frame,
                    (bb.origin_x, bb.origin_y),
                    (bb.origin_x + bb.width, bb.origin_y + bb.height),
                    (0, 255, 0), 2)

        # 3. 움직임 없음
        if detect_no_motion(gray_frame, current_time):
            alerts.append("NO MOTION")

        # 4. 떨어짐
        if detect_fall(pose_result, current_time):
            alerts.append("FALL")

        # ✅ 5. 울음 감지 (스레드 플래그만 확인)
        if cry_detected_flag:
            if not state.baby_crying_seen:
                state.baby_cry_start = current_time
            state.baby_crying_seen = True
            state.baby_crying_sec = current_time - state.baby_cry_start
            if state.baby_crying_sec >= CRY_THRESHOLD:
                alerts.append("BABY CRYING")
        else:
            state.baby_crying_seen = False
            state.baby_crying_sec  = 0.0

        frame = draw_alerts(frame, alerts, state.face_missing_sec, state.no_motion_sec)

        if alerts:
            print(f"[{time.strftime('%H:%M:%S')}] {' | '.join(alerts)}")

        cv2.imshow("AI Baby Safety Monitor", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    pose_landmarker.close()
    face_detector.close()

if __name__ == "__main__":
    main()