import cv2
import mediapipe as mp
import numpy as np
import pandas as pd

# 설정
VIDEO_PATH = "./videos/climbing10.mov"
OUTPUT_CSV = "shoulder_hip_xy10.csv"
SAMPLE_RATE = 5  # 5프레임마다 분석

# Mediapipe 초기화
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(
    static_image_mode=False,
    model_complexity=2,
    min_detection_confidence=0.5
)

# 비디오 열기
cap = cv2.VideoCapture(VIDEO_PATH)
fps = cap.get(cv2.CAP_PROP_FPS)
dt = 1.0 / fps if fps > 0 else 1.0 / 30

frame_idx = 0
data = []

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    if frame_idx % SAMPLE_RATE == 0:
        # 회전 (필요시만)
        # frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)

        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(image_rgb)

        if results.pose_landmarks:
            lm = results.pose_landmarks.landmark
            def get_xy(index):
                return np.array([lm[index].x, lm[index].y])

            # 관절 위치 추출 (x, y만)
            l_shoulder = get_xy(mp_pose.PoseLandmark.LEFT_SHOULDER.value)
            r_shoulder = get_xy(mp_pose.PoseLandmark.RIGHT_SHOULDER.value)
            l_hip = get_xy(mp_pose.PoseLandmark.LEFT_HIP.value)
            r_hip = get_xy(mp_pose.PoseLandmark.RIGHT_HIP.value)

            # 거리
            shoulder_width = np.linalg.norm(l_shoulder - r_shoulder)
            hip_width = np.linalg.norm(l_hip - r_hip)
            shoulder_hip_ratio = shoulder_width / hip_width if hip_width != 0 else np.nan

            # 중심 좌표
            shoulder_center = (l_shoulder + r_shoulder) / 2
            hip_center = (l_hip + r_hip) / 2

            data.append({
                "time": frame_idx * dt,
                "shoulder_width": shoulder_width,
                "hip_width": hip_width,
                "shoulder_hip_ratio": shoulder_hip_ratio,
                "shoulder_center_x": shoulder_center[0],
                "shoulder_center_y": shoulder_center[1],
                "hip_center_x": hip_center[0],
                "hip_center_y": hip_center[1]
            })

    frame_idx += 1

cap.release()
pose.close()

# CSV 저장
df = pd.DataFrame(data)
df.to_csv(OUTPUT_CSV, index=False)
print(f"분석 완료: {OUTPUT_CSV} 저장")
