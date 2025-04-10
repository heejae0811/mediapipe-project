import cv2
import mediapipe as mp
import numpy as np
import pandas as pd

mp_pose = mp.solutions.pose

# 🎥 영상 열기
cap = cv2.VideoCapture("climbing01.mov")

# 🕒 프레임 시간 계산
fps = cap.get(cv2.CAP_PROP_FPS)
dt = 1.0 / fps if fps > 0 else 1.0 / 30

hip_centers = []
timestamps = []

with mp_pose.Pose(
    static_image_mode=False,
    model_complexity=2,
    enable_segmentation=False,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
) as pose:

    frame_idx = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(image_rgb)

        if results.pose_landmarks:
            lm = results.pose_landmarks.landmark
            left_hip = lm[mp_pose.PoseLandmark.LEFT_HIP]
            right_hip = lm[mp_pose.PoseLandmark.RIGHT_HIP]

            cx = (left_hip.x + right_hip.x) / 2
            cy = (left_hip.y + right_hip.y) / 2
            cz = (left_hip.z + right_hip.z) / 2

            hip_centers.append([cx, cy, cz])
            timestamps.append(frame_idx * dt)

        frame_idx += 1

cap.release()

# NumPy 변환
hip_centers = np.array(hip_centers)
timestamps = np.array(timestamps)

# 물리량 계산
velocity = np.gradient(hip_centers, timestamps, axis=0)
acceleration = np.gradient(velocity, timestamps, axis=0)
jerk = np.gradient(acceleration, timestamps, axis=0)

# 크기(norm) 계산
speed = np.linalg.norm(velocity, axis=1)
velocity_norm = np.linalg.norm(velocity, axis=1)
acceleration_norm = np.linalg.norm(acceleration, axis=1)
jerk_norm = np.linalg.norm(jerk, axis=1)

# 통계 계산 함수
def calc_stats(arr, name):
    return {
        f"{name}_min": np.min(arr),
        f"{name}_mean": np.mean(arr),
        f"{name}_max": np.max(arr)
    }

# 요약 결과 저장
summary = {}
summary.update(calc_stats(speed, "speed"))
summary.update(calc_stats(velocity_norm, "velocity"))
summary.update(calc_stats(acceleration_norm, "acceleration"))
summary.update(calc_stats(jerk_norm, "jerk"))

# CSV 저장
df_summary = pd.DataFrame([summary])
df_summary.to_csv("hip_motion_summary.csv", index=False)
print("✅ 요약 완료: hip_motion_summary.csv 저장됨")