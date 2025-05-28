import numpy as np
import pandas as pd
import mediapipe as mp
import cv2
import os

# 설정
VIDEO_PATH = "./videos/climbing30.mov"
FILE_ID = os.path.splitext(os.path.basename(VIDEO_PATH))[0]
LABEL = 1  # 비숙련자(0) / 숙련자(1)
OUTPUT_CSV = f"features_{FILE_ID}.csv"

TARGET_IDX = {
    "nose": 0,
    "left_hip": 23,
    "right_hip": 24
}

FRAME_INTERVAL = 1

# Mediapipe Pose 초기화
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=False, min_detection_confidence=0.5)

# 비디오 열기
cap = cv2.VideoCapture(VIDEO_PATH)
if not cap.isOpened():
    raise IOError(f"Cannot open video: {VIDEO_PATH}")

actual_fps = cap.get(cv2.CAP_PROP_FPS)
print(f"Actual FPS: {actual_fps}")

frame_idx = 0

# landmark 좌표 저장
trajectory = {"nose": {"x": [], "y": []}, "pelvis": {"x": [], "y": []}}

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)

    if frame_idx % FRAME_INTERVAL == 0:
        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(image_rgb)

        if results.pose_landmarks:
            landmarks = results.pose_landmarks.landmark
            nose = landmarks[TARGET_IDX["nose"]]
            left_hip = landmarks[TARGET_IDX["left_hip"]]
            right_hip = landmarks[TARGET_IDX["right_hip"]]

            pelvis_x = (left_hip.x + right_hip.x) / 2
            pelvis_y = (left_hip.y + right_hip.y) / 2

            trajectory["nose"]["x"].append(nose.x)
            trajectory["nose"]["y"].append(nose.y)
            trajectory["pelvis"]["x"].append(pelvis_x)
            trajectory["pelvis"]["y"].append(pelvis_y)

    frame_idx += 1

cap.release()
pose.close()

# 운동 특성 계산 함수
def compute_metrics(x_list, y_list, fps):
    x = np.array(x_list)
    y = np.array(y_list)
    dt = 1 / fps

    dx = np.diff(x)
    dy = np.diff(y)
    distance = np.sqrt(dx**2 + dy**2)
    vx = dx / dt
    vy = dy / dt
    velocity = np.sqrt(vx**2 + vy**2)
    ax = np.diff(vx) / dt
    ay = np.diff(vy) / dt
    acceleration = np.sqrt(ax**2 + ay**2)
    jx = np.diff(ax) / dt
    jy = np.diff(ay) / dt
    jerk = np.sqrt(jx**2 + jy**2)

    return {
        "time": len(x) * dt,
        "distance": np.sum(distance),
        "speed_mean": np.mean(distance / dt),
        "speed_max": np.max(distance / dt),
        "velocity_mean": np.mean(velocity),
        "velocity_max": np.max(velocity),
        "acceleration_mean": np.mean(acceleration),
        "acceleration_max": np.max(acceleration),
        "jerk_mean": np.mean(jerk),
        "jerk_max": np.max(jerk)
    }

# 데이터 정리
row_data = {"id": FILE_ID, "label": LABEL}
for name in ["nose", "pelvis"]:
    metrics = compute_metrics(trajectory[name]["x"], trajectory[name]["y"], actual_fps)
    for k, v in metrics.items():
        row_data[f"{name}_{k}"] = v

# CSV 저장
motion_df = pd.DataFrame([row_data])
motion_df.to_csv(OUTPUT_CSV, index=False, encoding="utf-8-sig")

print(f"Feature 추출 완료: '{OUTPUT_CSV}' 저장됨")
