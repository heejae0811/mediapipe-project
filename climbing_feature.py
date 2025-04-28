import numpy as np
import pandas as pd
import mediapipe as mp
import cv2
import os

# 설정
VIDEO_PATH = "./videos/climbing01.mov"
FILE_ID = os.path.splitext(os.path.basename(VIDEO_PATH))[0]
LABEL = 0  # 숙련자(1) / 비숙련자(0)
OUTPUT_CSV = f"features_{FILE_ID}.csv"

TARGET_LANDMARKS = {
    "right_hip": 24,
    "left_hip": 23,
    "right_shoulder": 12,
    "left_shoulder": 11,
    "nose": 0
}

FRAME_INTERVAL = 1
FPS = 30

# Mediapipe Pose 초기화
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=False, min_detection_confidence=0.5)

# 비디오 열기
cap = cv2.VideoCapture(VIDEO_PATH)
frame_idx = 0

# landmark 좌표 저장
landmark_trajectories = {name: {"x": [], "y": []} for name in TARGET_LANDMARKS.keys()}

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)

    if frame_idx % FRAME_INTERVAL == 0:
        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(image_rgb)

        if results.pose_landmarks:
            for name, idx in TARGET_LANDMARKS.items():
                lm = results.pose_landmarks.landmark[idx]
                landmark_trajectories[name]["x"].append(lm.x)
                landmark_trajectories[name]["y"].append(lm.y)

    frame_idx += 1

cap.release()
pose.close()

# 계산 함수
def compute_motion_metrics(x_list, y_list, fps):
    x = np.array(x_list)
    y = np.array(y_list)

    dx = np.diff(x)
    dy = np.diff(y)
    distance = np.sqrt(dx**2 + dy**2)

    dt = 1 / fps
    speed = distance / dt
    vx = dx / dt
    vy = dy / dt
    velocity = np.sqrt(vx**2 + vy**2)
    ax = np.diff(vx) / dt
    ay = np.diff(vy) / dt
    acceleration = np.sqrt(ax**2 + ay**2)
    jx = np.diff(ax) / dt
    jy = np.diff(ay) / dt
    jerk = np.sqrt(jx**2 + jy**2)

    total_time = len(x) * dt

    return {
        "total_time": total_time,
        "total_distance": np.sum(distance),
        "speed_min": np.min(speed),
        "speed_max": np.max(speed),
        "speed_mean": np.mean(speed),
        "velocity_min": np.min(velocity),
        "velocity_max": np.max(velocity),
        "velocity_mean": np.mean(velocity),
        "acceleration_min": np.min(acceleration),
        "acceleration_max": np.max(acceleration),
        "acceleration_mean": np.mean(acceleration),
        "jerk_min": np.min(jerk),
        "jerk_max": np.max(jerk),
        "jerk_mean": np.mean(jerk),
        "mean_x_movement": np.mean(np.abs(np.diff(x))),
        "mean_y_movement": np.mean(np.abs(np.diff(y)))
    }

# 결과 저장
row_data = {"id": FILE_ID, "label": LABEL}

for name, coords in landmark_trajectories.items():
    metrics = compute_motion_metrics(coords["x"], coords["y"], FPS)
    for metric_name, value in metrics.items():
        # 이름을 모두 소문자로 통일
        col_name = f"{name}_{metric_name}".lower()
        row_data[col_name] = value

# 최종 데이터프레임 만들기
motion_summary_df = pd.DataFrame([row_data])

# CSV로 저장
motion_summary_df.to_csv(OUTPUT_CSV, index=False, encoding="utf-8-sig")

print(f"\nFeature 추출 완료: '{OUTPUT_CSV}' 파일로 저장했습니다.")
