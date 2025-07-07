import numpy as np
import pandas as pd
import mediapipe as mp
import cv2
import os

# 설정
LABEL = 0  # 비숙련자 0 / 숙련자 1
VIDEO_PATH = "./videos/climbing33_1.mov"
FILE_ID = os.path.splitext(os.path.basename(VIDEO_PATH))[0]
OUTPUT_CSV = f"./csv_features/features_{FILE_ID}.csv"
FRAME_INTERVAL = 1

# Mediapipe Pose 초기화
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=False, min_detection_confidence=0.5)

# 비디오 열기
cap = cv2.VideoCapture(VIDEO_PATH)
if not cap.isOpened():
    raise IOError(f"Cannot open video: {VIDEO_PATH}")

actual_fps = cap.get(cv2.CAP_PROP_FPS)
frame_idx = 0

# landmark 좌표 저장
trajectory = {i: {'x': [], 'y': []} for i in range(33)}  # 33개 관절

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
            for i, lm in enumerate(landmarks):
                trajectory[i]['x'].append(lm.x)
                trajectory[i]['y'].append(lm.y)

    frame_idx += 1

cap.release()
pose.close()

# 거리, 속도, 벡터속도(velocity), 가속도, jerk 계산
def compute_motion_metrics(x_list, y_list, fps):
    x = np.array(x_list)
    y = np.array(y_list)
    dt = 1 / fps

    if len(x) < 4:
        return [np.nan] * 15  # 5종류 × 3통계량

    coords = np.stack([x, y], axis=1)
    dist = np.linalg.norm(np.diff(coords, axis=0), axis=1)  # 거리

    speed = np.gradient(dist, dt)  # 스칼라 속도
    velocity_vectors = np.diff(coords, axis=0) / dt  # 벡터 속도
    velocity = np.linalg.norm(velocity_vectors, axis=1)  # 벡터 크기

    accel = np.gradient(speed, dt)
    jerk = np.gradient(accel, dt)

    # 각 metric에 대해 min, max, mean 계산
    metrics = {
        'dist': dist,
        'speed': speed,
        'velocity': velocity,
        'accel': accel,
        'jerk': jerk
    }

    stats = []
    for name, arr in metrics.items():
        stats.extend([np.min(arr), np.max(arr), np.mean(arr)])

    return stats

# 데이터 정리
row_data = {"id": FILE_ID, "label": LABEL}

for idx in range(33):
    (
        dist_min, dist_max, dist_mean,
        speed_min, speed_max, speed_mean,
        velocity_min, velocity_max, velocity_mean,
        accel_min, accel_max, accel_mean,
        jerk_min, jerk_max, jerk_mean
    ) = compute_motion_metrics(trajectory[idx]["x"], trajectory[idx]["y"], actual_fps)

    row_data[f"landmark{idx}_dist_min"] = dist_min
    row_data[f"landmark{idx}_dist_max"] = dist_max
    row_data[f"landmark{idx}_dist_mean"] = dist_mean

    row_data[f"landmark{idx}_speed_min"] = speed_min
    row_data[f"landmark{idx}_speed_max"] = speed_max
    row_data[f"landmark{idx}_speed_mean"] = speed_mean

    row_data[f"landmark{idx}_velocity_min"] = velocity_min
    row_data[f"landmark{idx}_velocity_max"] = velocity_max
    row_data[f"landmark{idx}_velocity_mean"] = velocity_mean

    row_data[f"landmark{idx}_accel_min"] = accel_min
    row_data[f"landmark{idx}_accel_max"] = accel_max
    row_data[f"landmark{idx}_accel_mean"] = accel_mean

    row_data[f"landmark{idx}_jerk_min"] = jerk_min
    row_data[f"landmark{idx}_jerk_max"] = jerk_max
    row_data[f"landmark{idx}_jerk_mean"] = jerk_mean

# CSV 저장
motion_df = pd.DataFrame([row_data])
motion_df.to_csv(OUTPUT_CSV, index=False, encoding="utf-8-sig")

print(f"✅ Feature 추출 완료: '{OUTPUT_CSV}' 저장됨")
