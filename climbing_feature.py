import numpy as np
import pandas as pd
import mediapipe as mp
import cv2
import os

# 설정
LABEL = 1  # 초기영상 0 / 최근영상 1
VIDEO_PATH = "./videos/47_전해빈_1_250708.mov"
FILE_ID = os.path.splitext(os.path.basename(VIDEO_PATH))[0]
OUTPUT_CSV = f"./csv_features/{FILE_ID}.csv"
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

# landmark 좌표 + visibility 저장
trajectory = {i: {'x': [], 'y': [], 'visibility': []} for i in range(33)}

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)

    if frame_idx % FRAME_INTERVAL == 0:
        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(image_rgb)

        if results.pose_landmarks:
            for i, lm in enumerate(results.pose_landmarks.landmark):
                trajectory[i]['x'].append(lm.x)
                trajectory[i]['y'].append(lm.y)
                trajectory[i]['visibility'].append(lm.visibility)

    frame_idx += 1

cap.release()
pose.close()

# 지표 계산 함수
def compute_motion_metrics(x_list, y_list, visibility_list, fps):
    x = np.array(x_list)
    y = np.array(y_list)
    vis = np.array(visibility_list)
    dt = 1 / fps
    T = len(x) * dt

    if len(x) < 4:
        return [np.nan] * 5  # 가속도 3개 + normalized jerk + 평균 visibility

    coords = np.stack([x, y], axis=1)
    dist = np.linalg.norm(np.diff(coords, axis=0), axis=1)
    total_dist = np.sum(dist)

    speed = np.gradient(dist, dt)
    accel = np.gradient(speed, dt)
    jerk = np.gradient(accel, dt)

    # 최대/최소/평균 가속도
    accel_max = np.max(accel)
    accel_min = np.min(accel)
    accel_mean = np.mean(accel)

    # normalized jerk
    jerk_squared_sum = np.sum(jerk**2) * dt
    normalized_jerk = jerk_squared_sum / (T**2 * total_dist**2) if T > 0 and total_dist > 0 else np.nan

    # 평균 visibility
    visibility_mean = np.mean(vis)

    return [accel_max, accel_min, accel_mean, normalized_jerk, visibility_mean]

# 데이터 정리
row_data = {"id": FILE_ID, "label": LABEL}

for idx in range(33):
    accel_max, accel_min, accel_mean, normalized_jerk, visibility_mean = compute_motion_metrics(
        trajectory[idx]["x"], trajectory[idx]["y"], trajectory[idx]["visibility"], actual_fps
    )

    row_data[f"landmark{idx}_accel_max"] = accel_max
    row_data[f"landmark{idx}_accel_min"] = accel_min
    row_data[f"landmark{idx}_accel_mean"] = accel_mean
    row_data[f"landmark{idx}_normalized_jerk"] = normalized_jerk
    row_data[f"landmark{idx}_visibility_mean"] = visibility_mean

# CSV 저장
os.makedirs(os.path.dirname(OUTPUT_CSV), exist_ok=True)
motion_df = pd.DataFrame([row_data])
motion_df.to_csv(OUTPUT_CSV, index=False, encoding="utf-8-sig")

print(f"✅ Feature 추출 완료: '{OUTPUT_CSV}' 저장")
