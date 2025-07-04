import numpy as np
import pandas as pd
import mediapipe as mp
import cv2
import os

# 설정
LABEL = 0  # 비숙련자 0 / 숙련자 1
VIDEO_PATH = "./videos/climbing20.mov"
FILE_ID = os.path.splitext(os.path.basename(VIDEO_PATH))[0]
OUTPUT_CSV = f"./csv_jerk/features_{FILE_ID}.csv"
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


# Jerk 계산
def compute_jerk(x_list, y_list, fps):
    x = np.array(x_list)
    y = np.array(y_list)
    dt = 1 / fps

    if len(x) < 4:  # 최소 4프레임 이상이어야 jerk를 계산할 수 있음
        return np.nan, np.nan, np.nan

    dx = np.diff(x)
    dy = np.diff(y)
    vx = dx / dt
    vy = dy / dt
    ax = np.diff(vx) / dt
    ay = np.diff(vy) / dt
    jx = np.diff(ax) / dt
    jy = np.diff(ay) / dt
    jerk = np.sqrt(jx**2 + jy**2)

    return np.min(jerk), np.max(jerk), np.mean(jerk)


# 데이터 정리
row_data = {"id": FILE_ID, "label": LABEL}

for idx in range(33):
    j_min, j_max, j_mean = compute_jerk(
        trajectory[idx]["x"], trajectory[idx]["y"], actual_fps
    )
    row_data[f"landmark{idx}_jerk_min"]  = j_min
    row_data[f"landmark{idx}_jerk_max"]  = j_max
    row_data[f"landmark{idx}_jerk_mean"] = j_mean

# CSV 저장
motion_df = pd.DataFrame([row_data])
motion_df.to_csv(OUTPUT_CSV, index=False, encoding="utf-8-sig")

print(f"✅ Feature 추출 완료: '{OUTPUT_CSV}' 저장됨")
