import os
import cv2
import mediapipe as mp
import pandas as pd
import numpy as np

VIDEO_PATH = "./videos/climbing30.mov"
OUTPUT_CSV = "./results_xy/joint_xy30.csv"
LABEL = 1 # 라벨 직접 입력 (0=비숙련자 / 1=숙련자)

# 저장 경로 없으면 생성
os.makedirs(os.path.dirname(OUTPUT_CSV), exist_ok=True)

mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=False, min_detection_confidence=0.5)
cap = cv2.VideoCapture(VIDEO_PATH)

landmark_names = [lm.name for lm in mp_pose.PoseLandmark]
all_landmark_rows = []
frame_idx = 0

prev_landmarks = {}

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = pose.process(image_rgb)

    if results.pose_landmarks:
        for idx, lm in enumerate(results.pose_landmarks.landmark):
            prev = prev_landmarks.get(idx, None)
            if prev:
                dx = lm.x - prev[0]
                dy = lm.y - prev[1]
                dd = np.sqrt(dx ** 2 + dy ** 2)
            else:
                dx = dy = dd = 0.0

            all_landmark_rows.append({
                "Label": LABEL,  # 👉 여기서 라벨값을 넣어줍니다
                "Frame": frame_idx,
                "Landmark_Index": idx,
                "X": lm.x,
                "Y": lm.y,
                "Delta_X": dx,
                "Delta_Y": dy,
                "Delta_Distance": dd,
                "Visibility": lm.visibility
            })
            prev_landmarks[idx] = (lm.x, lm.y)
    frame_idx += 1

cap.release()
pose.close()

# 컬럼 순서 지정
desired_columns = ["Label", "Frame", "Landmark_Index", "X", "Y", "Delta_X", "Delta_Y", "Delta_Distance", "Visibility"]
df = pd.DataFrame(all_landmark_rows)[desired_columns]

df.to_csv(OUTPUT_CSV, index=False, encoding="utf-8-sig")

print(f"33개 관절 x/y 위치 및 이동량(정규화) csv 저장 완료: {OUTPUT_CSV}")
