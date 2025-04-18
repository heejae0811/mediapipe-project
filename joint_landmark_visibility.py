import cv2
import numpy as np
import pandas as pd
import mediapipe as mp

# 설정
VIDEO_PATH = "./videos/climbing10.mov"
OUTPUT_CSV = "joint_visibility10.csv"
NUM_LANDMARKS = 33

# 관절 이름 리스트
landmark_names = [
    "nose", "left_eye_inner", "left_eye", "left_eye_outer", "right_eye_inner", "right_eye", "right_eye_outer",
    "left_ear", "right_ear", "mouth_left", "mouth_right",
    "left_shoulder", "right_shoulder", "left_elbow", "right_elbow", "left_wrist", "right_wrist",
    "left_pinky", "right_pinky", "left_index", "right_index", "left_thumb", "right_thumb",
    "left_hip", "right_hip", "left_knee", "right_knee", "left_ankle", "right_ankle",
    "left_heel", "right_heel", "left_foot_index", "right_foot_index"
]

# Mediapipe 초기화
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=False, min_detection_confidence=0.5)

cap = cv2.VideoCapture(VIDEO_PATH)
visibility_list = []

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = pose.process(image_rgb)

    if results.pose_landmarks:
        frame_vis = [lm.visibility for lm in results.pose_landmarks.landmark]
        visibility_list.append(frame_vis)

cap.release()
pose.close()

# 평균 visibility 계산
visibility_array = np.array(visibility_list)  # (프레임 수, 33)
mean_visibility = np.mean(visibility_array, axis=0)

# 결과 정리
df_visibility = pd.DataFrame({
    "Landmark_Index": list(range(NUM_LANDMARKS)),
    "Landmark_Name": landmark_names,
    "Mean_Visibility": mean_visibility
}).sort_values(by="Mean_Visibility", ascending=False)

print("\n전체 영상 기준 관절 인식률 (Mean Visibility 순)")
print(df_visibility.to_string(index=False))

df_visibility.to_csv(OUTPUT_CSV, index=False, encoding="utf-8-sig")
print(f"\n관절 visibility 분석 완료: \"{OUTPUT_CSV}\" 저장")
