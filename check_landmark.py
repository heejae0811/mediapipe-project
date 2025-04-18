import cv2
import numpy as np
import pandas as pd
import mediapipe as mp

# 파일
VIDEO_PATH = "./videos/climbing20.mov"
OUTPUT_CSV = "joint_visibility20.csv"
NUM_LANDMARKS = 33

# Mediapipe 초기화
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=False, min_detection_confidence=0.5)

# 비디오 열기
cap = cv2.VideoCapture(VIDEO_PATH)
visibility_list = []
frame_idx = 0
SAMPLE_RATE = 5  # 프레임 샘플링 (메모리 절약)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    if frame_idx % SAMPLE_RATE == 0:
        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(image_rgb)
        if results.pose_landmarks:
            frame_visibility = []
            for lm in results.pose_landmarks.landmark:
                frame_visibility.append(lm.visibility)
            visibility_list.append(frame_visibility)
    frame_idx += 1

cap.release()
pose.close()

# 관절 visibility 계산
visibility_array = np.array(visibility_list)  # (프레임 수, 33)
mean_visibility = np.mean(visibility_array, axis=0)

# 관절 이름
landmark_names = [
    "nose", "left_eye_inner", "left_eye", "left_eye_outer", "right_eye_inner", "right_eye", "right_eye_outer",
    "left_ear", "right_ear", "mouth_left", "mouth_right",
    "left_shoulder", "right_shoulder", "left_elbow", "right_elbow", "left_wrist", "right_wrist",
    "left_pinky", "right_pinky", "left_index", "right_index", "left_thumb", "right_thumb",
    "left_hip", "right_hip", "left_knee", "right_knee", "left_ankle", "right_ankle",
    "left_heel", "right_heel", "left_foot_index", "right_foot_index"
]

# 결과 정리
df_visibility = pd.DataFrame({
    "Landmark_Index": list(range(NUM_LANDMARKS)),
    "Landmark_Name": landmark_names,
    "Mean_Visibility": mean_visibility
}).sort_values(by="Mean_Visibility", ascending=False)

print("\n전체 관절 인식률 순위 (Mean Visibility)\n")
print(df_visibility.to_string(index=False))  # 전체 출력

# CSV 저장
df_visibility.to_csv(OUTPUT_CSV, index=False, encoding="utf-8-sig")
print(f"\n관절 visibility 분석 완료: \"{OUTPUT_CSV}\" 저장")
