import cv2
import numpy as np
import pandas as pd
import mediapipe as mp

# 설정
VIDEO_PATH = "./videos/climbing01.mov"  # 사용할 비디오 파일 경로
OUTPUT_CSV = "joint_visibility01.csv"  # 저장할 CSV 파일명
FRAME_INTERVAL = 3  # 3프레임마다 분석

# Mediapipe Pose 초기화
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=False, min_detection_confidence=0.5)

# 비디오 열기
cap = cv2.VideoCapture(VIDEO_PATH)

# visibility 저장용 리스트
visibility_list = []
frame_idx = 0

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    if frame_idx % FRAME_INTERVAL == 0:
        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(image_rgb)

        if results.pose_landmarks:
            frame_vis = [lm.visibility for lm in results.pose_landmarks.landmark]
            visibility_list.append(frame_vis)

    frame_idx += 1

cap.release()
pose.close()

# numpy 배열로 변환 (프레임 수, 33)
visibility_array = np.array(visibility_list)

# 관절별 평균 visibility 계산
mean_visibility = np.mean(visibility_array, axis=0)

# 관절 이름 가져오기 (Mediapipe PoseLandmark 활용)
landmark_names = [mp_pose.PoseLandmark(i).name.lower() for i in range(33)]

# 결과 정리
df_visibility = pd.DataFrame({
    "Landmark_Index": list(range(33)),
    "Landmark_Name": landmark_names,
    "Mean_Visibility": mean_visibility
}).sort_values(by="Mean_Visibility", ascending=False)

# 터미널 출력
print("\n전체 영상 기준 관절 인식률 Mean Visibility 순")
print(df_visibility.to_string(index=False))

# CSV로 저장
df_visibility.to_csv(OUTPUT_CSV, index=False, encoding="utf-8-sig")
print(f"\n관절 visibility 분석 결과가 '{OUTPUT_CSV}' 파일로 저장되었습니다.")