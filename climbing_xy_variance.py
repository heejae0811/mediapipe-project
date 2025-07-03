import cv2
import mediapipe as mp
import pandas as pd

video_path = './videos/climbing20.mov'
xy_path = './csv_xy/xy20.csv'
variability_path = './csv_variability/variability20.csv'

# MediaPipe Pose 설정
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(
    static_image_mode=False,
    model_complexity=1,
    min_detection_confidence=0.5
)

# 비디오 캡처 초기화
cap = cv2.VideoCapture(video_path)

landmarks_data = []
frame_idx = 0

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # BGR → RGB
    image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = pose.process(image_rgb)

    # 관절 랜드마크 추출
    if results.pose_landmarks:
        for idx, lm in enumerate(results.pose_landmarks.landmark):
            landmarks_data.append({
                'frame': frame_idx,
                'landmark_index': idx,
                'x': lm.x,
                'y': lm.y,
                'z': lm.z,
                'visibility': lm.visibility
            })

    frame_idx += 1

cap.release()
pose.close()

# DataFrame 생성
df = pd.DataFrame(landmarks_data)

# landmark_index 별로 x, y, z의 표준편차 계산
variability = df.groupby('landmark_index').agg(
    x_variability=('x', 'std'),
    y_variability=('y', 'std'),
    z_variability=('z', 'std')
).reset_index()

# CSV 저장
df.to_csv(xy_path, index=False)
variability.to_csv(variability_path, index=False)

# 터미널 출력
print("=== Landmark Variability Preview ===")
print(variability.head(10))
