import cv2
import mediapipe as mp
import numpy as np
import csv

# 각도 계산 함수
def calculate_angle(a, b, c):
    a = np.array(a)
    b = np.array(b)
    c = np.array(c)

    ba = a - b
    bc = c - b

    cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
    angle = np.arccos(np.clip(cosine_angle, -1.0, 1.0))
    return np.degrees(angle)

# 랜드마크 픽셀 위치 추출
def get_point(lm, idx, shape):
    h, w = shape[:2]
    return int(lm[idx].x * w), int(lm[idx].y * h)

# 초기화
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils
cap = cv2.VideoCapture('climbing_video.mov')  # ← 분석할 영상 경로

fps = cap.get(cv2.CAP_PROP_FPS)

# 각도 누적 저장용
angle_data = {
    'Right Elbow': [],
    'Left Elbow': [],
    'Right Knee': [],
    'Left Knee': []
}

with mp_pose.Pose() as pose:
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        shape = frame.shape
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(frame_rgb)

        if results.pose_landmarks:
            lm = results.pose_landmarks.landmark

            angle_points = {
                'Right Elbow': [mp_pose.PoseLandmark.RIGHT_SHOULDER.value,
                                mp_pose.PoseLandmark.RIGHT_ELBOW.value,
                                mp_pose.PoseLandmark.RIGHT_WRIST.value],
                'Left Elbow': [mp_pose.PoseLandmark.LEFT_SHOULDER.value,
                               mp_pose.PoseLandmark.LEFT_ELBOW.value,
                               mp_pose.PoseLandmark.LEFT_WRIST.value],
                'Right Knee': [mp_pose.PoseLandmark.RIGHT_HIP.value,
                               mp_pose.PoseLandmark.RIGHT_KNEE.value,
                               mp_pose.PoseLandmark.RIGHT_ANKLE.value],
                'Left Knee': [mp_pose.PoseLandmark.LEFT_HIP.value,
                              mp_pose.PoseLandmark.LEFT_KNEE.value,
                              mp_pose.PoseLandmark.LEFT_ANKLE.value]
            }

            # 관절 각도 계산 및 텍스트 표시
            x, y = 10, 30
            for name, indices in angle_points.items():
                a = get_point(lm, indices[0], shape)
                b = get_point(lm, indices[1], shape)
                c = get_point(lm, indices[2], shape)

                angle = calculate_angle(a, b, c)
                angle_data[name].append(angle)

                cv2.putText(frame, f'{name}: {angle:.2f}deg', (x, y),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2, cv2.LINE_AA)
                y += 30

                # 각도 기준이 되는 관절 점 표시 (파란 점)
                for point in [a, b, c]:
                    cv2.circle(frame, point, 5, (255, 0, 0), -1)

            # MediaPipe 스켈레톤 시각화
            mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

        # 실시간 영상 출력
        cv2.imshow('Real-Time Joint Angles with Skeleton', frame)
        if cv2.waitKey(int(1000 / fps)) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()

# 평균 각도 결과 CSV 저장
csv_filename = 'climbing_video_angle_result.csv'
with open(csv_filename, mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(['joint_name', 'mean_angle'])

    print("\n📊 평균 관절 각도:")
    for name, angles in angle_data.items():
        if angles:
            mean_angle = np.mean(angles)
            writer.writerow([name, f'{mean_angle:.2f}'])
            print(f"{name}: {mean_angle:.2f}°")
        else:
            writer.writerow([name, 'No Data'])
            print(f"{name}: 데이터 없음")

print(f"\n✅ 평균 각도 CSV 저장 완료: {csv_filename}")
