import cv2
import mediapipe as mp
import numpy as np
import csv

# MediaPipe pose 초기화
mp_pose = mp.solutions.pose

# 비디오 열기
cap = cv2.VideoCapture('climbing_video.mov')  # ← 영상 파일 경로 입력

# 기본 정보
fps = cap.get(cv2.CAP_PROP_FPS)
w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# 변수 초기화
prev_pos = None
points = []
total_distance = 0
frame_count = 0

# 포즈 추정 시작
with mp_pose.Pose() as pose:
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(frame_rgb)

        frame_count += 1
        elapsed_time = frame_count / fps

        if results.pose_landmarks:
            lm = results.pose_landmarks.landmark
            left_hip = np.array([lm[mp_pose.PoseLandmark.LEFT_HIP.value].x * w,
                                 lm[mp_pose.PoseLandmark.LEFT_HIP.value].y * h])
            right_hip = np.array([lm[mp_pose.PoseLandmark.RIGHT_HIP.value].x * w,
                                  lm[mp_pose.PoseLandmark.RIGHT_HIP.value].y * h])
            mid_hip = (left_hip + right_hip) / 2

            # 골반 위치 표시 (빨간 점)
            cv2.circle(frame, tuple(mid_hip.astype(int)), 6, (0, 0, 255), -1)

            # 이동 거리 계산
            if prev_pos is not None:
                dist = np.linalg.norm(mid_hip - prev_pos)
                total_distance += dist
                points.append((tuple(prev_pos.astype(int)), tuple(mid_hip.astype(int))))
            prev_pos = mid_hip

        # 이동 경로 그리기 (흰 선)
        for p1, p2 in points:
            cv2.line(frame, p1, p2, (255, 255, 255), 2)

        # 하단 정보 표시 (하얀 반투명 박스 + 텍스트)
        text = f'Distance: {total_distance:.2f}px   Time: {elapsed_time:.2f}s'
        overlay = frame.copy()
        text_bg_height = 40
        cv2.rectangle(overlay, (0, h - text_bg_height), (w, h), (255, 255, 255), -1)
        alpha = 0.5
        frame = cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0)

        cv2.putText(frame, text, (10, h - 12), cv2.FONT_HERSHEY_SIMPLEX,
                    0.7, (0, 0, 0), 2, cv2.LINE_AA)

        # 실시간 보기
        cv2.imshow('Pelvis Tracking Viewer', frame)
        if cv2.waitKey(int(1000 / fps)) & 0xFF == ord('q'):
            break

# 자원 해제
cap.release()
cv2.destroyAllWindows()

# 결과 요약 출력
print(f"\n📊 최종 이동 거리: {total_distance:.2f} px")
print(f"⏱️ 총 소요 시간: {elapsed_time:.2f} 초")

# CSV로 저장
csv_filename = 'climbing_video_tracking_result.csv'
with open(csv_filename, mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(['total_distance_px', 'elapsed_time_sec'])
    writer.writerow([f'{total_distance:.2f}', f'{elapsed_time:.2f}'])

print(f'✅ CSV 저장 완료: {csv_filename}')
