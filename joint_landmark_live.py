import cv2
import mediapipe as mp
import os

# 설정
VIDEO_PATH = "./videos/joint_live02.mov"
OUTPUT_PATH = "joint_live02.mp4"

# 영상 열기
cap = cv2.VideoCapture(VIDEO_PATH)

# FPS, 프레임 크기 얻기 (예외 처리 포함)
fps = cap.get(cv2.CAP_PROP_FPS)
if fps <= 1 or fps > 120 or fps is None:
    fps = 30.0
frame_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
if frame_w == 0 or frame_h == 0:
    frame_w, frame_h = 640, 480

# mp4(H.264) 코덱 사용
fourcc = cv2.VideoWriter_fourcc(*'avc1')
out = cv2.VideoWriter(OUTPUT_PATH, fourcc, fps, (frame_w, frame_h))

# Mediapipe 초기화
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils
pose = mp_pose.Pose(static_image_mode=False, min_detection_confidence=0.5)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # 프레임 크기 보정 (혹시라도 크기 오류 방지)
    frame = cv2.resize(frame, (frame_w, frame_h))

    # 회전 보정 필요 시 아래 코드 사용
    # frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)

    image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = pose.process(image_rgb)

    if results.pose_landmarks:
        mp_drawing.draw_landmarks(
            frame,
            results.pose_landmarks,
            mp_pose.POSE_CONNECTIONS
        )

    cv2.imshow("Pose Landmarks - Mediapipe Default", frame)
    out.write(frame)

    if cv2.waitKey(int(1000 / fps)) & 0xFF == ord('q'):
        break

cap.release()
out.release()
cv2.destroyAllWindows()
pose.close()

print(f"영상 저장 완료: {os.path.abspath(OUTPUT_PATH)}")
