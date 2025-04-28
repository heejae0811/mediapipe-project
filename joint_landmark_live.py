import cv2
import mediapipe as mp

# 설정
VIDEO_PATH = "./videos/climbing10.mov"

# Mediapipe 초기화
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils
pose = mp_pose.Pose(static_image_mode=False, min_detection_confidence=0.5)

# 영상 열기
cap = cv2.VideoCapture(VIDEO_PATH)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # 회전 보정 (필요한 경우만 사용)
    frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)

    image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = pose.process(image_rgb)

    if results.pose_landmarks:
        mp_drawing.draw_landmarks(
            frame,
            results.pose_landmarks,
            mp_pose.POSE_CONNECTIONS
        )

    cv2.imshow("Pose Landmarks - Mediapipe Default", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
pose.close()
