import cv2
import mediapipe as mp

# 설정
VIDEO_PATH = "./videos_all/04__1_250804.mov"

# 영상 열기
cap = cv2.VideoCapture(VIDEO_PATH)

# FPS, 프레임 크기 설정
fps = cap.get(cv2.CAP_PROP_FPS)
if fps <= 1 or fps > 120 or fps is None:
    fps = 30.0

# Mediapipe 초기화
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils
pose = mp_pose.Pose(static_image_mode=False, model_complexity=2, min_detection_confidence=0.5)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # 프레임을 RGB로 변환
    image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = pose.process(image_rgb)

    # 관절이 인식되면 랜드마크 그리기
    if results.pose_landmarks:
        mp_drawing.draw_landmarks(
            frame,
            results.pose_landmarks,
            mp_pose.POSE_CONNECTIONS
        )

    # 실시간 화면 출력
    cv2.imshow("Pose Landmarks - Mediapipe", frame)

    # q 키를 누르면 종료
    if cv2.waitKey(int(1000 / fps)) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
pose.close()
