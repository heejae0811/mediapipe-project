# 영상의 모든 관절(33개) 인식
import cv2
import mediapipe as mp

mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

# 영상 경로 (세로 영상 사용)
video_path = "climbing01.mov"
cap = cv2.VideoCapture(video_path)

with mp_pose.Pose(
    static_image_mode = False,
    model_complexity = 1,
    enable_segmentation = False,
    min_detection_confidence = 0.5
) as pose:

    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            break

        # 세로 영상이 눕는 경우 → 회전해서 세로 유지
        frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)

        # 전처리
        frame.flags.writeable = False
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(image)

        # 후처리 및 landmark 그리기
        frame.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        if results.pose_landmarks:
            mp_drawing.draw_landmarks(
                image,
                results.pose_landmarks,
                mp_pose.POSE_CONNECTIONS
            )

        # 결과 출력
        cv2.imshow('MediaPipe Pose (Vertical)', image)

        if cv2.waitKey(5) & 0xFF == 27:  # ESC
            break

cap.release()
cv2.destroyAllWindows()