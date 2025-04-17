# 양쪽 어깨와 골반 landmark
import cv2
import mediapipe as mp

mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

# 추적할 관절
target_landmarks = {
    mp_pose.PoseLandmark.LEFT_SHOULDER,
    mp_pose.PoseLandmark.RIGHT_SHOULDER,
    mp_pose.PoseLandmark.LEFT_HIP,
    mp_pose.PoseLandmark.RIGHT_HIP
}

# 사각형을 구성할 선 연결 순서
rectangle_connections = [
    (mp_pose.PoseLandmark.LEFT_SHOULDER, mp_pose.PoseLandmark.RIGHT_SHOULDER),
    (mp_pose.PoseLandmark.RIGHT_SHOULDER, mp_pose.PoseLandmark.RIGHT_HIP),
    (mp_pose.PoseLandmark.RIGHT_HIP, mp_pose.PoseLandmark.LEFT_HIP),
    (mp_pose.PoseLandmark.LEFT_HIP, mp_pose.PoseLandmark.LEFT_SHOULDER)
]

# 영상 열기
cap = cv2.VideoCapture("videos/climbing02.mov")

with mp_pose.Pose(
    static_image_mode = False,
    model_complexity = 1,
    enable_segmentation = False,
    min_detection_confidence = 0.5
) as pose:

    while cap.isOpened(): # 영상이 열려 있는 동안 한 프레임씩 읽음, 프레임이 없거나 끝나면 종료
        ret, frame = cap.read()
        if not ret:
            break

        # 세로 영상 회전 (필요 시)
        #frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)

        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(image_rgb)
        image_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)

        # 사람이 인식되어 landmarks 정보가 있을 때 실행
        if results.pose_landmarks:
            h, w, _ = image_bgr.shape
            landmark_coords = {}

            # 점 그리기 + 좌표 저장
            for landmark_id in target_landmarks:
                lm = results.pose_landmarks.landmark[landmark_id]
                cx, cy = int(lm.x * w), int(lm.y * h)
                landmark_coords[landmark_id] = (cx, cy)
                cv2.circle(image_bgr, (cx, cy), 6, (0, 255, 0), -1)

            # 사각형 선 그리기
            for start, end in rectangle_connections:
                if start in landmark_coords and end in landmark_coords:
                    cv2.line(
                        image_bgr,
                        landmark_coords[start],
                        landmark_coords[end],
                        (0, 255, 255),  # 노란색
                        2
                    )

        cv2.imshow("Pose Rectangle: Shoulders & Hips", image_bgr)

        if cv2.waitKey(5) & 0xFF == 27:
            break

cap.release()
cv2.destroyAllWindows()