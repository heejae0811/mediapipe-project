# 양쪽 어깨와 골반 landmark + 프레임 skip
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

# 사각형 선 연결 순서
rectangle_connections = [
    (mp_pose.PoseLandmark.LEFT_SHOULDER, mp_pose.PoseLandmark.RIGHT_SHOULDER),
    (mp_pose.PoseLandmark.RIGHT_SHOULDER, mp_pose.PoseLandmark.RIGHT_HIP),
    (mp_pose.PoseLandmark.RIGHT_HIP, mp_pose.PoseLandmark.LEFT_HIP),
    (mp_pose.PoseLandmark.LEFT_HIP, mp_pose.PoseLandmark.LEFT_SHOULDER)
]

# 영상 열기
cap = cv2.VideoCapture("videos/climbing02.mov")

# 프레임 스킵 설정
frame_skip = 2
frame_count = 0

# 이전 프레임의 landmark 좌표 저장용
prev_landmark_coords = {}

with mp_pose.Pose(
    static_image_mode=False,
    model_complexity=1,  # 정확도: 0(빠름), 1(보통), 2(정확)
    enable_segmentation=False,
    min_detection_confidence=0.5
) as pose:

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
        frame_count += 1

        h, w, _ = frame.shape
        landmark_coords = {}

        if frame_count % frame_skip == 0:
            # Pose 인식 수행
            image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = pose.process(image_rgb)

            if results.pose_landmarks:
                for landmark_id in target_landmarks:
                    lm = results.pose_landmarks.landmark[landmark_id]
                    cx, cy = int(lm.x * w), int(lm.y * h)
                    landmark_coords[landmark_id] = (cx, cy)
                prev_landmark_coords = landmark_coords.copy()
            else:
                # 인식 실패 → 이전 좌표 유지
                landmark_coords = prev_landmark_coords.copy()
        else:
            # 스킵된 프레임 → 이전 좌표 사용
            landmark_coords = prev_landmark_coords.copy()

        # 점 + 사각형 그리기
        for landmark_id in landmark_coords:
            cx, cy = landmark_coords[landmark_id]
            cv2.circle(frame, (cx, cy), 6, (0, 255, 0), -1)

        for start, end in rectangle_connections:
            if start in landmark_coords and end in landmark_coords:
                cv2.line(frame, landmark_coords[start], landmark_coords[end], (0, 255, 255), 2)

        # 출력
        cv2.imshow("Pose Rectangle: Shoulders & Hips", frame)
        if cv2.waitKey(5) & 0xFF == 27:
            break

cap.release()
cv2.destroyAllWindows()
