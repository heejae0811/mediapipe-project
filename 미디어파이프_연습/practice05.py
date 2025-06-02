# 골반 움직임 트레킹
import cv2
import mediapipe as mp

mp_pose = mp.solutions.pose

# 영상 열기
cap = cv2.VideoCapture("../videos/climbing01.mov")

# 골반 이동 경로를 저장할 리스트
hip_trail = []

with mp_pose.Pose(
    static_image_mode = False, # False: 영상 처리 / True: 이미지 처리
    model_complexity = 2, # 모델의 정확도 수준: 0 빠르지만 덜 정확, 1 보통, 2 느리지만 정확
    enable_segmentation = False, # False: 관절만 추적 / True: 관절 + 몸 영역 분리
    min_detection_confidence = 0.7, # Pose 모델이 사람이라고 판단하는 최소 신뢰도
    min_tracking_confidence = 0.7 # 추적 성공 여부 판단 기준
) as pose:

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # 세로 영상 회전 (필요 시)
        frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)

        # 관절 인식
        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(image_rgb)

        if results.pose_landmarks:
            lm = results.pose_landmarks.landmark
            left_hip = lm[mp_pose.PoseLandmark.LEFT_HIP]
            right_hip = lm[mp_pose.PoseLandmark.RIGHT_HIP]

            h, w, _ = frame.shape
            cx = int((left_hip.x + right_hip.x) / 2 * w)
            cy = int((left_hip.y + right_hip.y) / 2 * h)

            # 현재 위치 점 추가
            hip_trail.append((cx, cy))

            # 경로를 선으로 연결
            for i in range(1, len(hip_trail)):
                cv2.line(frame, hip_trail[i - 1], hip_trail[i], (0, 255, 0), 2)

            # 현재 위치에 점 찍기
            cv2.circle(frame, (cx, cy), 6, (0, 0, 255), -1)  # 빨간 점

        # 영상 출력
        cv2.imshow("Hip Movement Trail", frame)

        if cv2.waitKey(5) & 0xFF == 27:
            break

cap.release()
cv2.destroyAllWindows()