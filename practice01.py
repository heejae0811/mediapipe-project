# MediaPipe + OpenCV를 사용해서 웹캠으로 실시간 관절 인식
import cv2
import mediapipe as mp

mp_drawing = mp.solutions.drawing_utils # mp_drawing: 관절 점과 선을 화면에 그려주는 도구
mp_pose = mp.solutions.pose # mp_pose: 포즈 추적(관절 인식) 모듈

# 웹캠 열기, 0은 기본 내장 웹캠
cap = cv2.VideoCapture(0)

with mp_pose.Pose(
    static_image_mode = False, # 실시간 영상 처리
    model_complexity = 1, # 기본 정확도 (0: 빠름, 2: 정확하지만 느림)
    enable_segmentation = False, # 사람 분리 기능 안 씀
    min_detection_confidence = 0.5 # 관절 인식 최소 신뢰도
) as pose:

    while cap.isOpened():
        success, image = cap.read()
        if not success:
            print("카메라 프레임을 읽을 수 없습니다.")
            continue

        # 성능 향상을 위해 이미지 쓰기 불가로 설정
        image.flags.writeable = False
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = pose.process(image)

        # 이미지에 결과 그리기
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        if results.pose_landmarks: # 관절이 인식되면 영상 위에 landmarks과 연결선 그림
            mp_drawing.draw_landmarks(
                image,
                results.pose_landmarks,
                mp_pose.POSE_CONNECTIONS
            )

        # 화면에 출력
        cv2.imshow('MediaPipe Pose', image)

        if cv2.waitKey(5) & 0xFF == 27:  # ESC 눌러 종료
            break

cap.release()
cv2.destroyAllWindows()