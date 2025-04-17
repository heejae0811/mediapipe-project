import cv2
import mediapipe as mp
import numpy as np

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

# 랜드마크 픽셀 좌표로 변환
def get_point(lm, idx, image_shape):
    h, w = image_shape[:2]
    return int(lm[idx].x * w), int(lm[idx].y * h)

# MediaPipe 초기화
mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

# 이미지 경로
image_path = 'climbing_img.png'  # <- 경로 확인 필요
image = cv2.imread(image_path)

if image is None:
    raise FileNotFoundError(f"이미지를 불러올 수 없습니다: {image_path}")

image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# 포즈 인식 시작
with mp_pose.Pose(static_image_mode=True) as pose:
    results = pose.process(image_rgb)

    if results.pose_landmarks:
        lm = results.pose_landmarks.landmark
        img_shape = image.shape

        # 관절 각도 계산할 위치 설정
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

        # 각도 저장
        angle_info = []
        for name, indices in angle_points.items():
            a = get_point(lm, indices[0], img_shape)
            b = get_point(lm, indices[1], img_shape)
            c = get_point(lm, indices[2], img_shape)

            angle = calculate_angle(a, b, c)
            angle_info.append(f'{name}: {int(angle)} deg')

        # 📦 투명 박스 그리기
        num_lines = len(angle_info)
        dy = 25  # 줄 간격
        x, y_start = 10, image.shape[0] - (num_lines * dy + 20)
        box_width, box_height = 300, num_lines * dy + 20

        # 하얀 배경 (투명도 50%)
        overlay = image.copy()
        cv2.rectangle(overlay, (x - 5, y_start - 5), (x + box_width, y_start + box_height), (255, 255, 255), -1)
        alpha = 0.5
        image = cv2.addWeighted(overlay, alpha, image, 1 - alpha, 0)

        # 🔤 검정 글씨로 출력
        for i, text in enumerate(angle_info):
            y = y_start + (i + 1) * dy
            cv2.putText(image, text, (x, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1, cv2.LINE_AA)

        # 💡 포즈 라인 시각화도 함께
        mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

        # 💾 이미지 저장
        save_path = 'climbing_img_angle_result.png'
        cv2.imwrite(save_path, image)
        print(f"✅ 결과 이미지 저장 완료: {save_path}")

    else:
        print("⚠️ 사람 포즈를 인식하지 못했습니다.")
