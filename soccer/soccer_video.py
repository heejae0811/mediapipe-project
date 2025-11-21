import cv2
import numpy as np
import mediapipe as mp
from enum import Enum


class BodyPart(Enum):
    """신체 부위 옵션"""
    FULL = "full"
    UPPER = "upper"
    LOWER = "lower"


class PoseDetector:
    """MediaPipe Pose를 사용한 관절 감지"""

    def __init__(self):
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(
            static_image_mode=False,
            model_complexity=1,
            smooth_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )

    def get_crop_boundaries(self, frame, body_part):
        """관절 위치 기반으로 크롭 경계 계산"""
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.pose.process(rgb_frame)
        height = frame.shape[0]

        if not results.pose_landmarks:
            # 포즈 감지 실패시 기본값
            if body_part == BodyPart.FULL:
                return 0, height
            elif body_part == BodyPart.UPPER:
                return 0, int(height * 0.55)
            else:
                return int(height * 0.45), height

        landmarks = results.pose_landmarks.landmark

        # 주요 관절 좌표
        nose_y = int(landmarks[self.mp_pose.PoseLandmark.NOSE].y * height)
        hip_y = int((landmarks[self.mp_pose.PoseLandmark.LEFT_HIP].y + landmarks[self.mp_pose.PoseLandmark.RIGHT_HIP].y) / 2 * height)
        ankle_y = int((landmarks[self.mp_pose.PoseLandmark.LEFT_ANKLE].y + landmarks[self.mp_pose.PoseLandmark.RIGHT_ANKLE].y) / 2 * height)
        margin = int(height * 0.05)

        if body_part == BodyPart.FULL:
            return max(0, nose_y - margin), min(height, ankle_y + margin)
        elif body_part == BodyPart.UPPER:
            return max(0, nose_y - margin), min(height, hip_y + margin)
        else:  # LOWER
            return max(0, hip_y - margin), min(height, ankle_y + margin)

    def release(self):
        self.pose.close()


def analyze_video(video_path, body_part, sample_frames=10):
    """영상에서 크롭 영역 분석"""
    cap = cv2.VideoCapture(video_path)
    detector = PoseDetector()

    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # 샘플 프레임에서 크롭 경계 계산
    sample_indices = np.linspace(0, frame_count - 1, min(sample_frames, frame_count), dtype=int)
    y_starts, y_ends = [], []

    for idx in sample_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if ret:
            y_start, y_end = detector.get_crop_boundaries(frame, body_part)
            y_starts.append(y_start)
            y_ends.append(y_end)

    cap.release()
    detector.release()

    return int(np.median(y_starts)), int(np.median(y_ends)), width, height


def merge_videos(video1_path, video2_path, output_path, body_part1, body_part2):
    """두 영상을 상하로 합치기"""
    print("영상 분석 중...")

    # 영상 분석
    y1_start, y1_end, w1, h1 = analyze_video(video1_path, body_part1)
    y2_start, y2_end, w2, h2 = analyze_video(video2_path, body_part2)

    crop1_height = y1_end - y1_start
    crop2_height = y2_end - y2_start
    half_height = max(crop1_height, crop2_height)

    # 출력 크기 계산
    ratio1 = half_height / crop1_height
    ratio2 = half_height / crop2_height
    output_width = max(int(w1 * ratio1), int(w2 * ratio2))
    output_height = half_height * 2

    print(f"출력 크기: {output_width}x{output_height}")

    # 영상 열기
    cap1 = cv2.VideoCapture(video1_path)
    cap2 = cv2.VideoCapture(video2_path)
    fps = cap1.get(cv2.CAP_PROP_FPS)
    total_frames = min(int(cap1.get(cv2.CAP_PROP_FRAME_COUNT)), int(cap2.get(cv2.CAP_PROP_FRAME_COUNT)))

    # 출력 설정
    fourcc = cv2.VideoWriter_fourcc(*'avc1')
    out = cv2.VideoWriter(output_path, fourcc, fps, (output_width, output_height))

    print(f"영상 처리 중... (총 {total_frames} 프레임)")

    for frame_num in range(total_frames):
        ret1, frame1 = cap1.read()
        ret2, frame2 = cap2.read()

        if not ret1 or not ret2:
            break

        # 크롭 및 리사이즈
        crop1 = frame1[y1_start:y1_end, :]
        crop2 = frame2[y2_start:y2_end, :]

        ratio1 = half_height / crop1.shape[0]
        ratio2 = half_height / crop2.shape[0]

        resize1 = cv2.resize(crop1, (int(crop1.shape[1] * ratio1), half_height))
        resize2 = cv2.resize(crop2, (int(crop2.shape[1] * ratio2), half_height))

        # 중앙 정렬
        result1 = np.zeros((half_height, output_width, 3), dtype=np.uint8)
        result2 = np.zeros((half_height, output_width, 3), dtype=np.uint8)

        x_offset1 = (output_width - resize1.shape[1]) // 2
        x_offset2 = (output_width - resize2.shape[1]) // 2

        result1[:, x_offset1:x_offset1 + resize1.shape[1]] = resize1
        result2[:, x_offset2:x_offset2 + resize2.shape[1]] = resize2

        # 합치기 및 저장
        merged = np.vstack([result1, result2])
        out.write(merged)

        if (frame_num + 1) % 30 == 0:
            print(f"  진행: {frame_num + 1}/{total_frames} ({(frame_num + 1) / total_frames * 100:.1f}%)")

    cap1.release()
    cap2.release()
    out.release()

    print(f"'{output_path}' 저장됨")


def get_body_part():
    """신체 부위 선택"""
    print("1. 전신 (FULL)")
    print("2. 상체 (UPPER)")
    print("3. 하체 (LOWER)")
    choice = input("선택 (1-3): ").strip()

    if choice == '1':
        return BodyPart.FULL
    elif choice == '2':
        return BodyPart.UPPER
    elif choice == '3':
        return BodyPart.LOWER
    else:
        print("잘못된 입력, 전신으로 설정")
        return BodyPart.FULL


if __name__ == "__main__":
    print("=" * 50)
    print("MediaPipe 영상 편집")
    print("=" * 50)

    video1 = input("첫 번째 영상 경로 (위): ").strip()
    video2 = input("두 번째 영상 경로 (아래): ").strip()
    output = input("편집한 영상 출력 경로: ").strip()

    print("\n영상 1 신체 부위:")
    part1 = get_body_part()

    print("\n영상 2 신체 부위:")
    part2 = get_body_part()

    merge_videos(video1, video2, output, part1, part2)