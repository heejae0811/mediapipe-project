import cv2
import mediapipe as mp
import numpy as np
import os

# ========= 기본 설정 =========
VIDEO_PATH = './test2.mov'
OUTPUT_MP4 = 'pelvis_tracking2.mp4'

mp_pose = mp.solutions.pose
cap = cv2.VideoCapture(VIDEO_PATH)

fps = cap.get(cv2.CAP_PROP_FPS)
if fps <= 1 or fps > 120 or np.isnan(fps):
    fps = 30.0

frame_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
if frame_w == 0 or frame_h == 0:
    frame_w, frame_h = 640, 480

# output 설정
fourcc = cv2.VideoWriter_fourcc(*'avc1')
output_path = os.path.abspath(OUTPUT_MP4)
out = cv2.VideoWriter(output_path, fourcc, fps, (frame_w, frame_h))

# ========= 트래킹 변수 =========
pelvis_path = []
SMOOTHING = 0.2  # 부드러움 설정

def smooth_point(prev, current, alpha=SMOOTHING):
    """Simple exponential smoothing"""
    if prev is None:
        return current
    return (1 - alpha) * np.array(prev) + alpha * np.array(current)


# ========= Mediapipe Pose =========
with mp_pose.Pose(
        static_image_mode=False,
        model_complexity=1,
        smooth_landmarks=True) as pose:

    prev_pelvis = None

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.resize(frame, (frame_w, frame_h))
        overlay = frame.copy()  # 선과 점을 그릴 오버레이

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(frame_rgb)

        if results.pose_landmarks:
            lm = results.pose_landmarks.landmark

            # 골반 좌표
            l_hip = (lm[23].x * frame_w, lm[23].y * frame_h)
            r_hip = (lm[24].x * frame_w, lm[24].y * frame_h)
            pelvis = np.array([(l_hip[0] + r_hip[0]) / 2,
                               (l_hip[1] + r_hip[1]) / 2])

            # 부드러운 smoothing 적용
            pelvis = smooth_point(prev_pelvis, pelvis)
            prev_pelvis = pelvis

            # 전체 경로 저장 (지우지 않음)
            pelvis_path.append(pelvis.copy())

            # ========= 밝은 녹색 선으로 전체 경로 그리기 =========
            for i in range(1, len(pelvis_path)):
                p1 = tuple(np.int32(pelvis_path[i - 1]))
                p2 = tuple(np.int32(pelvis_path[i]))
                cv2.line(overlay, p1, p2, (0, 255, 0), 8)  # 밝은 녹색, 두께 5

            # ========= 현재 골반 위치 빨간 점 =========
            center = tuple(np.int32(pelvis))
            cv2.circle(overlay, center, 11, (0, 0, 255), -1)  # 빨간색 점


        # overlay를 원본과 합성
        draw_frame = cv2.addWeighted(overlay, 0.9, frame, 0.1, 0)

        # 영상 저장
        out.write(draw_frame)

        # 확인용 화면 출력
        cv2.imshow("Pelvis Tracking (Full Path)", draw_frame)
        if cv2.waitKey(int(1000 / fps)) & 0xFF == ord('q'):
            break

cap.release()
out.release()
cv2.destroyAllWindows()

print(f"트래킹 영상 저장 완료: {OUTPUT_MP4}")
print(output_path)
