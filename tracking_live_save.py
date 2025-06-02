import cv2
import mediapipe as mp
import numpy as np
import csv
import os

def euclidean_distance(p1, p2):
    return np.linalg.norm(np.array(p1) - np.array(p2))

def get_point(lm, idx, shape):
    h, w = shape[:2]
    return lm[idx].x * w, lm[idx].y * h

# --- 파일 및 파라미터 설정 ---
VIDEO_PATH = 'videos/joint_live01.mov'
OUTPUT_CSV = 'balance_tracking_result.csv'
OUTPUT_MP4 = 'balance_tracking_result.mp4'

mp_pose = mp.solutions.pose
cap = cv2.VideoCapture(VIDEO_PATH)

# FPS, 해상도 예외처리
fps = cap.get(cv2.CAP_PROP_FPS)
if fps <= 1 or fps > 120 or np.isnan(fps):
    fps = 30.0

frame_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
if frame_w == 0 or frame_h == 0:
    frame_w, frame_h = 640, 480

dt = 1 / fps

# mp4(H.264)로 저장. Mac/Win 모두 잘 되는 'avc1' 코덱, 안되면 mp4v로 시도
fourcc = cv2.VideoWriter_fourcc(*'avc1')
output_path = os.path.abspath(OUTPUT_MP4)
out = cv2.VideoWriter(output_path, fourcc, fps, (frame_w, frame_h))

prev_head = None
prev_pelvis = None
prev_v_head = 0
prev_v_pelvis = 0

head_path = []
pelvis_path = []

head_dists, head_vels, head_accs = [], [], []
pelvis_dists, pelvis_vels, pelvis_accs = [], [], []

# 고정 배경 박스 크기 (320x180 추천)
BG_BOX_W, BG_BOX_H = 320, 180

with mp_pose.Pose() as pose:
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # 프레임 크기 보정 (저장 오류 방지)
        frame = cv2.resize(frame, (frame_w, frame_h))
        shape = frame.shape

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(frame_rgb)

        if results.pose_landmarks:
            lm = results.pose_landmarks.landmark

            head = get_point(lm, 0, shape)
            l_hip = get_point(lm, 23, shape)
            r_hip = get_point(lm, 24, shape)
            pelvis = ((l_hip[0] + r_hip[0]) / 2, (l_hip[1] + r_hip[1]) / 2)

            if prev_head and prev_pelvis:
                d_head = euclidean_distance(head, prev_head)
                d_pelvis = euclidean_distance(pelvis, prev_pelvis)
                v_head = d_head / dt
                v_pelvis = d_pelvis / dt
                a_head = (v_head - prev_v_head) / dt
                a_pelvis = (v_pelvis - prev_v_pelvis) / dt
            else:
                d_head = d_pelvis = v_head = v_pelvis = a_head = a_pelvis = 0

            prev_head = head
            prev_pelvis = pelvis
            prev_v_head = v_head
            prev_v_pelvis = v_pelvis

            head_dists.append(d_head)
            head_vels.append(v_head)
            head_accs.append(a_head)

            pelvis_dists.append(d_pelvis)
            pelvis_vels.append(v_pelvis)
            pelvis_accs.append(a_pelvis)

            head_path.append(head)
            pelvis_path.append(pelvis)

            # 머리, 골반 위치 표시
            cv2.circle(frame, tuple(np.int32(head)), 6, (0, 0, 255), -1)
            cv2.circle(frame, tuple(np.int32(pelvis)), 6, (0, 0, 255), -1)

            # 경로(흰색)
            for path in [head_path, pelvis_path]:
                for i in range(1, len(path)):
                    cv2.line(frame, tuple(np.int32(path[i - 1])), tuple(np.int32(path[i])), (255, 255, 255), 2)

            # --- 텍스트 정보 준비 ---
            display_lines = [
                f'Head Dist: {d_head:.2f}',
                f'Head Vel: {v_head:.2f}',
                f'Head Acc: {a_head:.2f}',
                f'Pelvis Dist: {d_pelvis:.2f}',
                f'Pelvis Vel: {v_pelvis:.2f}',
                f'Pelvis Acc: {a_pelvis:.2f}'
            ]

            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.6
            font_thickness = 2
            text_color = (0, 0, 0)
            bg_color = (255, 255, 255)
            alpha = 0.5

            # 고정 배경 박스 (왼쪽 상단)
            x, y = 10, 25
            overlay = frame.copy()
            cv2.rectangle(
                overlay,
                (x - 10, y - 20),
                (x - 10 + BG_BOX_W, y - 20 + BG_BOX_H),
                bg_color, -1
            )
            cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)

            # 텍스트 쓰기
            y_offset = y
            for line in display_lines:
                cv2.putText(frame, line, (x, y_offset), font, font_scale, text_color, font_thickness, cv2.LINE_AA)
                y_offset += 30

        # --- 영상 저장 ---
        out.write(frame)

        # 화면 출력
        cv2.imshow('Head & Pelvis Tracking', frame)
        if cv2.waitKey(int(1000 / fps)) & 0xFF == ord('q'):
            break

cap.release()
out.release()
cv2.destroyAllWindows()

# --- CSV 저장 ---
head_total_dist = sum(head_dists)
head_avg_vel = np.mean(head_vels)
head_avg_acc = np.mean(head_accs)
pelvis_total_dist = sum(pelvis_dists)
pelvis_avg_vel = np.mean(pelvis_vels)
pelvis_avg_acc = np.mean(pelvis_accs)

with open(OUTPUT_CSV, 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(['head_total_dist', 'head_avg_vel', 'head_avg_acc',
                     'pelvis_total_dist', 'pelvis_avg_vel', 'pelvis_avg_acc'])
    writer.writerow([f'{head_total_dist:.2f}', f'{head_avg_vel:.2f}', f'{head_avg_acc:.2f}',
                     f'{pelvis_total_dist:.2f}', f'{pelvis_avg_vel:.2f}', f'{pelvis_avg_acc:.2f}'])

print(f"요약 CSV 저장 완료: {OUTPUT_CSV}")
print(f"영상 저장 완료: {OUTPUT_MP4} ({output_path})")
