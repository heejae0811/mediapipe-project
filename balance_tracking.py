import cv2
import mediapipe as mp
import numpy as np
import csv

# 거리 계산 함수
def euclidean_distance(p1, p2):
    return np.linalg.norm(np.array(p1) - np.array(p2))

# 랜드마크 좌표를 픽셀 기준으로 반환
def get_point(lm, idx, shape):
    h, w = shape[:2]
    return lm[idx].x * w, lm[idx].y * h

# 초기 설정
mp_pose = mp.solutions.pose
cap = cv2.VideoCapture('balance_video.mov')

fps = cap.get(cv2.CAP_PROP_FPS)
dt = 1 / fps if fps > 0 else 0.033

# 초기값
prev_head = None
prev_pelvis = None
prev_v_head = 0
prev_v_pelvis = 0

head_path = []
pelvis_path = []

# 누적용 값 저장 리스트
head_dists, head_vels, head_accs = [], [], []
pelvis_dists, pelvis_vels, pelvis_accs = [], [], []

with mp_pose.Pose() as pose:
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        shape = frame.shape
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(frame_rgb)

        if results.pose_landmarks:
            lm = results.pose_landmarks.landmark

            # 머리와 골반 중심 위치 계산
            head = get_point(lm, 0, shape)
            l_hip = get_point(lm, 23, shape)
            r_hip = get_point(lm, 24, shape)
            pelvis = ((l_hip[0] + r_hip[0]) / 2, (l_hip[1] + r_hip[1]) / 2)

            # 거리, 속도, 가속도 계산
            if prev_head and prev_pelvis:
                d_head = euclidean_distance(head, prev_head)
                d_pelvis = euclidean_distance(pelvis, prev_pelvis)

                v_head = d_head / dt
                v_pelvis = d_pelvis / dt

                a_head = (v_head - prev_v_head) / dt
                a_pelvis = (v_pelvis - prev_v_pelvis) / dt
            else:
                d_head = d_pelvis = v_head = v_pelvis = a_head = a_pelvis = 0

            # 이전 값 업데이트
            prev_head = head
            prev_pelvis = pelvis
            prev_v_head = v_head
            prev_v_pelvis = v_pelvis

            # 누적 저장
            head_dists.append(d_head)
            head_vels.append(v_head)
            head_accs.append(a_head)

            pelvis_dists.append(d_pelvis)
            pelvis_vels.append(v_pelvis)
            pelvis_accs.append(a_pelvis)

            # 경로 저장
            head_path.append(head)
            pelvis_path.append(pelvis)

            # 빨간 점 표시
            cv2.circle(frame, tuple(np.int32(head)), 6, (0, 0, 255), -1)
            cv2.circle(frame, tuple(np.int32(pelvis)), 6, (0, 0, 255), -1)

            # 이동 경로 선 그리기 (하얀색)
            for path in [head_path, pelvis_path]:
                for i in range(1, len(path)):
                    cv2.line(frame, tuple(np.int32(path[i - 1])), tuple(np.int32(path[i])), (255, 255, 255), 2)

            # 실시간 거리/속도/가속도 텍스트 표시
            y0 = 30
            for label, val in zip(
                ['Head Dist', 'Head Vel', 'Head Acc', 'Pelvis Dist', 'Pelvis Vel', 'Pelvis Acc'],
                [d_head, v_head, a_head, d_pelvis, v_pelvis, a_pelvis]
            ):
                cv2.putText(frame, f'{label}: {val:.2f}', (10, y0),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2, cv2.LINE_AA)
                y0 += 25

        # 화면 출력
        cv2.imshow('Head & Pelvis Tracking', frame)
        if cv2.waitKey(int(1000 / fps)) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()

# ✅ 최종 요약 계산
head_total_dist = sum(head_dists)
head_avg_vel = np.mean(head_vels)
head_avg_acc = np.mean(head_accs)

pelvis_total_dist = sum(pelvis_dists)
pelvis_avg_vel = np.mean(pelvis_vels)
pelvis_avg_acc = np.mean(pelvis_accs)

# ✅ CSV 요약 저장
with open('balance_tracking_result.csv', 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(['head_total_dist', 'head_avg_vel', 'head_avg_acc',
                     'pelvis_total_dist', 'pelvis_avg_vel', 'pelvis_avg_acc'])
    writer.writerow([f'{head_total_dist:.2f}', f'{head_avg_vel:.2f}', f'{head_avg_acc:.2f}',
                     f'{pelvis_total_dist:.2f}', f'{pelvis_avg_vel:.2f}', f'{pelvis_avg_acc:.2f}'])

print("✅ 요약 CSV 저장 완료: balance_tracking_result.csv")
