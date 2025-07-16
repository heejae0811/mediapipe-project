import numpy as np
import pandas as pd
import mediapipe as mp
import cv2
import os

# 설정
LABEL = 1
VIDEO_PATH = './videos/홍규화_1_250710.mov'
FILE_ID = os.path.splitext(os.path.basename(VIDEO_PATH))[0]
OUTPUT_XLSX = f'./csv_feature/{FILE_ID}.xlsx'
FRAME_INTERVAL = 1

# Mediapipe Pose 초기화
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=False, min_detection_confidence=0.5)

cap = cv2.VideoCapture(VIDEO_PATH)
if not cap.isOpened():
    raise IOError(f'Cannot open video: {VIDEO_PATH}')

actual_fps = cap.get(cv2.CAP_PROP_FPS)
frame_idx = 0

# landmark 좌표 + visibility 저장
trajectory = {i: {'x': [], 'y': [], 'visibility': []} for i in range(33)}

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    if frame_idx % FRAME_INTERVAL == 0:
        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(image_rgb)

        if results.pose_landmarks:
            for i, lm in enumerate(results.pose_landmarks.landmark):
                trajectory[i]['x'].append(lm.x)
                trajectory[i]['y'].append(lm.y)
                trajectory[i]['visibility'].append(lm.visibility)

    frame_idx += 1

cap.release()
pose.close()

# 정규화 (어깨–엉덩이) 계산
shoulder_x = np.array(trajectory[11]['x'])  # left_shoulder
shoulder_y = np.array(trajectory[11]['y'])
hip_x = np.array(trajectory[23]['x'])       # left_hip
hip_y = np.array(trajectory[23]['y'])

body_sizes = np.sqrt((shoulder_x - hip_x)**2 + (shoulder_y - hip_y)**2)
mean_body_size = np.mean(body_sizes)

# 위치 변화량 + 정규화 계산 함수
def compute_displacement_metrics(x_list, y_list, body_size):
    x = np.array(x_list)
    y = np.array(y_list)

    if len(x) < 2 or body_size == 0:
        return [np.nan] * 10

    coords = np.stack([x, y], axis=1)
    deltas = np.diff(coords, axis=0)
    distances = np.linalg.norm(deltas, axis=1)
    normalized = distances / body_size

    return [
        np.min(distances), np.max(distances), np.mean(distances), np.median(distances), np.std(distances),
        np.min(normalized), np.max(normalized), np.mean(normalized), np.median(normalized), np.std(normalized)
    ]

# 데이터 정리
pos_data = {'id': FILE_ID, 'label': LABEL}
norm_data = {'id': FILE_ID, 'label': LABEL}
vis_data = {'id': FILE_ID, 'label': LABEL}

for idx in range(33):
    metrics = compute_displacement_metrics(
        trajectory[idx]['x'], trajectory[idx]['y'], mean_body_size
    )

    # 위치 변화량
    pos_data[f'landmark{idx}_min'] = metrics[0]
    pos_data[f'landmark{idx}_max'] = metrics[1]
    pos_data[f'landmark{idx}_mean'] = metrics[2]
    pos_data[f'landmark{idx}_median'] = metrics[3]
    pos_data[f'landmark{idx}_std'] = metrics[4]

    # 정규화된 위치 변화량
    norm_data[f'landmark{idx}_norm_min'] = metrics[5]
    norm_data[f'landmark{idx}_norm_max'] = metrics[6]
    norm_data[f'landmark{idx}_norm_mean'] = metrics[7]
    norm_data[f'landmark{idx}_norm_median'] = metrics[8]
    norm_data[f'landmark{idx}_norm_std'] = metrics[9]

    # 인식률
    vis_array = np.array(trajectory[idx]['visibility'])
    vis_mean = np.mean(vis_array) if len(vis_array) > 0 else np.nan
    vis_data[f'landmark{idx}_visibility_mean'] = vis_mean

# DataFrame 생성
pos_df = pd.DataFrame([pos_data])
norm_df = pd.DataFrame([norm_data])
vis_df = pd.DataFrame([vis_data])

# 엑셀 저장 (Sheet1: 위치 변화량, Sheet2: 정규화된 위치 변화량, Sheet3: 인식률)
os.makedirs(os.path.dirname(OUTPUT_XLSX), exist_ok=True)
with pd.ExcelWriter(OUTPUT_XLSX, engine='openpyxl') as writer:
    pos_df.to_excel(writer, index=False, sheet_name='Position Metrics')
    norm_df.to_excel(writer, index=False, sheet_name='Normalized Position Metrics')
    vis_df.to_excel(writer, index=False, sheet_name='Visibility Metrics')

print(f"✅ 위치 변화량 + 정규화 + 관절 인식률 저장 완료: '{OUTPUT_XLSX}'")
