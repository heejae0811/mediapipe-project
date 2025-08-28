import os, cv2
import numpy as np
import pandas as pd
import mediapipe as mp

# 설정
LABEL = 0
VIDEO_PATH = './videos_all/02__0_230510.mov'
FILE_ID = os.path.splitext(os.path.basename(VIDEO_PATH))[0]
OUTPUT_XLSX = f'./features_xlsx/{FILE_ID}.xlsx'
FRAME_INTERVAL = 1

# Mediapipe 초기화
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=False, model_complexity=1, min_detection_confidence=0.5)

cap = cv2.VideoCapture(VIDEO_PATH)
if not cap.isOpened():
    raise IOError(f"Cannot open video: {VIDEO_PATH}")

# 관절 인식률
fps = cap.get(cv2.CAP_PROP_FPS)
trajectory = {i: {'x': [], 'y': [], 'visibility': []} for i in range(33)}
frame_idx = 0

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

# 신체 크기 정규화
def compute_body_size(trajectory, threshold=0.5):
    def get_valid_distance(idx1, idx2):
        x1 = np.array(trajectory[idx1]['x'])
        y1 = np.array(trajectory[idx1]['y'])
        v1 = np.array(trajectory[idx1]['visibility'])

        x2 = np.array(trajectory[idx2]['x'])
        y2 = np.array(trajectory[idx2]['y'])
        v2 = np.array(trajectory[idx2]['visibility'])

        valid_mask = (v1 > threshold) & (v2 > threshold)
        if np.sum(valid_mask) > 0:
            dist = np.sqrt((x1[valid_mask] - x2[valid_mask])**2 + (y1[valid_mask] - y2[valid_mask])**2)
            return np.mean(dist)
        else:
            return None

    left = get_valid_distance(11, 23)   # 왼쪽 어깨-왼쪽 엉덩이
    right = get_valid_distance(12, 24)  # 오른쪽 어깨-오른쪽 엉덩이

    if left is not None and right is not None:
        return (left + right) / 2
    elif left is not None:
        return left
    elif right is not None:
        return right
    else:
        return 1.0

# 결과 저장용 딕셔너리
speed_data = {'id': FILE_ID, 'label': LABEL}
accel_data = {'id': FILE_ID, 'label': LABEL}
jerk_data = {'id': FILE_ID, 'label': LABEL}
distance_data = {'id': FILE_ID, 'label': LABEL}
visibility_data = {'id': FILE_ID, 'label': LABEL}
data_map = {'speed': speed_data, 'accel': accel_data, 'jerk': jerk_data}

# 정규화 기준값
total_time = frame_idx / fps
body_size = compute_body_size(trajectory)

# 관절별 분석
for i in range(33):
    x = np.array(trajectory[i]['x'])
    y = np.array(trajectory[i]['y'])
    vis_array = np.array(trajectory[i]['visibility'])

    # 속력, 가속도, 저크, 이동거리 계산
    xy = np.stack([x, y], axis=1)
    s = np.linalg.norm(np.diff(xy, axis=0), axis=1) * fps # 이동 거리 × fps
    a = np.diff(s) * fps # speed 차이 × fps
    j = np.diff(a) * fps # acceleration 차이 × fps
    d = np.sum(np.linalg.norm(np.diff(xy, axis=0), axis=1))  # 총 이동 거리

    def compute_stats(arr):
        if len(arr) > 3:
            return [np.min(arr), np.max(arr), np.mean(arr), np.median(arr), np.std(arr)]
        else:
            return [np.nan]*5

    stats = {
        'speed': compute_stats(s),
        'accel': compute_stats(a),
        'jerk': compute_stats(j)
    }

    # Speed / Accel / Jerk 저장
    for name in ['speed', 'accel', 'jerk']:
        raw = stats[name]
        body_norm = [val / body_size for val in raw]

        for k, stat_name in zip(['min', 'max', 'mean', 'median', 'std'], range(5)):
            data_map[name][f'landmark{i}_{name}_{k}_raw'] = raw[stat_name]
            data_map[name][f'landmark{i}_{name}_{k}_bodyNorm'] = body_norm[stat_name]

    # Distance 저장
    distance_data[f'landmark{i}_totalDistance_raw'] = d
    distance_data[f'landmark{i}_totalDistance_timeBodyNorm'] = d / (total_time * body_size)

    # Visibility 평균 저장
    visibility_data[f'landmark{i}_visibility_mean'] = np.mean(vis_array) if len(vis_array) > 0 else np.nan

# 엑셀 저장
os.makedirs(os.path.dirname(OUTPUT_XLSX), exist_ok=True)
with pd.ExcelWriter(OUTPUT_XLSX, engine='openpyxl') as writer:
    pd.DataFrame([speed_data]).to_excel(writer, index=False, sheet_name='Speed')
    pd.DataFrame([accel_data]).to_excel(writer, index=False, sheet_name='Acceleration')
    pd.DataFrame([jerk_data]).to_excel(writer, index=False, sheet_name='Jerk')
    pd.DataFrame([distance_data]).to_excel(writer, index=False, sheet_name='Distance')
    pd.DataFrame([visibility_data]).to_excel(writer, index=False, sheet_name='Visibility')

print(f"✅ 엑셀 저장 완료: {OUTPUT_XLSX}")
