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

        valid = (v1 > threshold) & (v2 > threshold)
        if np.any(valid):
            return np.mean(np.sqrt((x1[valid] - x2[valid])**2 + (y1[valid] - y2[valid])**2))
        return None

    left = get_valid_distance(11, 23)   # 왼쪽 어깨-왼쪽 엉덩이
    right = get_valid_distance(12, 24)  # 오른쪽 어깨-오른쪽 엉덩이

    if left and right: return (left + right) / 2
    if left: return left
    if right: return right
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
def compute_stats(arr):
    if len(arr) > 3:
        return {
            'min': np.min(arr),
            'max': np.max(arr),
            'mean': np.mean(arr),
            'median': np.median(arr),
            'std': np.std(arr)
        }
    else:
        return {k: np.nan for k in ['min', 'max', 'mean', 'median', 'std']}

for i in range(33):
    x = np.array(trajectory[i]['x'])
    y = np.array(trajectory[i]['y'])
    vis = np.array(trajectory[i]['visibility'])

    # 속력, 가속도, 저크, 이동거리 계산
    xy = np.stack([x, y], axis=1)
    speed = np.linalg.norm(np.diff(xy, axis=0), axis=1) * fps # 이동 거리 × fps
    accel = np.diff(speed) * fps # speed 차이 × fps
    jerk = np.diff(accel) * fps # acceleration 차이 × fps
    total_dist = np.sum(np.linalg.norm(np.diff(xy, axis=0), axis=1)) # 총 이동 거리

    stats = {
        'speed': compute_stats(speed),
        'accel': compute_stats(accel),
        'jerk': compute_stats(jerk)
    }

    # Speed, Accel, Jerk 저장
    for feature_name in ['speed', 'accel', 'jerk']:
        for stat_name, val in stats[feature_name].items():
            data_map[feature_name][f'landmark{i}_{feature_name}_{stat_name}_raw'] = val
            data_map[feature_name][f'landmark{i}_{feature_name}_{stat_name}_bodyNorm'] = val / body_size

    # Distance 저장
    distance_data[f'landmark{i}_totalDistance_raw'] = total_dist
    distance_data[f'landmark{i}_totalDistance_timeBodyNorm'] = total_dist / (total_time * body_size)

    # Visibility 평균 저장
    visibility_data[f'landmark{i}_visibility_mean'] = np.mean(vis) if len(vis) > 0 else np.nan

# 엑셀 저장
os.makedirs(os.path.dirname(OUTPUT_XLSX), exist_ok=True)
with pd.ExcelWriter(OUTPUT_XLSX, engine='openpyxl') as writer:
    pd.DataFrame([speed_data]).to_excel(writer, index=False, sheet_name='Speed')
    pd.DataFrame([accel_data]).to_excel(writer, index=False, sheet_name='Acceleration')
    pd.DataFrame([jerk_data]).to_excel(writer, index=False, sheet_name='Jerk')
    pd.DataFrame([distance_data]).to_excel(writer, index=False, sheet_name='Distance')
    pd.DataFrame([visibility_data]).to_excel(writer, index=False, sheet_name='Visibility')

print(f"✅ 엑셀 저장 완료: {OUTPUT_XLSX}")
