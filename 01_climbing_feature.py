import os, cv2
import numpy as np
import pandas as pd
import mediapipe as mp

# 설정
LABEL = 0
VIDEO_PATH = './videos/71_오윤택_0_220328.mov'
FILE_ID = os.path.splitext(os.path.basename(VIDEO_PATH))[0]
OUTPUT_XLSX = f'./features_xlsx/{FILE_ID}.xlsx'
FRAME_INTERVAL = 1

# Mediapipe 초기화
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=False, model_complexity=1, min_detection_confidence=0.5)

cap = cv2.VideoCapture(VIDEO_PATH)
if not cap.isOpened():
    raise IOError(f"Cannot open video: {VIDEO_PATH}")

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

# 신체 크기 정규화 기준 계산
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

    left = get_valid_distance(11, 23)
    right = get_valid_distance(12, 24)

    if left and right: return (left + right) / 2
    if left: return left
    if right: return right
    return 1.0

# 통계 함수
def compute_stats(arr):
    if len(arr) > 3:
        return {
            'mean': np.mean(arr),
            'max': np.max(arr),
            'std': np.std(arr)
        }
    else:
        return {k: np.nan for k in ['mean', 'max', 'std']}

# 파생량 계산
def compute_derivatives(position_series):
    velocity = np.diff(position_series) * fps
    acceleration = np.diff(velocity) * fps
    jerk = np.diff(acceleration) * fps
    return velocity, acceleration, jerk

# 정규화 기준값
body_size = compute_body_size(trajectory)

# 결과 저장 리스트
distance_rows = []
speed_rows = []
acceleration_rows = []
jerk_rows = []

# 관절별 계산
for i in range(33):
    x = np.array(trajectory[i]['x'])
    y = np.array(trajectory[i]['y'])

    vx, ax, jx = compute_derivatives(x)
    vy, ay, jy = compute_derivatives(y)
    vxy, axy, jxy = compute_derivatives(x + y)

    # Distance
    dx = np.sum(np.abs(np.diff(x)))
    dy = np.sum(np.abs(np.diff(y)))
    dxy = np.sum(np.abs(np.diff(x + y)))

    distance_rows.append({
        'id': FILE_ID, 'label': LABEL, 'landmark': i,
        'distance_x_raw': dx,
        'distance_y_raw': dy,
        'distance_xy_raw': dxy,
        'distance_x_bodyNorm': dx / body_size,
        'distance_y_bodyNorm': dy / body_size,
        'distance_xy_bodyNorm': dxy / body_size
    })

    axis_data = {
        'x': (vx, ax, jx),
        'y': (vy, ay, jy),
        'xy': (vxy, axy, jxy)
    }

    for axis_name, (vel, acc, jrk) in axis_data.items():
        vel_stats = compute_stats(vel)
        acc_stats = compute_stats(acc)
        jrk_stats = compute_stats(jrk)

        speed_rows.append({
            'id': FILE_ID, 'label': LABEL, 'landmark': i,
            **{f'speed_{axis_name}_{k}_raw': v for k, v in vel_stats.items()},
            **{f'speed_{axis_name}_{k}_bodyNorm': v / body_size for k, v in vel_stats.items()}
        })

        acceleration_rows.append({
            'id': FILE_ID, 'label': LABEL, 'landmark': i,
            **{f'acceleration_{axis_name}_{k}_raw': v for k, v in acc_stats.items()},
            **{f'acceleration_{axis_name}_{k}_bodyNorm': v / body_size for k, v in acc_stats.items()}
        })

        jerk_rows.append({
            'id': FILE_ID, 'label': LABEL, 'landmark': i,
            **{f'jerk_{axis_name}_{k}_raw': v for k, v in jrk_stats.items()},
            **{f'jerk_{axis_name}_{k}_bodyNorm': v / body_size for k, v in jrk_stats.items()}
        })

# 행 → 열 변환
def flatten_rows(rows):
    flat = {}
    for row in rows:
        landmark = row['landmark']
        for k, v in row.items():
            if k not in ['id', 'label', 'landmark']:
                flat[f'landmark{landmark}_{k}'] = v
    return flat

flat_speed = flatten_rows(speed_rows)
flat_accel = flatten_rows(acceleration_rows)
flat_jerk = flatten_rows(jerk_rows)
flat_distance = flatten_rows(distance_rows)

# 전체 통합
merged_flat = {
    'id': FILE_ID,
    'label': LABEL,
    **flat_speed,
    **flat_accel,
    **flat_jerk,
    **flat_distance
}

# DataFrames (모두 id, label 포함)
df_speed = pd.DataFrame([{'id': FILE_ID, 'label': LABEL, **flat_speed}])
df_accel = pd.DataFrame([{'id': FILE_ID, 'label': LABEL, **flat_accel}])
df_jerk = pd.DataFrame([{'id': FILE_ID, 'label': LABEL, **flat_jerk}])
df_distance = pd.DataFrame([{'id': FILE_ID, 'label': LABEL, **flat_distance}])
df_flat = pd.DataFrame([merged_flat])

# 엑셀 저장
os.makedirs(os.path.dirname(OUTPUT_XLSX), exist_ok=True)
with pd.ExcelWriter(OUTPUT_XLSX, engine='openpyxl') as writer:
    df_speed.to_excel(writer, index=False, sheet_name='Speed')
    df_accel.to_excel(writer, index=False, sheet_name='Acceleration')
    df_jerk.to_excel(writer, index=False, sheet_name='Jerk')
    df_distance.to_excel(writer, index=False, sheet_name='Distance')
    df_flat.to_excel(writer, index=False, sheet_name='Flattened')

print(f"✅ 엑셀 저장 완료 (시트 5개 포함): {OUTPUT_XLSX}")
