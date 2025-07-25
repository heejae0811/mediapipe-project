import os, cv2
import numpy as np
import pandas as pd
import mediapipe as mp

# 설정
LABEL = 1
VIDEO_PATH = './videos/홍규화_1_250710.mov'
FILE_ID = os.path.splitext(os.path.basename(VIDEO_PATH))[0]
OUTPUT_XLSX = f'./features_xlsx/{FILE_ID}.xlsx'
FRAME_INTERVAL = 1

# Mediapipe 초기화
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=False, min_detection_confidence=0.5)

cap = cv2.VideoCapture(VIDEO_PATH)
if not cap.isOpened():
    raise IOError(f"Cannot open video: {VIDEO_PATH}")

fps = cap.get(cv2.CAP_PROP_FPS)
trajectory = {i: {'x': [], 'y': [], 'z': [], 'visibility': []} for i in range(33)}
frame_idx = 0

# 프레임 반복
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
                trajectory[i]['z'].append(lm.z)
                trajectory[i]['visibility'].append(lm.visibility)
    frame_idx += 1

cap.release()
pose.close()

# 기준값 계산
total_time = frame_idx / fps
body_size = np.mean(np.sqrt(
    (np.array(trajectory[11]['x']) - np.array(trajectory[23]['x']))**2 + (np.array(trajectory[11]['y']) - np.array(trajectory[23]['y']))**2
))

# 결과 저장용 딕셔너리
velocity_data = {'id': FILE_ID, 'label': LABEL}
accel_data = {'id': FILE_ID, 'label': LABEL}
jerk_data = {'id': FILE_ID, 'label': LABEL}
vis_data = {'id': FILE_ID, 'label': LABEL}

data_map = {
    'velocity': velocity_data,
    'accel': accel_data,
    'jerk': jerk_data
}

# 관절별 분석
for i in range(33):
    x, y, z = np.array(trajectory[i]['x']), np.array(trajectory[i]['y']), np.array(trajectory[i]['z'])
    vis_array = np.array(trajectory[i]['visibility'])

    # 속도, 가속도, 저크 계산
    v = np.linalg.norm(np.diff(np.stack([x, y, z], axis=1), axis=0), axis=1) * fps
    a = np.diff(v) * fps
    j = np.diff(a) * fps

    def compute_stats(arr):
        return [np.min(arr), np.max(arr), np.mean(arr), np.std(arr)] if len(arr) > 3 else [np.nan]*4

    stats = {
        'velocity': compute_stats(v),
        'accel': compute_stats(a),
        'jerk': compute_stats(j)
    }

    for name in ['velocity', 'accel', 'jerk']:
        raw = stats[name]
        time_norm = [val / total_time for val in raw]
        dist_norm = [val / body_size for val in raw]

        for k, stat_name in zip(['min', 'max', 'mean', 'std'], range(4)):
            data_map[name][f'landmark{i}_{name}_{k}_raw'] = raw[stat_name]
            data_map[name][f'landmark{i}_{name}_{k}_timeNorm'] = time_norm[stat_name]
            data_map[name][f'landmark{i}_{name}_{k}_distNorm'] = dist_norm[stat_name]

    vis_data[f'landmark{i}_visibility_mean'] = np.mean(vis_array) if len(vis_array) > 0 else np.nan

# 엑셀 저장
os.makedirs(os.path.dirname(OUTPUT_XLSX), exist_ok=True)
with pd.ExcelWriter(OUTPUT_XLSX, engine='openpyxl') as writer:
    pd.DataFrame([velocity_data]).to_excel(writer, index=False, sheet_name='Velocity')
    pd.DataFrame([accel_data]).to_excel(writer, index=False, sheet_name='Acceleration')
    pd.DataFrame([jerk_data]).to_excel(writer, index=False, sheet_name='Jerk')
    pd.DataFrame([vis_data]).to_excel(writer, index=False, sheet_name='Visibility')

print(f"✅ 저장 완료: {OUTPUT_XLSX}")
