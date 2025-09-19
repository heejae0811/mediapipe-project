import os, cv2, glob, re
import numpy as np
import pandas as pd
import mediapipe as mp

# ====== 설정 ======
VIDEO_DIR = './videos/'
OUTPUT_DIR = './features_xlsx/'
os.makedirs(OUTPUT_DIR, exist_ok=True)
FRAME_INTERVAL = 1

# ====== 라벨 추출 ======
def extract_label(filename):
    match = re.search(r'_([01])_', filename)
    return int(match.group(1)) if match else 0

# ====== 통계 계산 함수 ======
def compute_stats(arr):
    if len(arr) > 3:
        return {
            'mean': np.mean(arr),
            'max': np.max(arr),
            'std': np.std(arr)
        }
    else:
        return {k: np.nan for k in ['mean', 'max', 'std']}

# ====== 파생 변수 계산 ======
def compute_derivatives(position_series, fps):
    speed = np.diff(position_series) * fps
    acceleration = np.diff(speed) * fps
    jerk = np.diff(acceleration) * fps
    return speed, acceleration, jerk

# ====== 신체 크기 계산 ======
def compute_body_size(trajectory, threshold=0.5):
    def get_valid_distance(idx1, idx2):
        x1, y1, v1 = np.array(trajectory[idx1]['x']), np.array(trajectory[idx1]['y']), np.array(trajectory[idx1]['visibility'])
        x2, y2, v2 = np.array(trajectory[idx2]['x']), np.array(trajectory[idx2]['y']), np.array(trajectory[idx2]['visibility'])
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

# ====== 변수 합치기 ======
def flatten_rows(rows):
    flat = {}
    for row in rows:
        landmark = row['landmark']
        for k, v in row.items():
            if k not in ['id', 'label', 'landmark']:
                flat[f'landmark{landmark}_{k}'] = v
    return flat

# ====== 비디오 처리 함수 ======
def process_video(VIDEO_PATH, LABEL=0):
    FILE_ID = os.path.splitext(os.path.basename(VIDEO_PATH))[0]
    OUTPUT_XLSX = os.path.join(OUTPUT_DIR, f'{FILE_ID}.xlsx')

    cap = cv2.VideoCapture(VIDEO_PATH)
    if not cap.isOpened():
        raise IOError(f"Cannot open video: {VIDEO_PATH}")
    fps = cap.get(cv2.CAP_PROP_FPS)

    mp_pose = mp.solutions.pose
    pose = mp_pose.Pose(static_image_mode=False, model_complexity=1, min_detection_confidence=0.5)

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

    body_size = compute_body_size(trajectory)
    total_time = frame_idx / fps

    distance_rows, speed_rows, acceleration_rows, jerk_rows = [], [], [], []

    for i in range(33):
        x = np.array(trajectory[i]['x'])
        y = np.array(trajectory[i]['y'])

        dx = np.sum(np.abs(np.diff(x)))
        dy = np.sum(np.abs(np.diff(y)))
        dxy = np.sum(np.linalg.norm(np.diff(np.stack([x, y], axis=1), axis=0), axis=1))

        sx, ax, jx = compute_derivatives(x, fps)
        sy, ay, jy = compute_derivatives(y, fps)
        sxy, axy, jxy = compute_derivatives(np.linalg.norm(np.diff(np.stack([x, y], axis=1), axis=0), axis=1), fps)

        distance_rows.append({
            'id': FILE_ID, 'label': LABEL, 'landmark': i,
            'distance_x_raw': dx,
            'distance_x_timeBodyNorm': dx / (total_time * body_size),
            'distance_y_raw': dy,
            'distance_y_timeBodyNorm': dy / (total_time * body_size),
            'distance_xy_raw': dxy,
            'distance_xy_timeBodyNorm': dxy / (total_time * body_size)
        })

        axis_data = {
            'x': (sx, ax, jx),
            'y': (sy, ay, jy),
            'xy': (sxy, axy, jxy)
        }

        for axis_name, (spd, acc, jrk) in axis_data.items():
            spd_stats = compute_stats(spd)
            acc_stats = compute_stats(acc)
            jrk_stats = compute_stats(jrk)

            speed_rows.append({
                'id': FILE_ID, 'label': LABEL, 'landmark': i,
                **{f'speed_{axis_name}_{k}_raw': v for k, v in spd_stats.items()},
                **{f'speed_{axis_name}_{k}_bodyNorm': v / body_size for k, v in spd_stats.items()}
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

    # ====== Flattened ======
    flat_distance = flatten_rows(distance_rows)
    flat_speed = flatten_rows(speed_rows)
    flat_accel = flatten_rows(acceleration_rows)
    flat_jerk = flatten_rows(jerk_rows)

    merged_flat = {
        'id': FILE_ID,
        'label': LABEL,
        **flat_distance,
        **flat_speed,
        **flat_accel,
        **flat_jerk
    }

    # ====== 저장 ======
    df_distance = pd.DataFrame([{'id': FILE_ID, 'label': LABEL, **flat_distance}])
    df_speed = pd.DataFrame([{'id': FILE_ID, 'label': LABEL, **flat_speed}])
    df_accel = pd.DataFrame([{'id': FILE_ID, 'label': LABEL, **flat_accel}])
    df_jerk = pd.DataFrame([{'id': FILE_ID, 'label': LABEL, **flat_jerk}])
    df_flat = pd.DataFrame([merged_flat])

    with pd.ExcelWriter(OUTPUT_XLSX, engine='openpyxl') as writer:
        df_distance.to_excel(writer, index=False, sheet_name='Distance')
        df_speed.to_excel(writer, index=False, sheet_name='Speed')
        df_accel.to_excel(writer, index=False, sheet_name='Acceleration')
        df_jerk.to_excel(writer, index=False, sheet_name='Jerk')
        df_flat.to_excel(writer, index=False, sheet_name='Flattened')

    print(f"✅ 저장 완료: {OUTPUT_XLSX}")

# ====== 전체 처리 루프 ======
video_files = glob.glob(os.path.join(VIDEO_DIR, '*.mov')) + glob.glob(os.path.join(VIDEO_DIR, '*.mp4'))

for video_path in video_files:
    try:
        label = extract_label(os.path.basename(video_path))
        print(f"\n🔄 처리 중: {os.path.basename(video_path)} (Label: {label})")
        process_video(video_path, LABEL=label)
    except Exception as e:
        print(f"❌ 오류 발생: {video_path}")
        print(e)
