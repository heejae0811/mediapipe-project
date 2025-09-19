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


# ====== 안전한 나누기 함수 ======
def safe_divide(numerator, denominator, default=np.nan):
    """0으로 나누기 방지"""
    if denominator == 0 or np.isnan(denominator) or denominator is None:
        return default
    return numerator / denominator


# ====== 개선된 통계 계산 함수 ======
def compute_stats(arr):
    """NaN 값을 제거하고 통계 계산"""
    arr = np.array(arr)
    valid_arr = arr[~np.isnan(arr)]  # NaN 제거

    if len(valid_arr) > 3:
        return {
            'mean': np.mean(valid_arr),
            'max': np.max(valid_arr),
            'std': np.std(valid_arr)
        }
    else:
        return {k: np.nan for k in ['mean', 'max', 'std']}


# ====== 안전한 파생 변수 계산 ======
def compute_derivatives(position_series, fps):
    """빈 배열과 짧은 배열 처리 개선"""
    position_series = np.array(position_series)

    if len(position_series) < 2:
        return np.array([]), np.array([]), np.array([])

    speed = np.diff(position_series) * fps

    if len(speed) < 2:
        return speed, np.array([]), np.array([])

    acceleration = np.diff(speed) * fps

    if len(acceleration) < 2:
        return speed, acceleration, np.array([])

    jerk = np.diff(acceleration) * fps
    return speed, acceleration, jerk


# ====== 안전한 XY 거리 계산 ======
def compute_xy_distance_safe(x, y):
    """배열 길이 불일치 방지"""
    min_len = min(len(x), len(y))
    if min_len < 2:
        return 0

    x_trimmed = np.array(x[:min_len])
    y_trimmed = np.array(y[:min_len])

    try:
        return np.sum(np.linalg.norm(
            np.diff(np.stack([x_trimmed, y_trimmed], axis=1), axis=0), axis=1
        ))
    except Exception:
        return 0


# ====== 신체 크기 계산 ======
def compute_body_size(trajectory, threshold=0.5):
    def get_valid_distance(idx1, idx2):
        x1, y1, v1 = np.array(trajectory[idx1]['x']), np.array(trajectory[idx1]['y']), np.array(trajectory[idx1]['visibility'])
        x2, y2, v2 = np.array(trajectory[idx2]['x']), np.array(trajectory[idx2]['y']), np.array(trajectory[idx2]['visibility'])
        valid = (v1 > threshold) & (v2 > threshold)
        if np.any(valid):
            return np.mean(np.sqrt((x1[valid] - x2[valid]) ** 2 + (y1[valid] - y2[valid]) ** 2))
        return None

    left = get_valid_distance(11, 23)
    right = get_valid_distance(12, 24)

    if left and right:
        return (left + right) / 2
    if left:
        return left
    if right:
        return right
    return 1.0  # 기본값


# ====== 변수 합치기 ======
def flatten_rows(rows):
    flat = {}
    for row in rows:
        landmark = row['landmark']
        for k, v in row.items():
            if k not in ['id', 'label', 'landmark']:
                flat[f'landmark{landmark}_{k}'] = v
    return flat


# ====== 개선된 비디오 처리 함수 ======
def process_video(VIDEO_PATH, LABEL=0):
    FILE_ID = os.path.splitext(os.path.basename(VIDEO_PATH))[0]
    OUTPUT_XLSX = os.path.join(OUTPUT_DIR, f'{FILE_ID}.xlsx')

    cap = None
    pose = None

    try:
        cap = cv2.VideoCapture(VIDEO_PATH)
        if not cap.isOpened():
            raise IOError(f"Cannot open video: {VIDEO_PATH}")

        fps = cap.get(cv2.CAP_PROP_FPS)
        if fps <= 0:  # fps가 0이거나 음수인 경우
            fps = 30  # 기본값 설정

        mp_pose = mp.solutions.pose
        pose = mp_pose.Pose(
            static_image_mode=False,
            model_complexity=1,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )

        trajectory = {i: {'x': [], 'y': [], 'visibility': []} for i in range(33)}
        frame_idx = 0

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            if frame_idx % FRAME_INTERVAL == 0:
                try:
                    image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    results = pose.process(image_rgb)

                    if results.pose_landmarks:
                        for i, lm in enumerate(results.pose_landmarks.landmark):
                            trajectory[i]['x'].append(lm.x)
                            trajectory[i]['y'].append(lm.y)
                            trajectory[i]['visibility'].append(lm.visibility)
                    else:
                        # 포즈가 감지되지 않은 경우 NaN으로 채우기
                        for i in range(33):
                            trajectory[i]['x'].append(np.nan)
                            trajectory[i]['y'].append(np.nan)
                            trajectory[i]['visibility'].append(0.0)

                except Exception as e:
                    print(f"프레임 {frame_idx} 처리 중 오류: {e}")
                    # 오류 발생시 NaN으로 채우기
                    for i in range(33):
                        trajectory[i]['x'].append(np.nan)
                        trajectory[i]['y'].append(np.nan)
                        trajectory[i]['visibility'].append(0.0)

            frame_idx += 1

        body_size = compute_body_size(trajectory)
        total_time = frame_idx / fps

        distance_rows, speed_rows, acceleration_rows, jerk_rows = [], [], [], []

        for i in range(33):
            x = np.array(trajectory[i]['x'])
            y = np.array(trajectory[i]['y'])

            # 거리 계산 (NaN 제거)
            valid_x = x[~np.isnan(x)]
            valid_y = y[~np.isnan(y)]

            dx = np.sum(np.abs(np.diff(valid_x))) if len(valid_x) > 1 else 0
            dy = np.sum(np.abs(np.diff(valid_y))) if len(valid_y) > 1 else 0
            dxy = compute_xy_distance_safe(trajectory[i]['x'], trajectory[i]['y'])

            # 파생 변수 계산
            sx, ax, jx = compute_derivatives(valid_x, fps)
            sy, ay, jy = compute_derivatives(valid_y, fps)

            # xy 합성 좌표의 파생 변수
            min_len = min(len(valid_x), len(valid_y))
            if min_len > 1:
                xy_positions = np.sqrt(valid_x[:min_len] ** 2 + valid_y[:min_len] ** 2)
                sxy, axy, jxy = compute_derivatives(xy_positions, fps)
            else:
                sxy, axy, jxy = np.array([]), np.array([]), np.array([])

            # 거리 데이터 저장
            distance_rows.append({
                'id': FILE_ID,
                'label': LABEL,
                'landmark': i,
                'distance_x_raw': dx,
                'distance_y_raw': dy,
                'distance_xy_raw': dxy,
                'distance_x_timeBodyNorm': safe_divide(dx, total_time * body_size),
                'distance_y_timeBodyNorm': safe_divide(dy, total_time * body_size),
                'distance_xy_timeBodyNorm': safe_divide(dxy, total_time * body_size)
            })

            # 축별 데이터 처리
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
                    'id': FILE_ID,
                    'label': LABEL,
                    'landmark': i,
                    **{f'speed_{axis_name}_{k}_raw': v for k, v in spd_stats.items()},
                    **{f'speed_{axis_name}_{k}_bodyNorm': safe_divide(v, body_size) for k, v in spd_stats.items()}
                })

                acceleration_rows.append({
                    'id': FILE_ID,
                    'label': LABEL,
                    'landmark': i,
                    **{f'acceleration_{axis_name}_{k}_raw': v for k, v in acc_stats.items()},
                    **{f'acceleration_{axis_name}_{k}_bodyNorm': safe_divide(v, body_size) for k, v in
                       acc_stats.items()}
                })

                jerk_rows.append({
                    'id': FILE_ID,
                    'label': LABEL,
                    'landmark': i,
                    **{f'jerk_{axis_name}_{k}_raw': v for k, v in jrk_stats.items()},
                    **{f'jerk_{axis_name}_{k}_bodyNorm': safe_divide(v, body_size) for k, v in jrk_stats.items()}
                })

        # ====== Flattened 데이터 생성 ======
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

        # ====== 엑셀 파일 저장 ======
        try:
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

        except Exception as e:
            print(f"❌ 파일 저장 오류: {OUTPUT_XLSX}")
            print(f"오류 내용: {e}")

    except Exception as e:
        print(f"❌ 비디오 처리 오류: {VIDEO_PATH}")
        print(f"오류 내용: {e}")

    finally:
        # 리소스 정리
        if cap is not None:
            cap.release()
        if pose is not None:
            pose.close()


# ====== 전체 처리 루프 ======
def main():
    video_files = glob.glob(os.path.join(VIDEO_DIR, '*.mov')) + glob.glob(os.path.join(VIDEO_DIR, '*.mp4'))

    if not video_files:
        print("❌ 비디오 파일을 찾을 수 없습니다.")
        print(f"경로 확인: {VIDEO_DIR}")
        return

    print(f"📁 {len(video_files)}개의 비디오 파일을 발견했습니다.")

    success_count = 0
    for video_path in video_files:
        try:
            label = extract_label(os.path.basename(video_path))
            print(f"\n🔄 처리 중: {os.path.basename(video_path)} (Label: {label})")
            process_video(video_path, LABEL=label)
            success_count += 1

        except Exception as e:
            print(f"❌ 오류 발생: {video_path}")
            print(f"오류 내용: {e}")

    print(f"\n🎉 처리 완료: {success_count}/{len(video_files)}개 파일 성공")


if __name__ == "__main__":
    main()