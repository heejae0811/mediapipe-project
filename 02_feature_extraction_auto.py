import os, glob, cv2, re
import numpy as np
import pandas as pd
import mediapipe as mp


# ====== 설정 ======
VIDEO_DIR = './videos/'
OUTPUT_DIR = './features_xlsx/'
os.makedirs(OUTPUT_DIR, exist_ok=True)
FRAME_INTERVAL = 1


# ------ Threshold ------
IMMOBILITY_SPEED_THRESH = 0.01
PLATEAU_DY_THRESH = 0.002
PLATEAU_MIN_DURATION_SEC = 0.5
STABILITY_SPEED_THRESH = 0.02
STABILITY_MIN_FRAMES = 5


# ====== 라벨 추출 ======
def extract_label(filename):
    match = re.search(r'_([01])_', filename)
    return int(match.group(1)) if match else 0


# ====== 기본 유틸 ======
def safe_divide(num, den, default=np.nan):
    if den is None or den == 0 or np.isnan(den):
        return default
    return num / den


def rms(arr):
    arr = np.asarray(arr, dtype=float)
    arr = arr[~np.isnan(arr)]
    if arr.size == 0:
        return np.nan
    return np.sqrt(np.mean(arr**2))


# ====== body-size 계산 ======
def compute_body_size(lm):
    """
    shoulder(11-12), hip(23-24), shoulder-hip 거리 중
    유효한 값들의 평균을 body_size로 사용.
    """
    def dist(a, b):
        if any(np.isnan([a.x, a.y, b.x, b.y])):
            return np.nan
        return np.sqrt((a.x - b.x)**2 + (a.y - b.y)**2)

    d1 = dist(lm[11], lm[23])
    d2 = dist(lm[12], lm[24])
    d3 = dist(lm[11], lm[12])
    d4 = dist(lm[23], lm[24])

    vals = [d for d in [d1, d2, d3, d4] if not np.isnan(d)]
    if len(vals) == 0:
        return 1.0
    return np.mean(vals)


# ====== kinematics (raw & normalized 모두 계산) ======
def compute_kinematics_raw_norm(x, y, fps, body_size):
    """
    x, y: landmark의 좌표 배열
    fps: video framerate
    body_size: 정규화 기준 값

    return:
        speed_raw, acc_raw, jerk_raw,
        speed_norm, acc_norm, jerk_norm
    """

    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)

    valid = (~np.isnan(x)) & (~np.isnan(y))
    x_v = x[valid]
    y_v = y[valid]

    if len(x_v) < 2:
        return (np.array([]), np.array([]), np.array([]), np.array([]), np.array([]), np.array([]))

    # ----- speed -----
    vx = np.diff(x_v) * fps
    vy = np.diff(y_v) * fps
    speed_raw = np.sqrt(vx**2 + vy**2)
    speed_norm = safe_divide(speed_raw, body_size)

    if len(speed_raw) < 2:
        return speed_raw, np.array([]), np.array([]), speed_norm, np.array([]), np.array([])

    # ----- acceleration -----
    ax = np.diff(vx) * fps
    ay = np.diff(vy) * fps
    acc_raw = np.sqrt(ax**2 + ay**2)
    acc_norm = safe_divide(acc_raw, body_size)

    if len(acc_raw) < 2:
        return speed_raw, acc_raw, np.array([]), speed_norm, acc_norm, np.array([])

    # ----- jerk -----
    jx = np.diff(ax) * fps
    jy = np.diff(ay) * fps
    jerk_raw = np.sqrt(jx**2 + jy**2)
    jerk_norm = safe_divide(jerk_raw, body_size)

    return speed_raw, acc_raw, jerk_raw, speed_norm, acc_norm, jerk_norm


# ====== distance 계산 (raw & normalized) ======
def compute_path_length_raw_norm(x, y, body_size):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)

    valid = (~np.isnan(x)) & (~np.isnan(y))
    x_v = x[valid]
    y_v = y[valid]

    if len(x_v) < 2:
        return np.nan, np.nan

    dx = np.diff(x_v)
    dy = np.diff(y_v)
    d = np.sqrt(dx**2 + dy**2)

    dist_raw = np.sum(d)
    dist_norm = safe_divide(dist_raw, body_size)

    return dist_raw, dist_norm


# ====== Plateau 계산 ======
def compute_plateau_features(y, fps):
    y = np.asarray(y, dtype=float)
    valid = ~np.isnan(y)
    y_v = y[valid]

    if len(y_v) < 2:
        return 0, np.nan, np.nan

    dy = np.abs(np.diff(y_v))
    dt = 1.0 / fps

    plateau_mask = dy < PLATEAU_DY_THRESH
    plateau_mask = np.concatenate([[plateau_mask[0]], plateau_mask])

    durations = []
    in_seg = False
    seg_len = 0

    for flag in plateau_mask:
        if flag:
            if not in_seg:
                in_seg = True
                seg_len = 1
            else:
                seg_len += 1
        else:
            if in_seg:
                dur = seg_len * dt
                if dur >= PLATEAU_MIN_DURATION_SEC:
                    durations.append(dur)
                in_seg = False
                seg_len = 0

    if in_seg:
        dur = seg_len * dt
        if dur >= PLATEAU_MIN_DURATION_SEC:
            durations.append(dur)

    if len(durations) == 0:
        return 0, np.nan, np.nan

    return len(durations), np.mean(durations), np.max(durations)


# ====== Immobility 계산 ======
def compute_immobility_ratio(speed_norm, fps):
    """
    immobility는 정규화된 hip speed로 판단해야
    카메라 거리/배율에 영향을 받지 않음
    """
    if speed_norm.size == 0:
        return np.nan

    dt = 1.0 / fps
    immobile_mask = speed_norm < IMMOBILITY_SPEED_THRESH
    return (immobile_mask.sum() * dt) / (speed_norm.size * dt)


# ====== Stability 계산 ======
def compute_stability(speed_norm):
    """
    stability는 손/발 contact 상태에서 속도 변동성을 측정
    normalized speed 사용이 맞음
    """
    speed_norm = np.asarray(speed_norm)
    mask = speed_norm < STABILITY_SPEED_THRESH
    if mask.sum() < STABILITY_MIN_FRAMES:
        return np.nan
    return np.var(speed_norm[mask])


# ====== Joint Feature Calculator (raw + norm) ======
def joint_feature_block(x, y, fps, body_size):
    """
    return:
        v_mean_raw, v_max_raw, a_rms_raw,
        v_mean_norm, v_max_norm, a_rms_norm,
        stability_raw, stability_norm
    """
    (speed_raw, acc_raw, jerk_raw,
     speed_norm, acc_norm, jerk_norm) = compute_kinematics_raw_norm(x, y, fps, body_size)

    v_mean_raw = np.nanmean(speed_raw) if speed_raw.size > 0 else np.nan
    v_max_raw  = np.nanmax(speed_raw) if speed_raw.size > 0 else np.nan
    a_rms_raw  = rms(acc_raw)

    v_mean_norm = np.nanmean(speed_norm) if speed_norm.size > 0 else np.nan
    v_max_norm  = np.nanmax(speed_norm) if speed_norm.size > 0 else np.nan
    a_rms_norm  = rms(acc_norm)

    stab_raw  = compute_stability(speed_raw)
    stab_norm = compute_stability(speed_norm)

    return (
        v_mean_raw, v_max_raw, a_rms_raw,
        v_mean_norm, v_max_norm, a_rms_norm,
        stab_raw, stab_norm
    )


# ====== 비디오 처리 ======
def process_video(VIDEO_PATH, LABEL=0):
    FILE_ID = os.path.splitext(os.path.basename(VIDEO_PATH))[0]
    OUTPUT_XLSX = os.path.join(OUTPUT_DIR, f'{FILE_ID}.xlsx')

    cap = cv2.VideoCapture(VIDEO_PATH)
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        fps = 30.0

    mp_pose = mp.solutions.pose.Pose(
        static_image_mode=False,
        model_complexity=1,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    )

    # ---- 좌표 저장 ----
    traj = {name: {'x': [], 'y': []} for name in
            ['hipL', 'hipR', 'handL', 'handR', 'footL', 'footR']}

    body_sizes = []
    frame_idx = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_idx % FRAME_INTERVAL == 0:
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = mp_pose.process(rgb)

            if results.pose_landmarks:
                lm = results.pose_landmarks.landmark

                # body_size 추출
                bs = compute_body_size(lm)
                body_sizes.append(bs)

                pairs = {
                    'hipL': 23, 'hipR': 24,
                    'handL': 15, 'handR': 16,
                    'footL': 27, 'footR': 28
                }

                for k, idx in pairs.items():
                    traj[k]['x'].append(lm[idx].x)
                    traj[k]['y'].append(lm[idx].y)

            else:
                # 포즈가 안 잡힌 프레임
                for k in traj.keys():
                    traj[k]['x'].append(np.nan)
                    traj[k]['y'].append(np.nan)

        frame_idx += 1

    cap.release()
    mp_pose.close()

    total_time = frame_idx / fps
    body_size = np.nanmean(body_sizes) if len(body_sizes) > 0 else 1.0

    # ====== HIP Center ======
    hip_x = np.nanmean([traj['hipL']['x'], traj['hipR']['x']], axis=0)
    hip_y = np.nanmean([traj['hipL']['y'], traj['hipR']['y']], axis=0)

    # hip kinematics
    (hip_speed_raw, hip_acc_raw, hip_jerk_raw, hip_speed_norm, hip_acc_norm, hip_jerk_norm) = compute_kinematics_raw_norm(hip_x, hip_y, fps, body_size)

    jerkRMS_raw = rms(hip_jerk_raw)
    jerkRMS_norm = rms(hip_jerk_norm)

    vel_mean_raw = np.nanmean(hip_speed_raw) if hip_speed_raw.size > 0 else np.nan
    vel_mean_norm = np.nanmean(hip_speed_norm) if hip_speed_norm.size > 0 else np.nan

    vel_rms_raw = rms(hip_speed_raw)
    vel_rms_norm = rms(hip_speed_norm)

    acc_rms_raw = rms(hip_acc_raw)
    acc_rms_norm = rms(hip_acc_norm)

    # distance_total
    dist_raw, dist_norm = compute_path_length_raw_norm(hip_x, hip_y, body_size)

    # immobility: normalized speed 기준
    immobile_ratio = compute_immobility_ratio(hip_speed_norm, fps)

    # straight distance
    valid_idx = np.where(~np.isnan(hip_x) & ~np.isnan(hip_y))[0]
    if len(valid_idx) > 1:
        x0, y0 = hip_x[valid_idx[0]], hip_y[valid_idx[0]]
        x1, y1 = hip_x[valid_idx[-1]], hip_y[valid_idx[-1]]
        straight_raw = np.sqrt((x1 - x0)**2 + (y1 - y0)**2)
        straight_norm = straight_raw / body_size if body_size != 0 else np.nan
        path_ineff_raw  = safe_divide(dist_raw,  straight_raw)
        path_ineff_norm = safe_divide(dist_norm, straight_norm)
    else:
        path_ineff_raw = path_ineff_norm = np.nan

    # plateau
    plateau_count, plateau_mean, plateau_max = compute_plateau_features(hip_y, fps)

    # vertical speed
    vy = hip_y[~np.isnan(hip_y)]
    if len(vy) > 1:
        mean_vertical_raw  = (vy[0] - vy[-1]) / total_time
        mean_vertical_norm = mean_vertical_raw / body_size
    else:
        mean_vertical_raw = mean_vertical_norm = np.nan

    # ===== Local joints ======
    def get_joint_features(name):
        x = traj[name]['x']
        y = traj[name]['y']

        return joint_feature_block(x, y, fps, body_size)

    (
        hL_vmean_raw, hL_vmax_raw, hL_arm_r,
        hL_vmean_norm, hL_vmax_norm, hL_arm_n,
        hL_stab_raw, hL_stab_norm
    ) = get_joint_features('handL')

    (
        hR_vmean_raw, hR_vmax_raw, hR_arm_r,
        hR_vmean_norm, hR_vmax_norm, hR_arm_n,
        hR_stab_raw, hR_stab_norm
    ) = get_joint_features('handR')

    (
        fL_vmean_raw, fL_vmax_raw, fL_arm_r,
        fL_vmean_norm, fL_vmax_norm, fL_arm_n,
        fL_stab_raw, fL_stab_norm
    ) = get_joint_features('footL')

    (
        fR_vmean_raw, fR_vmax_raw, fR_arm_r,
        fR_vmean_norm, fR_vmax_norm, fR_arm_n,
        fR_stab_raw, fR_stab_norm
    ) = get_joint_features('footR')

    # ===== 최종 Feature Row ======
    feature_row = {
        'id': FILE_ID,
        'label': LABEL,

        # ===== Global HIP (raw + norm) =====
        'jerkRMS_raw': jerkRMS_raw,
        'jerkRMS_norm': jerkRMS_norm,

        'velocity_mean_raw': vel_mean_raw,
        'velocity_mean_norm': vel_mean_norm,

        'velocity_rms_raw': vel_rms_raw,
        'velocity_rms_norm': vel_rms_norm,

        'acceleration_rms_raw': acc_rms_raw,
        'acceleration_rms_norm': acc_rms_norm,

        'distance_total_raw': dist_raw,
        'distance_total_norm': dist_norm,

        'path_inefficiency_raw': path_ineff_raw,
        'path_inefficiency_norm': path_ineff_norm,

        'mean_vertical_speed_raw': mean_vertical_raw,
        'mean_vertical_speed_norm': mean_vertical_norm,

        'immobility_ratio': immobile_ratio,
        'plateau_count': plateau_count,
        'plateau_duration_mean': plateau_mean,
        'plateau_duration_max': plateau_max,
        'ascent_time': total_time,

        # ===== Hand L (raw + norm) =====
        'hand_velocity_mean_L_raw': hL_vmean_raw,
        'hand_velocity_mean_L_norm': hL_vmean_norm,
        'hand_velocity_max_L_raw': hL_vmax_raw,
        'hand_velocity_max_L_norm': hL_vmax_norm,
        'hand_acceleration_rms_L_raw': hL_arm_r,
        'hand_acceleration_rms_L_norm': hL_arm_n,
        'hand_stability_L_raw': hL_stab_raw,
        'hand_stability_L_norm': hL_stab_norm,

        # ===== Hand R =====
        'hand_velocity_mean_R_raw': hR_vmean_raw,
        'hand_velocity_mean_R_norm': hR_vmean_norm,
        'hand_velocity_max_R_raw': hR_vmax_raw,
        'hand_velocity_max_R_norm': hR_vmax_norm,
        'hand_acceleration_rms_R_raw': hR_arm_r,
        'hand_acceleration_rms_R_norm': hR_arm_n,
        'hand_stability_R_raw': hR_stab_raw,
        'hand_stability_R_norm': hR_stab_norm,

        # ===== Foot L =====
        'foot_velocity_mean_L_raw': fL_vmean_raw,
        'foot_velocity_mean_L_norm': fL_vmean_norm,
        'foot_velocity_max_L_raw': fL_vmax_raw,
        'foot_velocity_max_L_norm': fL_vmax_norm,
        'foot_acceleration_rms_L_raw': fL_arm_r,
        'foot_acceleration_rms_L_norm': fL_arm_n,
        'foot_stability_L_raw': fL_stab_raw,
        'foot_stability_L_norm': fL_stab_norm,

        # ===== Foot R =====
        'foot_velocity_mean_R_raw': fR_vmean_raw,
        'foot_velocity_mean_R_norm': fR_vmean_norm,
        'foot_velocity_max_R_raw': fR_vmax_raw,
        'foot_velocity_max_R_norm': fR_vmax_norm,
        'foot_acceleration_rms_R_raw': fR_arm_r,
        'foot_acceleration_rms_R_norm': fR_arm_n,
        'foot_stability_R_raw': fR_stab_raw,
        'foot_stability_R_norm': fR_stab_norm,
    }

    # 저장
    df = pd.DataFrame([feature_row])
    df.to_excel(OUTPUT_XLSX, index=False, sheet_name='Features')
    print(f"✅ 저장 완료: {OUTPUT_XLSX}")


# ====== 메인 ======
def main():
    files = glob.glob(VIDEO_DIR + '/*.mp4') + glob.glob(VIDEO_DIR + '/*.mov')
    if not files:
        print("❌ 비디오 없음.")
        return

    print(f"📁 총 {len(files)}개 비디오 발견")
    for f in files:
        label = extract_label(os.path.basename(f))
        process_video(f, label)

    print("🎉 전체 완료!")


if __name__ == "__main__":
    main()
