import os
import re
import cv2
import joblib
import numpy as np
import pandas as pd
import mediapipe as mp
from flask import Flask, request, jsonify
from flask_cors import CORS

# ====================================================================
#  1. 특징 추출 코드 (제공해주신 새 버전 기반)
# ====================================================================
# ... (이전과 동일한 특징 추출 Helper 함수들) ...
# ------ Thresholds ------
IMMOBILITY_SPEED_THRESH = 0.01
PLATEAU_DY_THRESH = 0.002
PLATEAU_MIN_DURATION_SEC = 0.5
STABILITY_SPEED_THRESH = 0.02
STABILITY_MIN_FRAMES = 5


# ------ Helper Functions ------
def safe_divide(num, den, default=np.nan):
    if den is None or den == 0 or np.isnan(den): return default
    return num / den


def rms(arr):
    arr = arr[~np.isnan(np.asarray(arr, dtype=float))]
    return np.sqrt(np.mean(arr ** 2)) if arr.size > 0 else np.nan


def compute_body_size(lm):
    def dist(a, b):
        if any(np.isnan([a.x, a.y, b.x, b.y])): return np.nan
        return np.sqrt((a.x - b.x) ** 2 + (a.y - b.y) ** 2)

    vals = [d for d in [dist(lm[11], lm[23]), dist(lm[12], lm[24]), dist(lm[11], lm[12]), dist(lm[23], lm[24])] if
            not np.isnan(d)]
    return np.mean(vals) if len(vals) > 0 else 1.0


def compute_kinematics_raw_norm(x, y, fps, body_size):
    x, y = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
    valid = (~np.isnan(x)) & (~np.isnan(y));
    x_v, y_v = x[valid], y[valid]
    if len(x_v) < 2: return (np.array([]),) * 6
    vx, vy = np.diff(x_v) * fps, np.diff(y_v) * fps;
    speed_raw = np.sqrt(vx ** 2 + vy ** 2);
    speed_norm = safe_divide(speed_raw, body_size)
    if len(speed_raw) < 2: return speed_raw, np.array([]), np.array([]), speed_norm, np.array([]), np.array([])
    ax, ay = np.diff(vx) * fps, np.diff(vy) * fps;
    acc_raw = np.sqrt(ax ** 2 + ay ** 2);
    acc_norm = safe_divide(acc_raw, body_size)
    if len(acc_raw) < 2: return speed_raw, acc_raw, np.array([]), speed_norm, acc_norm, np.array([])
    jx, jy = np.diff(ax) * fps, np.diff(ay) * fps;
    jerk_raw = np.sqrt(jx ** 2 + jy ** 2);
    jerk_norm = safe_divide(jerk_raw, body_size)
    return speed_raw, acc_raw, jerk_raw, speed_norm, acc_norm, jerk_norm


def compute_path_length_raw_norm(x, y, body_size):
    x, y = np.asarray(x, dtype=float), np.asarray(y, dtype=float);
    valid = (~np.isnan(x)) & (~np.isnan(y));
    x_v, y_v = x[valid], y[valid]
    if len(x_v) < 2: return np.nan, np.nan
    dist_raw = np.sum(np.sqrt(np.diff(x_v) ** 2 + np.diff(y_v) ** 2));
    return dist_raw, safe_divide(dist_raw, body_size)


def compute_plateau_features(y, fps):
    y = y[~np.isnan(np.asarray(y, dtype=float))];
    if len(y) < 2: return 0, np.nan, np.nan
    dy, dt, durations, in_seg, seg_len = np.abs(np.diff(y)), 1.0 / fps, [], False, 0
    for flag in np.concatenate([[False], dy < PLATEAU_DY_THRESH]):
        if flag:
            seg_len += 1; in_seg = True
        elif in_seg:
            dur = seg_len * dt
            if dur >= PLATEAU_MIN_DURATION_SEC: durations.append(dur)
            in_seg, seg_len = False, 0
    if in_seg and (seg_len * dt >= PLATEAU_MIN_DURATION_SEC): durations.append(seg_len * dt)
    return len(durations), np.mean(durations) if durations else np.nan, np.max(durations) if durations else np.nan


def compute_immobility_ratio(speed_norm, fps):
    if speed_norm.size == 0: return np.nan
    return (speed_norm < IMMOBILITY_SPEED_THRESH).sum() / speed_norm.size


def compute_stability(speed_norm):
    mask = speed_norm < STABILITY_SPEED_THRESH
    if mask.sum() < STABILITY_MIN_FRAMES: return np.nan
    return np.var(speed_norm[mask])


def joint_feature_block(x, y, fps, body_size):
    s_r, a_r, j_r, s_n, a_n, j_n = compute_kinematics_raw_norm(x, y, fps, body_size)
    return (np.nanmean(s_r) if s_r.size > 0 else np.nan, np.nanmax(s_r) if s_r.size > 0 else np.nan, rms(a_r),
            np.nanmean(s_n) if s_n.size > 0 else np.nan, np.nanmax(s_n) if s_n.size > 0 else np.nan, rms(a_n),
            compute_stability(s_r), compute_stability(s_n))


def extract_features_from_video(VIDEO_PATH):
    cap = cv2.VideoCapture(VIDEO_PATH)
    fps = cap.get(cv2.CAP_PROP_FPS) if cap.get(cv2.CAP_PROP_FPS) > 0 else 30.0
    mp_pose = mp.solutions.pose.Pose(static_image_mode=False, model_complexity=1)
    traj, body_sizes, frame_idx = {name: {'x': [], 'y': []} for name in
                                   ['hipL', 'hipR', 'handL', 'handR', 'footL', 'footR']}, [], 0
    while True:
        ret, frame = cap.read();
        if not ret: break
        results = mp_pose.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        if results.pose_landmarks:
            lm = results.pose_landmarks.landmark
            body_sizes.append(compute_body_size(lm))
            for k, idx in {'hipL': 23, 'hipR': 24, 'handL': 15, 'handR': 16, 'footL': 27, 'footR': 28}.items():
                traj[k]['x'].append(lm[idx].x);
                traj[k]['y'].append(lm[idx].y)
        else:
            for k in traj.keys(): traj[k]['x'].append(np.nan); traj[k]['y'].append(np.nan)
        frame_idx += 1
    cap.release();
    mp_pose.close()

    total_time, body_size = frame_idx / fps, np.nanmean(body_sizes) if body_sizes else 1.0
    hip_x, hip_y = np.nanmean([traj['hipL']['x'], traj['hipR']['x']], axis=0), np.nanmean(
        [traj['hipL']['y'], traj['hipR']['y']], axis=0)
    hs_r, ha_r, hj_r, hs_n, ha_n, hj_n = compute_kinematics_raw_norm(hip_x, hip_y, fps, body_size)
    dist_r, dist_n = compute_path_length_raw_norm(hip_x, hip_y, body_size)
    valid_idx = np.where(~np.isnan(hip_x) & ~np.isnan(hip_y))[0]

    # --- [BUG FIX] 아래 코드를 수정했습니다 ---
    if len(valid_idx) > 1:
        x0, y0, x1, y1 = hip_x[valid_idx[0]], hip_y[valid_idx[0]], hip_x[valid_idx[-1]], hip_y[valid_idx[-1]]
        straight_r = np.sqrt((x1 - x0) ** 2 + (y1 - y0) ** 2)  # 1. 먼저 계산
        straight_n = safe_divide(straight_r, body_size)  # 2. 그 다음 사용
        path_ineff_r = safe_divide(dist_r, straight_r)
        path_ineff_n = safe_divide(dist_n, straight_n)
    else:
        straight_r, straight_n, path_ineff_r, path_ineff_n = (np.nan,) * 4

    vy = hip_y[~np.isnan(hip_y)]
    if len(vy) > 1:
        mean_vertical_raw = (vy[0] - vy[-1]) / total_time
        mean_vertical_norm = safe_divide(mean_vertical_raw, body_size)
    else:
        mean_vertical_raw, mean_vertical_norm = np.nan, np.nan

    hL_f, hR_f, fL_f, fR_f = (joint_feature_block(traj[name]['x'], traj[name]['y'], fps, body_size) for name in
                              ['handL', 'handR', 'footL', 'footR'])

    feature_row = {'jerkRMS_raw': rms(hj_r), 'jerkRMS_norm': rms(hj_n),
                   'velocity_mean_raw': np.nanmean(hs_r) if hs_r.size > 0 else np.nan,
                   'velocity_mean_norm': np.nanmean(hs_n) if hs_n.size > 0 else np.nan, 'velocity_rms_raw': rms(hs_r),
                   'velocity_rms_norm': rms(hs_n), 'acceleration_rms_raw': rms(ha_r),
                   'acceleration_rms_norm': rms(ha_n), 'distance_total_raw': dist_r, 'distance_total_norm': dist_n,
                   'path_inefficiency_raw': path_ineff_r, 'path_inefficiency_norm': path_ineff_n,
                   'mean_vertical_speed_raw': mean_vertical_raw, 'mean_vertical_speed_norm': mean_vertical_norm,
                   'immobility_ratio': compute_immobility_ratio(hs_n, fps),
                   'plateau_count': compute_plateau_features(hip_y, fps)[0],
                   'plateau_duration_mean': compute_plateau_features(hip_y, fps)[1],
                   'plateau_duration_max': compute_plateau_features(hip_y, fps)[2], 'ascent_time': total_time}
    for name, feats in zip(['hand_L', 'hand_R', 'foot_L', 'foot_R'], [hL_f, hR_f, fL_f, fR_f]):
        for i, suffix in enumerate(['velocity_mean', 'velocity_max', 'acceleration_rms', 'stability']):
            feature_row[f'{name}_{suffix}_raw'] = feats[i * 2]
            feature_row[f'{name}_{suffix}_norm'] = feats[i * 2 + 1]
    return pd.DataFrame([feature_row])


# ====================================================================
#  2. Flask 서버 코드 (이전과 동일)
# ====================================================================
app = Flask(__name__)
CORS(app)
MODEL_PATH, FEATURES_PATH = "./result/best_model.pkl", "./result/selected_features.pkl"
print("🧠 모델과 특징 리스트를 로드합니다...")
try:
    model = joblib.load(MODEL_PATH)
    selected_features = joblib.load(FEATURES_PATH)
    print(f"✅ 로드 성공!\n   - 모델: {MODEL_PATH}\n   - 필요한 특징 수: {len(selected_features)}개")
except FileNotFoundError:
    print(f"❌ 에러: 모델 또는 특징 파일을 찾을 수 없습니다. 경로를 확인하세요.")
    model, selected_features = None, None


@app.route("/predict", methods=["POST"])
def predict():
    if not model or not selected_features: return jsonify({"error": "모델이 로드되지 않았습니다."}), 500
    if 'video' not in request.files: return jsonify({"error": "요청에 'video' 파일이 없습니다."}), 400
    video_file = request.files['video'];
    temp_video_path = f"temp_{video_file.filename}";
    video_file.save(temp_video_path)
    print(f"\n📥 '{video_file.filename}' 파일 수신 완료.")
    try:
        print("🛠️ 영상 분석 및 특징 추출 시작...");
        all_features_df = extract_features_from_video(temp_video_path);
        print("✅ 특징 추출 완료.")
        if all_features_df.empty: return jsonify({"error": "영상 분석 중 특징을 추출하지 못했습니다."}), 500
        print(f"✨ {len(selected_features)}개의 주요 특징 선택 및 정렬 중...");
        predict_df = all_features_df.reindex(columns=selected_features).fillna(0)
        print("🤖 모델 예측 수행...");
        prediction_result = model.predict(predict_df)[0];
        prediction_proba = model.predict_proba(predict_df)[0]
        result_label = 'Good' if prediction_result == 1 else 'Bad'
        confidence = prediction_proba[np.where(model.classes_ == prediction_result)[0][0]]
        print(f"👍 예측 결과: {result_label} (신뢰도: {confidence:.2f})")
        return jsonify({"prediction": result_label, "confidence": float(confidence)})
    except Exception as e:
        error_message = f"예측 처리 중 서버 에러 발생: {e}";
        print(f"🔥 {error_message}");
        return jsonify({"error": error_message}), 500
    finally:
        if os.path.exists(temp_video_path): os.remove(temp_video_path); print(f"🗑️ 임시 파일 '{temp_video_path}' 삭제 완료.")


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5001, debug=False)