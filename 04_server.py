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
#  1. 최종 버전 특징 추출 코드 (버그 수정됨)
# ====================================================================
FRAME_INTERVAL = 3
MICRO_MOVEMENT_THRESH = 5.0
HIP_MISSING_RATIO_MAX = 0.30
LIMB_MISSING_RATIO_MAX = 0.40
mp_pose = mp.solutions.pose


# --- Helper Functions ---
def fill_missing(arr):
    return pd.Series(arr, dtype="float").interpolate(limit_direction="both").to_numpy()


def nan_ratio(arr):
    return np.mean(np.isnan(np.asarray(arr, dtype=float)))


def center_point(p1, p2):
    return ((p1[0] + p2[0]) / 2.0, (p1[1] + p2[1]) / 2.0)


def velocity_series(pts, dt):
    v = [0.0]
    for t in range(1, len(pts)):
        dx, dy = pts[t][0] - pts[t - 1][0], pts[t][1] - pts[t - 1][1]
        v.append(np.sqrt(dx ** 2 + dy ** 2) / dt)
    return np.array(v, dtype=float)


def acc_series(v, dt):
    a = [0.0]
    for t in range(1, len(v)):
        a.append((v[t] - v[t - 1]) / dt)
    return np.array(a, dtype=float)


def jerk_series(a, dt):
    j = [0.0]
    for t in range(1, len(a)):
        j.append((a[t] - a[t - 1]) / dt)
    return np.array(j, dtype=float)


def body_size_from_landmarks(lm):
    def dist(i, j):
        return np.sqrt((lm[i].x - lm[j].x) ** 2 + (lm[i].y - lm[j].y) ** 2)

    pairs = [(11, 12), (23, 24), (11, 23), (12, 24)]
    vals = [dist(i, j) for i, j in pairs if not np.isnan(dist(i, j))]
    return np.mean(vals) if len(vals) > 0 else 1.0


def limb_distance_series(pts):
    d = [0.0]
    for t in range(1, len(pts)):
        dx, dy = pts[t][0] - pts[t - 1][0], pts[t][1] - pts[t - 1][1]
        d.append(np.sqrt(dx ** 2 + dy ** 2))
    return np.array(d, dtype=float)


def exploration_features(d, micro_th=MICRO_MOVEMENT_THRESH):
    d = np.asarray(d, dtype=float)
    total_mov = np.sum(d > 0)
    distance_mean = float(np.mean(d)) if d.size > 0 else np.nan
    micro_mask = d < micro_th
    micro_sum = float(np.sum(d[micro_mask])) if d.size > 0 else np.nan
    return_dist = float(np.sum(np.abs(d[micro_mask]))) if d.size > 0 else np.nan
    ratio = float(np.sum(micro_mask & (d > 0)) / total_mov) if total_mov > 0 else np.nan
    return distance_mean, micro_sum, return_dist, ratio


# --- Main Feature Extraction Function ---
def extract_features(video_path):
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    dt = 1.0 / fps
    hip_pts, lh_pts, rh_pts, lf_pts, rf_pts, body_sizes, frame_idx = [], [], [], [], [], [], 0

    with mp_pose.Pose(static_image_mode=False, model_complexity=1) as pose:
        while True:
            ret, frame = cap.read()
            if not ret: break
            if frame_idx % FRAME_INTERVAL != 0:
                frame_idx += 1;
                continue

            h, w = frame.shape[:2]
            result = pose.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

            if result.pose_landmarks:
                lm = result.pose_landmarks.landmark
                body_sizes.append(body_size_from_landmarks(lm))
                L_HIP, R_HIP = (lm[23].x * w, lm[23].y * h), (lm[24].x * w, lm[24].y * h)
                hip_pts.append(center_point(L_HIP, R_HIP))
                lh_pts.append((lm[15].x * w, lm[15].y * h));
                rh_pts.append((lm[16].x * w, lm[16].y * h))
                lf_pts.append((lm[27].x * w, lm[27].y * h));
                rf_pts.append((lm[28].x * w, lm[28].y * h))
            else:
                for p_list in [hip_pts, lh_pts, rh_pts, lf_pts, rf_pts]: p_list.append((np.nan, np.nan))
            frame_idx += 1
    cap.release()

    if len(hip_pts) < 2: return None
    hip_x, hip_y = np.array([p[0] for p in hip_pts], dtype=float), np.array([p[1] for p in hip_pts], dtype=float)
    if nan_ratio(hip_x) > HIP_MISSING_RATIO_MAX or nan_ratio(hip_y) > HIP_MISSING_RATIO_MAX: return None
    hip_x, hip_y = fill_missing(hip_x), fill_missing(hip_y)
    hip_xy = list(zip(hip_x, hip_y))

    limb_dict = {"left_hand": lh_pts, "right_hand": rh_pts, "left_foot": lf_pts, "right_foot": rf_pts}
    for name, pts in limb_dict.items():
        xs, ys = np.array([p[0] for p in pts], dtype=float), np.array([p[1] for p in pts], dtype=float)
        if nan_ratio(xs) > LIMB_MISSING_RATIO_MAX or nan_ratio(ys) > LIMB_MISSING_RATIO_MAX:
            limb_dict[name] = None
        else:
            limb_dict[name] = list(zip(fill_missing(xs), fill_missing(ys)))

    dt_eff = dt * FRAME_INTERVAL
    body_size = np.mean(body_sizes) if body_sizes else 1.0
    total_time = len(hip_xy) * dt_eff

    hip_v = velocity_series(hip_xy, dt_eff);
    hip_a = acc_series(hip_v, dt_eff);
    hip_j = jerk_series(hip_a, dt_eff)
    path_length = float(np.sum(hip_v * dt_eff))
    straight_distance = float(np.sqrt((hip_x[-1] - hip_x[0]) ** 2 + (hip_y[-1] - hip_y[0]) ** 2))
    path_efficiency = float(straight_distance / (path_length + 1e-6))
    immobile_mask = hip_v < 1.0
    immobility_time, immobility_ratio = float(np.sum(immobile_mask) * dt_eff), float(np.sum(immobile_mask) / len(hip_v))

    fluency = {"fluency_hip_velocity_mean": np.mean(hip_v), "fluency_hip_velocity_max": np.max(hip_v),
               "fluency_hip_acc_mean": np.mean(hip_a), "fluency_hip_acc_max": np.max(hip_a),
               "fluency_hip_jerk_mean": np.mean(hip_j), "fluency_hip_jerk_max": np.max(hip_j),
               "fluency_hip_path_length": path_length, "fluency_hip_straight_distance": straight_distance,
               "fluency_hip_path_efficiency": path_efficiency, "fluency_hip_immobility_time": immobility_time,
               "fluency_hip_immobility_ratio": immobility_ratio,
               "fluency_hip_velocity_mean_norm_body": np.mean(hip_v) / body_size,
               "fluency_hip_acc_mean_norm_body": np.mean(hip_a) / body_size,
               "fluency_hip_jerk_mean_norm_body": np.mean(hip_j) / body_size,
               "fluency_hip_path_length_per_sec": path_length / total_time}

    limb_feats = {}
    for name in ["left_hand", "right_hand", "left_foot", "right_foot"]:
        pts = limb_dict[name]
        keys = [f"exploration_{name}_{k}" for k in
                ["distance_mean", "micro_sum", "return_distance", "ratio", "distance_mean_norm_body",
                 "micro_sum_per_sec", "return_distance_per_sec"]]
        if pts is None:
            for k in keys: limb_feats[k] = np.nan
            continue
        d = limb_distance_series(pts)
        dist_mean, micro_sum, ret_dist, ratio = exploration_features(d)
        limb_feats.update(dict(zip(keys, [dist_mean, micro_sum, ret_dist, ratio, dist_mean / body_size,
                                          micro_sum / total_time, ret_dist / total_time])))

    stability = {"stability_hip_velocity_sd": np.std(hip_v), "stability_hip_acc_sd": np.std(hip_a),
                 "stability_hip_jerk_sd": np.std(hip_j),
                 "stability_hip_velocity_sd_norm_body": np.std(hip_v) / body_size,
                 "stability_hip_acc_sd_norm_body": np.std(hip_a) / body_size,
                 "stability_hip_jerk_sd_norm_body": np.std(hip_j) / body_size}

    control = {"control_hip_smoothness": 1.0 / (np.std(hip_j) + 1e-6),
               "control_hip_smoothness_norm_body": (1.0 / (np.std(hip_j) + 1e-6)) / body_size}
    for name in ["left_hand", "right_hand", "left_foot", "right_foot"]:
        pts, key_r, key_n = limb_dict[name], f"control_{name}_smoothness", f"control_{name}_smoothness_norm_body"
        if pts is None:
            control[key_r], control[key_n] = np.nan, np.nan
        else:
            # --- [BUG FIX] 아래 코드를 순차적으로 계산하도록 수정했습니다 ---
            v = velocity_series(pts, dt_eff)
            a = acc_series(v, dt_eff)
            j = jerk_series(a, dt_eff)
            sm = 1.0 / (np.std(j) + 1e-6)
            control[key_r], control[key_n] = sm, sm / body_size

    ascent_speed = (hip_y[0] - hip_y[-1]) / total_time
    speed = {"speed_hip_ascent_speed": ascent_speed, "speed_hip_ascent_speed_norm_body": ascent_speed / body_size}

    feats = {"total_time": total_time, **fluency, **limb_feats, **stability, **control, **speed}
    return pd.DataFrame([feats])


# ====================================================================
#  2. 최종 Flask 서버 코드 (이전과 동일)
# ====================================================================
app = Flask(__name__)
CORS(app)

try:
    print("🧠 모델, 스케일러, 피처 리스트를 로드합니다...")
    model = joblib.load("./result/best_model.pkl")
    scaler = joblib.load("./result/scaler.pkl")
    selected_features = joblib.load("./result/selected_features.pkl")
    print(f"✅ 로드 성공! (필요한 특징 수: {len(selected_features)}개)")
except FileNotFoundError as e:
    print(f"❌ 치명적 에러: '{e.filename}' 파일을 찾을 수 없습니다. 먼저 머신러닝 파이프라인을 실행하세요.")
    model, scaler, selected_features = None, None, None


@app.route("/predict", methods=["POST"])
def predict():
    if not all([model, scaler, selected_features]):
        return jsonify({"error": "서버가 올바르게 초기화되지 않았습니다. 서버 로그를 확인하세요."}), 500
    if 'video' not in request.files:
        return jsonify({"error": "요청에 'video' 파일이 없습니다."}), 400

    video_file = request.files['video'];
    temp_video_path = f"temp_{video_file.filename}";
    video_file.save(temp_video_path)
    print(f"\n📥 '{video_file.filename}' 파일 수신 완료.")

    try:
        print("🛠️ 영상 분석 및 특징 추출 시작...");
        all_features_df = extract_features(temp_video_path)
        if all_features_df is None:
            return jsonify({"error": "영상 분석 중 특징을 추출하지 못했습니다 (결측치 과다 등)."}), 500
        print("✅ 특징 추출 완료.")

        print(f"✨ {len(selected_features)}개의 주요 특징 선택 및 정렬 중...");
        predict_df = all_features_df.reindex(columns=selected_features).fillna(0)

        print("📏 스케일링 적용 중...");
        predict_df_scaled = scaler.transform(predict_df)

        print("🤖 모델 예측 수행...");
        prediction_result = model.predict(predict_df_scaled)[0]
        prediction_proba = model.predict_proba(predict_df_scaled)[0]
        result_label = 'Good' if prediction_result == 1 else 'Bad'
        confidence = prediction_proba[np.where(model.classes_ == prediction_result)[0][0]]
        print(f"👍 예측 결과: {result_label} (신뢰도: {confidence:.2f})")

        analysis_data = all_features_df.iloc[0].to_dict()
        gpt_prompt_data = {
            "path_inefficiency": round(analysis_data.get('fluency_hip_path_efficiency', 0), 2),
            "immobility_ratio": round(analysis_data.get('fluency_hip_immobility_ratio', 0), 2),
            "jerk_mean": round(analysis_data.get('fluency_hip_jerk_mean_norm_body', 0), 2),
            "ascent_speed": round(analysis_data.get('speed_hip_ascent_speed_norm_body', 0), 2)
        }

        return jsonify(
            {"prediction": result_label, "confidence": float(confidence), "gpt_prompt_data": gpt_prompt_data})
    except Exception as e:
        error_message = f"예측 처리 중 서버 에러 발생: {e}";
        print(f"🔥 {error_message}")
        return jsonify({"error": error_message}), 500
    finally:
        if os.path.exists(temp_video_path):
            os.remove(temp_video_path)
            print(f"🗑️ 임시 파일 '{temp_video_path}' 삭제 완료.")


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5001, debug=False)