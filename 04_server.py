import os
import re
import cv2
import joblib
import numpy as np
import pandas as pd
import mediapipe as mp
from flask import Flask, request, jsonify
from flask_cors import CORS

# -----------------------------
#  Feature Extraction (너의 최신 코드)
# -----------------------------
FRAME_INTERVAL = 3
MICRO_MOVEMENT_THRESH = 5.0
HIP_MISSING_RATIO_MAX = 0.30
LIMB_MISSING_RATIO_MAX = 0.40
mp_pose = mp.solutions.pose

# ==== Helper Functions =====================================
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
    return np.array([0.0] + [(v[t] - v[t - 1]) / dt for t in range(1, len(v))], dtype=float)

def jerk_series(a, dt):
    return np.array([0.0] + [(a[t] - a[t - 1]) / dt for t in range(1, len(a))], dtype=float)

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

# ==== Feature Extractor =====================================
def extract_features(video_path):
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    dt = 1.0 / fps

    hip_pts, lh_pts, rh_pts, lf_pts, rf_pts = [], [], [], [], []
    body_sizes = []
    frame_idx = 0

    with mp_pose.Pose(static_image_mode=False, model_complexity=1) as pose:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            if frame_idx % FRAME_INTERVAL != 0:
                frame_idx += 1
                continue

            h, w = frame.shape[:2]
            result = pose.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

            if result.pose_landmarks:
                lm = result.pose_landmarks.landmark
                body_sizes.append(body_size_from_landmarks(lm))

                L_HIP = (lm[23].x * w, lm[23].y * h)
                R_HIP = (lm[24].x * w, lm[24].y * h)
                hip_pts.append(center_point(L_HIP, R_HIP))

                lh_pts.append((lm[15].x * w, lm[15].y * h))
                rh_pts.append((lm[16].x * w, lm[16].y * h))
                lf_pts.append((lm[27].x * w, lm[27].y * h))
                rf_pts.append((lm[28].x * w, lm[28].y * h))
            else:
                hip_pts.append((np.nan, np.nan))
                lh_pts.append((np.nan, np.nan))
                rh_pts.append((np.nan, np.nan))
                lf_pts.append((np.nan, np.nan))
                rf_pts.append((np.nan, np.nan))

            frame_idx += 1

    cap.release()

    if len(hip_pts) < 2:
        return None

    # HIP 결측치 처리
    hip_x = fill_missing([p[0] for p in hip_pts])
    hip_y = fill_missing([p[1] for p in hip_pts])
    hip_xy = list(zip(hip_x, hip_y))

    if nan_ratio(hip_x) > HIP_MISSING_RATIO_MAX or nan_ratio(hip_y) > HIP_MISSING_RATIO_MAX:
        return None

    # Limb 처리
    limb_dict = {"left_hand": lh_pts, "right_hand": rh_pts, "left_foot": lf_pts, "right_foot": rf_pts}
    for limb, pts in limb_dict.items():
        xs = np.array([p[0] for p in pts], dtype=float)
        ys = np.array([p[1] for p in pts], dtype=float)
        if nan_ratio(xs) > LIMB_MISSING_RATIO_MAX or nan_ratio(ys) > LIMB_MISSING_RATIO_MAX:
            limb_dict[limb] = None
        else:
            limb_dict[limb] = list(zip(fill_missing(xs), fill_missing(ys)))

    dt_eff = dt * FRAME_INTERVAL
    body_size = np.mean(body_sizes) if body_sizes else 1.0
    total_time = len(hip_xy) * dt_eff

    # ==== FLUENCY ====
    hip_v = velocity_series(hip_xy, dt_eff)
    hip_a = acc_series(hip_v, dt_eff)
    hip_j = jerk_series(hip_a, dt_eff)

    path_length = float(np.sum(hip_v * dt_eff))
    straight_distance = float(np.sqrt((hip_x[-1]-hip_x[0])**2 + (hip_y[-1]-hip_y[0])**2))
    path_efficiency = float(straight_distance / (path_length + 1e-6))

    immobile_mask = hip_v < 1.0
    immobility_ratio = float(np.sum(immobile_mask) / len(hip_v))

    fluency = {
        "fluency_hip_path_length": path_length,
        "fluency_hip_path_efficiency": path_efficiency,
        "fluency_hip_immobility_ratio": immobility_ratio,
    }

    # ==== EXPLORATION ====
    limb_feats = {}
    for limb, pts in limb_dict.items():
        prefix = f"exploration_{limb}"
        if pts is None:
            limb_feats[f"{prefix}_distance_mean_norm_body"] = np.nan
            continue
        d = limb_distance_series(pts)
        mean_dist = float(np.mean(d))
        limb_feats[f"{prefix}_distance_mean_norm_body"] = mean_dist / body_size

    # ==== STABILITY ====
    stability = {
        "stability_hip_velocity_sd_norm_body": float(np.std(hip_v) / body_size)
    }

    # ==== SPEED ====
    ascent_speed = (hip_y[0] - hip_y[-1]) / total_time
    speed = {
        "speed_hip_ascent_speed": float(ascent_speed),
        "speed_hip_ascent_speed_norm_body": float(ascent_speed / body_size),
    }

    return pd.DataFrame([{
        "total_time": total_time,
        **fluency,
        **limb_feats,
        **stability,
        **speed
    }])


# -----------------------------
#    FLASK SERVER
# -----------------------------
app = Flask(__name__)
CORS(app)

try:
    print("🧠 Loading model/scaler/features...")
    model = joblib.load("./result/best_model.pkl")
    scaler = joblib.load("./result/best_scaler.pkl")
    selected_features = joblib.load("./result/best_features.pkl")
    print("✅ Model loaded successfully!")
except Exception as e:
    print("❌ Error:", e)
    model, scaler, selected_features = None, None, None


@app.route("/predict", methods=["POST"])
def predict():
    if model is None:
        return jsonify({"error": "Server model not initialized."}), 500

    if "video" not in request.files:
        return jsonify({"error": "'video' 파일이 없습니다."}), 400

    video = request.files["video"]
    temp_path = f"temp_{video.filename}"
    video.save(temp_path)

    try:
        # Feature extraction
        feats_df = extract_features(temp_path)
        if feats_df is None:
            return jsonify({"error": "Feature extraction 실패 (결측치 과다 등)"}), 500

        analysis_data = feats_df.iloc[0].to_dict()

        # ML용 피처만 정렬
        ml_df = feats_df.reindex(columns=selected_features).fillna(0)

        if scaler:
            X = scaler.transform(ml_df)
        else:
            X = ml_df.values

        # Predict
        pred = model.predict(X)[0]
        proba = model.predict_proba(X)[0]
        confidence = float(proba[pred])
        label = "Advanced" if pred == 1 else "Intermediate"

        # ---- UI 필수 요소 ----
        ui_data = {
            "total_time": float(analysis_data["total_time"]),
            "hip_path_length": float(analysis_data["fluency_hip_path_length"]),
            "hip_ascent_speed": float(analysis_data["speed_hip_ascent_speed"]),
        }

        # ---- GPT 프롬프트용 ----
        exploration_mean = np.nanmean([
            analysis_data.get("exploration_left_hand_distance_mean_norm_body", np.nan),
            analysis_data.get("exploration_right_hand_distance_mean_norm_body", np.nan),
            analysis_data.get("exploration_left_foot_distance_mean_norm_body", np.nan),
            analysis_data.get("exploration_right_foot_distance_mean_norm_body", np.nan),
        ])

        gpt_prompt_data = {
            "path_efficiency": round(analysis_data["fluency_hip_path_efficiency"], 3),
            "immobility_ratio": round(analysis_data["fluency_hip_immobility_ratio"], 3),
            "jerk_mean_norm": round(analysis_data.get("fluency_hip_jerk_mean_norm_body", 0), 3),
            "ascent_speed_norm": round(analysis_data["speed_hip_ascent_speed_norm_body"], 3),
            "exploration_mean_norm": round(float(exploration_mean), 3),
            "stability_velocity_sd_norm": round(analysis_data["stability_hip_velocity_sd_norm_body"], 3),
        }

        return jsonify({
            "prediction": label,
            "confidence": confidence,
            "ui_data": ui_data,
            "gpt_prompt_data": gpt_prompt_data
        })

    except Exception as e:
        return jsonify({"error": f"Server error: {e}"}), 500

    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5001, debug=False)
