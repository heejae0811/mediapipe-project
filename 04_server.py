import os
import cv2
import joblib
import numpy as np
import pandas as pd
import mediapipe as mp
from flask import Flask, request, jsonify
from flask_cors import CORS

# ==========================================================
# 0. Config
# ==========================================================
FRAME_INTERVAL = 3
HIP_MISSING_RATIO_MAX = 0.30
LIMB_MISSING_RATIO_MAX = 0.40

UPLOAD_DIR = "./temp"
os.makedirs(UPLOAD_DIR, exist_ok=True)

mp_pose = mp.solutions.pose

# ==========================================================
# 1. Load ML Artifacts
# ==========================================================
model = joblib.load("./result/best_model.pkl")
scaler = joblib.load("./result/best_scaler.pkl")
selected_features = joblib.load("./result/best_features.pkl")

# ==========================================================
# 2. Utils
# ==========================================================
def fill_missing(arr):
    return pd.Series(arr, dtype=float).interpolate(limit_direction="both").to_numpy()

def nan_ratio(arr):
    return np.mean(np.isnan(arr))

def center_point(p1, p2):
    return ((p1[0] + p2[0]) / 2, (p1[1] + p2[1]) / 2)

def velocity_series(pts, dt):
    return np.array([
        0.0 if i == 0 else np.linalg.norm(np.array(pts[i]) - np.array(pts[i-1])) / dt
        for i in range(len(pts))
    ])

def acc_series(v, dt):
    return np.array([0.0 if i == 0 else (v[i] - v[i-1]) / dt for i in range(len(v))])

def jerk_series(a, dt):
    return np.array([0.0 if i == 0 else (a[i] - a[i-1]) / dt for i in range(len(a))])

def limb_distance_series(pts):
    return np.array([
        0.0 if i == 0 else np.linalg.norm(np.array(pts[i]) - np.array(pts[i-1]))
        for i in range(len(pts))
    ])

def body_size_from_landmarks(lm):
    pairs = [(11,12), (23,24), (11,23), (12,24)]
    return np.mean([
        np.linalg.norm([lm[i].x - lm[j].x, lm[i].y - lm[j].y])
        for i, j in pairs
    ])

# ==========================================================
# 3. Feature Extraction (FULL)
# ==========================================================
def extract_features(video_path):
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    fps = 30 if fps <= 0 else fps
    dt = 1 / fps

    hip_pts, lf_pts, rf_pts = [], [], []
    body_sizes = []
    frame_idx = 0

    with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            if frame_idx % FRAME_INTERVAL != 0:
                frame_idx += 1
                continue

            h, w = frame.shape[:2]
            res = pose.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

            if res.pose_landmarks:
                lm = res.pose_landmarks.landmark
                body_sizes.append(body_size_from_landmarks(lm))

                hip_pts.append(center_point(
                    (lm[23].x*w, lm[23].y*h),
                    (lm[24].x*w, lm[24].y*h)
                ))
                lf_pts.append((lm[27].x*w, lm[27].y*h))
                rf_pts.append((lm[28].x*w, lm[28].y*h))
            else:
                hip_pts.append((np.nan, np.nan))
                lf_pts.append((np.nan, np.nan))
                rf_pts.append((np.nan, np.nan))

            frame_idx += 1

    cap.release()

    hip_x = fill_missing([p[0] for p in hip_pts])
    hip_y = fill_missing([p[1] for p in hip_pts])

    hip_xy = list(zip(hip_x, hip_y))
    dt_eff = dt * FRAME_INTERVAL
    body_size = np.mean(body_sizes)
    total_time = len(hip_xy) * dt_eff

    hip_v = velocity_series(hip_xy, dt_eff)
    hip_a = acc_series(hip_v, dt_eff)
    hip_j = jerk_series(hip_a, dt_eff)

    path_length = np.sum(hip_v * dt_eff)

    feats = {
        "total_time": total_time,

        "fluency_hip_acc_mean": np.mean(hip_a),
        "fluency_hip_jerk_mean": np.mean(hip_j),
        "fluency_hip_jerk_rms": np.sqrt(np.mean(hip_j**2)),
        "fluency_hip_path_length": path_length,
        "fluency_hip_velocity_mean_norm_body": np.mean(hip_v) / body_size,
        "fluency_hip_jerk_mean_norm_body": np.mean(hip_j) / body_size,
        "fluency_hip_path_length_per_sec_norm_body": path_length / total_time / body_size,

        "stability_hip_velocity_sd_norm_body": np.std(hip_v) / body_size,
        "stability_hip_acc_sd_norm_body": np.std(hip_a) / body_size,
    }

    for name, pts in {"left": lf_pts, "right": rf_pts}.items():
        xs = fill_missing([p[0] for p in pts])
        ys = fill_missing([p[1] for p in pts])
        d = limb_distance_series(list(zip(xs, ys)))
        feats[f"exploration_{name}_foot_distance_mean"] = np.mean(d)
        feats[f"exploration_{name}_foot_distance_mean_norm_body"] = np.mean(d) / body_size

    return feats

# ==========================================================
# 4. Feedback Rules
# ==========================================================
def generate_feedback(feats):
    msgs = []

    if feats["fluency_hip_jerk_rms"] > 0.05:
        msgs.append("움직임이 다소 끊기는 경향이 있어요. 동작을 더 부드럽게 연결해보세요.")
    else:
        msgs.append("전반적으로 부드러운 움직임을 유지하고 있어요.")

    if feats["fluency_hip_path_length"] > 3.0:
        msgs.append("이동 경로가 길어요. 불필요한 움직임을 줄이면 효율이 더 좋아질 거예요.")

    if feats["stability_hip_velocity_sd_norm_body"] > 0.08:
        msgs.append("속도의 변화가 커서 리듬이 흔들릴 수 있어요.")

    if feats["exploration_left_foot_distance_mean_norm_body"] > 0.6:
        msgs.append("왼발 탐색 동작이 많은 편이에요.")

    if feats["exploration_right_foot_distance_mean_norm_body"] > 0.6:
        msgs.append("오른발 위치 선택이 다소 불확실해 보여요.")

    return msgs

# ==========================================================
# 5. Flask API
# ==========================================================
app = Flask(__name__)
CORS(app)

@app.route("/predict", methods=["POST"])
def predict():

    video = request.files["video"]
    path = os.path.join(UPLOAD_DIR, video.filename)
    video.save(path)

    feats = extract_features(path)

    X = pd.DataFrame([feats])[selected_features]
    if scaler is not None:
        X[:] = scaler.transform(X)

    pred = int(model.predict(X)[0])
    prob = float(model.predict_proba(X)[0, 1])

    os.remove(path)

    return jsonify({
        "prediction": {
            "label": "Advanced" if pred == 1 else "Intermediate",
            "probability": round(prob, 3)
        },
        "feature_values": feats,
        "feedback": generate_feedback(feats)
    })

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5001)
