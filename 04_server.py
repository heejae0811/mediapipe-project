import os
import cv2
import joblib
import numpy as np
import pandas as pd
import mediapipe as mp
from flask import Flask, request, jsonify
from flask_cors import CORS

# ==========================================================
# 0. 설정
# ==========================================================
FRAME_INTERVAL = 3
HIP_MISSING_RATIO_MAX = 0.30
LIMB_MISSING_RATIO_MAX = 0.40

UPLOAD_DIR = "./temp"
os.makedirs(UPLOAD_DIR, exist_ok=True)

mp_pose = mp.solutions.pose

# ==========================================================
# 1. ML 모델 로드
# ==========================================================
print("🔹 Loading ML artifacts...")

model = joblib.load("./result/best_model.pkl")
scaler = joblib.load("./result/best_scaler.pkl")
selected_features = joblib.load("./result/best_features.pkl")

print(f"✔ Model loaded | #features = {len(selected_features)}")

# ==========================================================
# 2. Feature Extraction Utils
# ==========================================================
def fill_missing(arr):
    return pd.Series(arr, dtype=float).interpolate(limit_direction="both").to_numpy()

def nan_ratio(arr):
    return np.mean(np.isnan(arr))

def center_point(p1, p2):
    return ((p1[0] + p2[0]) / 2, (p1[1] + p2[1]) / 2)

def velocity_series(pts, dt):
    v = [0.0]
    for i in range(1, len(pts)):
        dx = pts[i][0] - pts[i - 1][0]
        dy = pts[i][1] - pts[i - 1][1]
        v.append(np.sqrt(dx**2 + dy**2) / dt)
    return np.array(v)

def limb_distance_series(pts):
    d = [0.0]
    for i in range(1, len(pts)):
        dx = pts[i][0] - pts[i - 1][0]
        dy = pts[i][1] - pts[i - 1][1]
        d.append(np.sqrt(dx**2 + dy**2))
    return np.array(d)

def body_size_from_landmarks(lm):
    pairs = [(11,12), (23,24), (11,23), (12,24)]
    vals = [
        np.sqrt((lm[i].x - lm[j].x)**2 + (lm[i].y - lm[j].y)**2)
        for i, j in pairs
    ]
    return np.mean(vals) if vals else 1.0

# ==========================================================
# 3. Feature Extraction (ML 학습과 동일)
# ==========================================================
def extract_features(video_path):

    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    fps = 30 if fps <= 0 else fps
    dt = 1 / fps

    hip_pts, lh_pts, rh_pts, lf_pts, rf_pts = [], [], [], [], []
    body_sizes = []
    frame_idx = 0

    with mp_pose.Pose(
        static_image_mode=False,
        model_complexity=1,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    ) as pose:

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

                lh_pts.append((lm[15].x*w, lm[15].y*h))
                rh_pts.append((lm[16].x*w, lm[16].y*h))
                lf_pts.append((lm[27].x*w, lm[27].y*h))
                rf_pts.append((lm[28].x*w, lm[28].y*h))
            else:
                for lst in [hip_pts, lh_pts, rh_pts, lf_pts, rf_pts]:
                    lst.append((np.nan, np.nan))

            frame_idx += 1

    cap.release()

    if len(hip_pts) < 2:
        return None

    hip_x = fill_missing([p[0] for p in hip_pts])
    hip_y = fill_missing([p[1] for p in hip_pts])
    if nan_ratio(hip_x) > HIP_MISSING_RATIO_MAX:
        return None

    hip_xy = list(zip(hip_x, hip_y))
    dt_eff = dt * FRAME_INTERVAL
    body_size = np.mean(body_sizes)
    total_time = len(hip_xy) * dt_eff

    hip_v = velocity_series(hip_xy, dt_eff)
    path_length = np.sum(hip_v * dt_eff)

    feats = {
        "total_time": total_time,
        "fluency_hip_path_length_norm_body": path_length / body_size,
        "stability_hip_velocity_sd_norm_body": np.std(hip_v) / body_size,
    }

    # exploration (평균)
    exp_vals = []
    for pts in [lh_pts, rh_pts, lf_pts, rf_pts]:
        xs = fill_missing([p[0] for p in pts])
        ys = fill_missing([p[1] for p in pts])
        if nan_ratio(xs) < LIMB_MISSING_RATIO_MAX:
            d = limb_distance_series(list(zip(xs, ys)))
            exp_vals.append(np.mean(d) / body_size)

    feats["exploration_mean_norm_body"] = np.mean(exp_vals) if exp_vals else 0.0

    return feats

# ==========================================================
# 4. 한국어 피드백 자동 매핑
# ==========================================================
def generate_korean_feedback(feats):
    messages = []

    for f in selected_features:
        val = feats.get(f, 0.0)

        if "jerk" in f:
            if val > 0.05:
                messages.append("움직임이 다소 급하게 이어지고 있어요. 조금 더 천천히 이어가 보세요.")
            else:
                messages.append("움직임이 부드럽고 안정적으로 이어지고 있어요.")

        elif "velocity_sd" in f:
            if val > 0.08:
                messages.append("동작의 속도 변화가 큰 편이에요. 리듬을 일정하게 유지해 보세요.")
            else:
                messages.append("전반적으로 안정적인 움직임을 유지하고 있어요.")

        elif "path_length" in f:
            if val > 3.0:
                messages.append("이동 경로가 다소 길어요. 불필요한 움직임을 줄여보세요.")
            else:
                messages.append("효율적인 경로로 잘 이동하고 있어요.")

        elif "exploration" in f:
            if val > 0.6:
                messages.append("발과 손의 탐색 동작이 많은 편이에요. 다음 동작을 미리 계획해 보세요.")
            else:
                messages.append("탐색이 적고 동작 선택이 명확해 보여요.")

    return list(dict.fromkeys(messages))  # 중복 제거

# ==========================================================
# 5. Flask Server
# ==========================================================
app = Flask(__name__)
CORS(app)

@app.route("/predict", methods=["POST"])
def predict():

    if "video" not in request.files:
        return jsonify({"error": "No video uploaded"}), 400

    video = request.files["video"]
    temp_path = os.path.join(UPLOAD_DIR, video.filename)
    video.save(temp_path)

    try:
        feats = extract_features(temp_path)
        if feats is None:
            return jsonify({"error": "Feature extraction failed"}), 422

        # ML 입력
        X = pd.DataFrame([feats]).reindex(columns=selected_features).fillna(0.0)
        if scaler is not None:
            X[:] = scaler.transform(X)

        pred = int(model.predict(X)[0])
        prob = float(model.predict_proba(X)[0, 1])

        feedback_texts = generate_korean_feedback(feats)

        return jsonify({
            "prediction": {
                "label": "Advanced" if pred == 1 else "Intermediate",
                "probability": round(prob, 3)
            },
            "feedback_features": {
                f: round(float(feats.get(f, 0.0)), 3)
                for f in selected_features
            },
            "feedback_messages": feedback_texts
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500

    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)

# ==========================================================
# 6. Run
# ==========================================================
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5001)
