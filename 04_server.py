import os
import cv2
import joblib
import numpy as np
import pandas as pd
import mediapipe as mp
from flask import Flask, request, jsonify
from flask_cors import CORS

# ==========================================================
# 0. 설정 및 모델 로드
# ==========================================================
FRAME_INTERVAL = 3
UPLOAD_DIR = "./temp"
os.makedirs(UPLOAD_DIR, exist_ok=True)

mp_pose = mp.solutions.pose

# ML 모델 및 스케일러 로드
print("🔹 Loading ML artifacts...")
model = joblib.load("./result/best_model.pkl")
scaler = joblib.load("./result/best_scaler.pkl")
selected_features = joblib.load("./result/best_features.pkl")
print(f"✔ Loaded. Features: {len(selected_features)}")


# ==========================================================
# 1. 유틸리티 함수 (제공해주신 스크립트와 동일)
# ==========================================================
def fill_missing(arr):
    s = pd.Series(arr, dtype=float)
    s = s.interpolate(limit_direction="both").ffill().bfill()
    return s.to_numpy()


def center_point(p1, p2):
    return ((p1[0] + p2[0]) / 2.0, (p1[1] + p2[1]) / 2.0)


def velocity_series(pts, dt):
    v = [0.0]
    for i in range(1, len(pts)):
        dx = pts[i][0] - pts[i - 1][0]
        dy = pts[i][1] - pts[i - 1][1]
        v.append(np.sqrt(dx ** 2 + dy ** 2) / dt)
    return np.array(v)


def acc_series(v, dt):
    return np.gradient(v, dt)


def jerk_series(a, dt):
    return np.gradient(a, dt)


def body_size_from_landmarks(lm):
    def dist(i, j):
        return np.sqrt((lm[i].x - lm[j].x) ** 2 + (lm[i].y - lm[j].y) ** 2)

    pairs = [(11, 12), (23, 24), (11, 23), (12, 24)]
    return np.mean([dist(i, j) for i, j in pairs])


def limb_distance_series(pts):
    d = [0.0]
    for i in range(1, len(pts)):
        dx = pts[i][0] - pts[i - 1][0]
        dy = pts[i][1] - pts[i - 1][1]
        d.append(np.sqrt(dx ** 2 + dy ** 2))
    return np.array(d)


# ==========================================================
# 2. 특징 추출 (27개 변수 로직 반영)
# ==========================================================
def extract_features(video_path):
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    fps = 30.0 if fps <= 0 else fps
    dt = 1.0 / fps

    hip_pts, lh_pts, rh_pts, lf_pts, rf_pts = [], [], [], [], []
    body_sizes = []
    frame_idx = 0

    with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
        while True:
            ret, frame = cap.read()
            if not ret: break
            if frame_idx % FRAME_INTERVAL != 0:
                frame_idx += 1
                continue

            h, w = frame.shape[:2]
            res = pose.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

            if res.pose_landmarks:
                lm = res.pose_landmarks.landmark
                body_sizes.append(body_size_from_landmarks(lm))
                L_HIP, R_HIP = (lm[23].x * w, lm[23].y * h), (lm[24].x * w, lm[24].y * h)
                hip_pts.append(center_point(L_HIP, R_HIP))
                lh_pts.append((lm[15].x * w, lm[15].y * h))
                rh_pts.append((lm[16].x * w, lm[16].y * h))
                lf_pts.append((lm[27].x * w, lm[27].y * h))
                rf_pts.append((lm[28].x * w, lm[28].y * h))
            else:
                for lst in [hip_pts, lh_pts, rh_pts, lf_pts, rf_pts]:
                    lst.append((np.nan, np.nan))
            frame_idx += 1
    cap.release()

    body_size = np.mean(body_sizes) if body_sizes else 1.0
    dt_eff = dt * FRAME_INTERVAL

    hip_x = fill_missing([p[0] for p in hip_pts])
    hip_y = fill_missing([p[1] for p in hip_pts])
    hip_xy = list(zip(hip_x, hip_y))

    # Fluency & Stability (Hip)
    v = velocity_series(hip_xy, dt_eff)
    a = acc_series(v, dt_eff)
    j = jerk_series(a, dt_eff)
    path = np.sum(v * dt_eff)
    total_time = len(hip_xy) * dt_eff

    feats = {
        "total_time": total_time,
        "fluency_hip_velocity_mean_norm": np.mean(v) / body_size,
        "fluency_hip_velocity_max_norm": np.max(v) / body_size,
        "fluency_hip_acc_mean_norm": np.mean(np.abs(a)) / body_size,
        "fluency_hip_acc_max_norm": np.max(np.abs(a)) / body_size,
        "fluency_hip_jerk_mean_norm": np.mean(np.abs(j)) / body_size,
        "fluency_hip_jerk_max_norm": np.max(np.abs(j)) / body_size,
        "fluency_hip_jerk_rms_norm": np.sqrt(np.mean(j ** 2)) / body_size,
        "fluency_hip_path_length_norm": path / body_size,
        "fluency_hip_path_per_sec_norm": path / total_time / body_size,
        "stability_hip_velocity_sd_norm": np.std(v) / body_size,
        "stability_hip_acc_sd_norm": np.std(a) / body_size,
        "stability_hip_jerk_sd_norm": np.std(j) / body_size,
    }

    # Limbs (Stability & Exploration)
    for name, pts in {"left_hand": lh_pts, "right_hand": rh_pts, "left_foot": lf_pts, "right_foot": rf_pts}.items():
        xs = fill_missing([p[0] for p in pts])
        ys = fill_missing([p[1] for p in pts])
        pts_clean = list(zip(xs, ys))
        lv = velocity_series(pts_clean, dt_eff)
        ld = limb_distance_series(pts_clean)
        feats[f"stability_{name}_velocity_sd_norm"] = np.std(lv) / body_size
        feats[f"exploration_{name}_velocity_mean_norm"] = np.mean(lv) / body_size
        feats[f"exploration_{name}_path_length_norm"] = np.sum(ld) / body_size

    return feats


# ==========================================================
# 3. 한국어 피드백 생성
# ==========================================================
def generate_korean_feedback(feats):
    msg = []
    if feats.get("fluency_hip_jerk_mean_norm", 0) > 0.05:
        msg.append("움직임이 다소 급합니다. 무게 중심을 더 천천히 이동시켜 보세요.")
    else:
        msg.append("중심 이동이 매우 부드럽고 안정적입니다.")

    if feats.get("stability_hip_velocity_sd_norm", 0) > 0.08:
        msg.append("일정한 속도를 유지하기보다 끊기는 동작이 보입니다. 리듬감을 높여보세요.")

    if feats.get("exploration_left_hand_path_length_norm", 0) + feats.get("exploration_right_hand_path_length_norm",
                                                                          0) > 1.5:
        msg.append("홀드를 잡기 전 손의 탐색 동작이 많습니다. 다음 홀드를 명확히 정하고 움직여보세요.")

    return msg if msg else ["전반적으로 안정적인 등반입니다."]


# ==========================================================
# 4. Flask 서버 설정
# ==========================================================
app = Flask(__name__)
CORS(app)


@app.route("/predict", methods=["POST"])
def predict():
    temp_path = None
    try:
        if "video" not in request.files:
            return jsonify({"error": "No video"}), 400

        video = request.files["video"]
        temp_path = os.path.join(UPLOAD_DIR, video.filename)
        video.save(temp_path)

        # 1. 특징 추출
        feats = extract_features(temp_path)

        # 2. ML 예측 전용 데이터셋 구성 (Selected features만 추출)
        X = pd.DataFrame([feats]).reindex(columns=selected_features).fillna(0)
        if scaler:
            X[:] = scaler.transform(X)

        pred = int(model.predict(X)[0])
        prob = float(model.predict_proba(X)[0, 1])

        # 3. 응답 데이터 구성 (플러터 앱 형식)
        return jsonify({
            "prediction": {
                "label": "Advanced" if pred == 0 else "Intermediate",  # 0: Advanced, 1: Intermediate 기준
                "probability": round(prob, 3)
            },
            "feedback_features": {k: round(float(v), 4) for k, v in feats.items()},
            "feedback_messages": generate_korean_feedback(feats)
        })

    except Exception as e:
        print(f"🔥 Error: {e}")
        return jsonify({"error": str(e)}), 500
    finally:
        if temp_path and os.path.exists(temp_path):
            os.remove(temp_path)


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5001)