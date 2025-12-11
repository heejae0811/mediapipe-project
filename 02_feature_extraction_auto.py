import re
import os
import glob
import cv2
import numpy as np
import pandas as pd
import mediapipe as mp

# ==========================================================
# 0. 기본 설정
# ==========================================================
VIDEO_DIR = "./videos/"
OUTPUT_DIR = "./features_xlsx/"
os.makedirs(OUTPUT_DIR, exist_ok=True)
FRAME_INTERVAL = 3

HIP_MISSING_RATIO_MAX = 0.30
LIMB_MISSING_RATIO_MAX = 0.40
mp_pose = mp.solutions.pose


# ==========================================================
# 1. id, label
# ==========================================================
def extract_id_and_label(video_path):
    fname = os.path.basename(video_path)
    stem = os.path.splitext(fname)[0]
    m = re.search(r'_(\d)_', stem)
    label = int(m.group(1)) if m else None
    return stem, label


# ==========================================================
# 2. Missing 처리
# ==========================================================
def fill_missing(arr):
    s = pd.Series(arr, dtype=float)
    s = s.interpolate(limit_direction="both")
    return s.to_numpy()

def nan_ratio(arr):
    arr = np.asarray(arr, dtype=float)
    return np.mean(np.isnan(arr))


# ==========================================================
# 3. Kinematics utils
# ==========================================================
def center_point(p1, p2):
    return ((p1[0] + p2[0]) / 2.0, (p1[1] + p2[1]) / 2.0)

def velocity_series(pts, dt):
    v = [0.0]
    for t in range(1, len(pts)):
        dx = pts[t][0] - pts[t-1][0]
        dy = pts[t][1] - pts[t-1][1]
        v.append(np.sqrt(dx**2 + dy**2) / dt)
    return np.array(v, float)

def acc_series(v, dt):
    a = [0.0]
    for t in range(1, len(v)):
        a.append((v[t] - v[t-1]) / dt)
    return np.array(a, float)

def jerk_series(a, dt):
    j = [0.0]
    for t in range(1, len(a)):
        j.append((a[t] - a[t-1]) / dt)
    return np.array(j, float)

def body_size_from_landmarks(lm):
    def dist(i, j):
        return np.sqrt((lm[i].x - lm[j].x)**2 + (lm[i].y - lm[j].y)**2)
    pairs = [(11, 12), (23, 24), (11, 23), (12, 24)]
    vals = [dist(i, j) for i, j in pairs]
    return np.mean(vals) if vals else 1.0


# ==========================================================
# 4. Limb exploration distance
# ==========================================================
def limb_distance_series(pts):
    d = [0.0]
    for t in range(1, len(pts)):
        dx = pts[t][0] - pts[t-1][0]
        dy = pts[t][1] - pts[t-1][1]
        d.append(np.sqrt(dx**2 + dy**2))
    return np.array(d, float)


# ==========================================================
# 5. Feature Extraction (최종 43개)
# ==========================================================
def extract_features(video_path):

    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    fps = 30.0 if fps <= 0 else fps
    dt = 1.0 / fps

    hip_pts = []
    lh_pts, rh_pts = [], []
    lf_pts, rf_pts = [], []
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

    # HIP missing 처리
    hip_x = np.array([p[0] for p in hip_pts], float)
    hip_y = np.array([p[1] for p in hip_pts], float)

    if nan_ratio(hip_x) > HIP_MISSING_RATIO_MAX or nan_ratio(hip_y) > HIP_MISSING_RATIO_MAX:
        return None

    hip_x = fill_missing(hip_x)
    hip_y = fill_missing(hip_y)
    hip_xy = list(zip(hip_x, hip_y))

    # Limb missing 처리
    limb_dict = {
        "left_hand": lh_pts,
        "right_hand": rh_pts,
        "left_foot": lf_pts,
        "right_foot": rf_pts
    }

    for limb, pts in limb_dict.items():
        xs = np.array([p[0] for p in pts], float)
        ys = np.array([p[1] for p in pts], float)

        if nan_ratio(xs) > LIMB_MISSING_RATIO_MAX or nan_ratio(ys) > LIMB_MISSING_RATIO_MAX:
            limb_dict[limb] = None
        else:
            xs = fill_missing(xs)
            ys = fill_missing(ys)
            limb_dict[limb] = list(zip(xs, ys))

    dt_eff = dt * FRAME_INTERVAL
    body_size = np.mean(body_sizes)
    total_time = len(hip_xy) * dt_eff

    # ============================
    # Fluency (18개)
    # ============================
    hip_v = velocity_series(hip_xy, dt_eff)
    hip_a = acc_series(hip_v, dt_eff)
    hip_j = jerk_series(hip_a, dt_eff)

    path_length = np.sum(hip_v * dt_eff)

    fluency = {
        "fluency_hip_velocity_mean": float(np.mean(hip_v)),
        "fluency_hip_velocity_max": float(np.max(hip_v)),
        "fluency_hip_acc_mean": float(np.mean(hip_a)),
        "fluency_hip_acc_max": float(np.max(hip_a)),
        "fluency_hip_jerk_mean": float(np.mean(hip_j)),
        "fluency_hip_jerk_max": float(np.max(hip_j)),
        "fluency_hip_jerk_rms": float(np.sqrt(np.mean(hip_j**2))),
        "fluency_hip_path_length": float(path_length),
        "fluency_hip_path_length_per_sec": float(path_length / total_time),

        "fluency_hip_velocity_mean_norm_body": float(np.mean(hip_v) / body_size),
        "fluency_hip_velocity_max_norm_body": float(np.max(hip_v) / body_size),
        "fluency_hip_acc_mean_norm_body": float(np.mean(hip_a) / body_size),
        "fluency_hip_acc_max_norm_body": float(np.max(hip_a) / body_size),
        "fluency_hip_jerk_mean_norm_body": float(np.mean(hip_j) / body_size),
        "fluency_hip_jerk_max_norm_body": float(np.max(hip_j) / body_size),
        "fluency_hip_jerk_rms_norm_body": float(np.sqrt(np.mean(hip_j**2)) / body_size),
        "fluency_hip_path_length_norm_body": float(path_length / body_size),
        "fluency_hip_path_length_per_sec_norm_body": float(path_length / total_time / body_size),
    }

    # ============================
    # Exploration (8개)
    # ============================
    limb_exploration = {}
    for limb in ["left_hand", "right_hand", "left_foot", "right_foot"]:
        pts = limb_dict[limb]
        if pts is None:
            limb_exploration[f"exploration_{limb}_distance_mean"] = np.nan
            limb_exploration[f"exploration_{limb}_distance_mean_norm_body"] = np.nan
            continue

        d = limb_distance_series(pts)

        limb_exploration[f"exploration_{limb}_distance_mean"] = float(np.mean(d))
        limb_exploration[f"exploration_{limb}_distance_mean_norm_body"] = float(np.mean(d) / body_size)

    # ============================
    # Stability (14개)
    # ============================
    stability = {
        "stability_hip_velocity_sd": float(np.std(hip_v)),
        "stability_hip_acc_sd": float(np.std(hip_a)),
        "stability_hip_jerk_sd": float(np.std(hip_j)),
        "stability_hip_velocity_sd_norm_body": float(np.std(hip_v) / body_size),
        "stability_hip_acc_sd_norm_body": float(np.std(hip_a) / body_size),
        "stability_hip_jerk_sd_norm_body": float(np.std(hip_j) / body_size),
    }

    for limb in ["left_hand", "right_hand", "left_foot", "right_foot"]:
        pts = limb_dict[limb]

        if pts is None:
            stability[f"stability_{limb}_velocity_sd"] = np.nan
            stability[f"stability_{limb}_velocity_sd_norm_body"] = np.nan
            continue

        v_l = velocity_series(pts, dt_eff)

        stability[f"stability_{limb}_velocity_sd"] = float(np.std(v_l))
        stability[f"stability_{limb}_velocity_sd_norm_body"] = float(np.std(v_l) / body_size)

    # ============================
    # 통합: 모든 feature + id/label/total_time
    # ============================
    feats = {
        "id": os.path.splitext(os.path.basename(video_path))[0].split("_")[0],
        "label": extract_id_and_label(video_path)[1],
        "total_time": float(total_time)
    }

    feats.update(fluency)
    feats.update(limb_exploration)
    feats.update(stability)

    return feats


# ==========================================================
# 6. MAIN
# ==========================================================
def main():
    files = glob.glob(os.path.join(VIDEO_DIR, "*.mp4")) + \
            glob.glob(os.path.join(VIDEO_DIR, "*.mov"))

    if not files:
        print("❌ 분석할 비디오가 없습니다.")
        return

    print(f"📁 총 {len(files)}개 비디오 분석 시작")

    for video_path in files:
        base = os.path.splitext(os.path.basename(video_path))[0]
        out_path = os.path.join(OUTPUT_DIR, f"{base}.xlsx")

        print(f"▶ 분석 중: {video_path}")
        feats = extract_features(video_path)

        if feats is None:
            print(f"⚠️ Feature 추출 실패 → {video_path}")
            continue

        df = pd.DataFrame([feats])
        df.to_excel(out_path, index=False)
        print(f"✅ 저장 완료:", out_path)

    print("🎉 모든 비디오 분석 완료!")


if __name__ == "__main__":
    main()
