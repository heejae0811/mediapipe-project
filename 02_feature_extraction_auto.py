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
VIDEO_DIR = "./videos/"           # 분석할 비디오 폴더
OUTPUT_DIR = "./features_xlsx/"   # 결과 저장 폴더
os.makedirs(OUTPUT_DIR, exist_ok=True)
FRAME_INTERVAL = 3


# ----------------------------------------------------------
# 결측치 허용 비율
#   - HIP: 30% 초과 → 영상 전체를 신뢰하기 어려움 → None 반환
#   - Limb: 40% 초과 → 해당 limb feature만 NaN 처리
# ----------------------------------------------------------
HIP_MISSING_RATIO_MAX = 0.30
LIMB_MISSING_RATIO_MAX = 0.40
mp_pose = mp.solutions.pose


# ==========================================================
# 1. id, label 설정
# ==========================================================
def extract_id_and_label(video_path):
    """
    파일명에서 id와 label 추출.
    label = _0_ 또는 _1_ 중에 들어 있는 숫자
    id = 확장자를 제외한 전체 파일명
    """
    fname = os.path.basename(video_path)
    stem = os.path.splitext(fname)[0]

    # label 추출 (_숫자_ 형식)
    m = re.search(r'_(\d)_', stem)
    label = int(m.group(1)) if m else None

    return stem, label


# ==========================================================
# 2. 결측치 처리
# ==========================================================
def fill_missing(arr):
    """
    1차원 배열에서 결측치(NaN)를 선형 보간 (linear interpolation)으로 채움.
    - 수학적으로: 결측 구간을 앞/뒤 값으로 직선 보간 x(t) ~ linear interpolation between known samples
    - limit_direction='both' → 배열 양 끝단의 NaN도 가장 가까운 유효값으로 채움.
    """
    s = pd.Series(arr, dtype="float")
    s = s.interpolate(limit_direction="both")
    return s.to_numpy()


def nan_ratio(arr):
    """
    배열에서 NaN 비율 계산
    r = (#NaN) / (전체 길이)
    """
    arr = np.asarray(arr, dtype=float)
    return np.mean(np.isnan(arr))


# ==========================================================
# 3. Kinematics 유틸 (위치 → 속도 → 가속도 → 저크)
# ==========================================================
def center_point(p1, p2):
    """두 점 p1, p2의 중앙점: ( (x1+x2)/2 , (y1+y2)/2 )"""
    return ((p1[0] + p2[0]) / 2.0, (p1[1] + p2[1]) / 2.0)


def velocity_series(pts, dt):
    """
    위치 시계열 pts = [(x_t, y_t)]로부터 속도 시계열 v_t 계산.
    수식: v_t = sqrt( (x_t - x_{t-1})^2 + (y_t - y_{t-1})^2 ) / dt
    """
    v = [0.0]
    for t in range(1, len(pts)):
        dx = pts[t][0] - pts[t - 1][0]
        dy = pts[t][1] - pts[t - 1][1]
        v.append(np.sqrt(dx**2 + dy**2) / dt)
    return np.array(v, dtype=float)


def acc_series(v, dt):
    """
    속도 시계열 v_t 로부터 가속도 a_t 계산.
    수식: a_t = (v_t - v_{t-1}) / dt
    """
    a = [0.0]
    for t in range(1, len(v)):
        a.append((v[t] - v[t - 1]) / dt)
    return np.array(a, dtype=float)


def jerk_series(a, dt):
    """
    가속도 시계열 a_t 로부터 jerk j_t 계산.
    수식: j_t = (a_t - a_{t-1}) / dt
    """
    j = [0.0]
    for t in range(1, len(a)):
        j.append((a[t] - a[t - 1]) / dt)
    return np.array(j, dtype=float)


def body_size_from_landmarks(lm):
    """
    body_size: 어깨-어깨, 골반-골반, 어깨-골반 길이의 평균.
    - 카메라 거리/줌 차이를 보정하는 스케일링 기준.
    - 수식:
        d_ij = sqrt( (x_i - x_j)^2 + (y_i - y_j)^2 )
        body_size = mean( d_11-12, d_23-24, d_11-23, d_12-24 )
    """
    def dist(i, j):
        return np.sqrt((lm[i].x - lm[j].x)**2 + (lm[i].y - lm[j].y)**2)

    pairs = [(11, 12), (23, 24), (11, 23), (12, 24)]
    vals = [dist(i, j) for i, j in pairs]
    vals = [v for v in vals if not np.isnan(v)]
    return np.mean(vals) if len(vals) > 0 else 1.0


# ==========================================================
# 4. Exploration (거리 기반 탐색 지표)
# ==========================================================
def limb_distance_series(pts):
    """
    limb 위치 시계열 pts에서 프레임 간 이동 거리 d_t 계산.
    수식: d_t = sqrt( (x_t - x_{t-1})^2 + (y_t - y_{t-1})^2 )
    """
    d = [0.0]
    for t in range(1, len(pts)):
        dx = pts[t][0] - pts[t - 1][0]
        dy = pts[t][1] - pts[t - 1][1]
        d.append(np.sqrt(dx**2 + dy**2))
    return np.array(d, dtype=float)


# ==========================================================
# 5. 비디오에서 Feature 추출
# ==========================================================
def extract_features(video_path):
    """
    하나의 비디오에서 54개 퍼포먼스 지표를 계산하여 dict로 반환.
    - Fluency: 12
    - Exploration: 8
    - Stability: 30
    - Speed: 2
    """

    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        fps = 30.0
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

    # HIP 결측치 처리
    hip_x = np.array([p[0] for p in hip_pts], float)
    hip_y = np.array([p[1] for p in hip_pts], float)

    if nan_ratio(hip_x) > HIP_MISSING_RATIO_MAX or nan_ratio(hip_y) > HIP_MISSING_RATIO_MAX:
        return None

    hip_x = fill_missing(hip_x)
    hip_y = fill_missing(hip_y)
    hip_xy = list(zip(hip_x, hip_y))

    # Limb 결측치 처리
    limb_dict = {
        "left_hand": lh_pts,
        "right_hand": rh_pts,
        "left_foot": lf_pts,
        "right_foot": rf_pts
    }

    for name, pts in limb_dict.items():
        xs = np.array([p[0] for p in pts], float)
        ys = np.array([p[1] for p in pts], float)

        if nan_ratio(xs) > LIMB_MISSING_RATIO_MAX or nan_ratio(ys) > LIMB_MISSING_RATIO_MAX:
            limb_dict[name] = None
        else:
            xs = fill_missing(xs)
            ys = fill_missing(ys)
            limb_dict[name] = list(zip(xs, ys))

    # 시간, body size 계산
    dt_eff = dt * FRAME_INTERVAL
    body_size = np.mean(body_sizes)
    total_time = len(hip_xy) * dt_eff

    # =====================================================================
    # 1) FLUENCY (HIP)
    # =====================================================================
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
        "fluency_hip_path_length": float(path_length),

        "fluency_hip_velocity_mean_norm_body": float(np.mean(hip_v) / body_size),
        "fluency_hip_acc_mean_norm_body": float(np.mean(hip_a) / body_size),
        "fluency_hip_jerk_mean_norm_body": float(np.mean(hip_j) / body_size),
        "fluency_hip_path_length_norm_body": float(path_length / body_size),

        "fluency_hip_path_length_per_sec": float(path_length / total_time),
    }

    # =====================================================================
    # 2) EXPLORATION (Limb)
    # =====================================================================
    limb_exploration = {}
    for limb_name in ["left_hand", "right_hand", "left_foot", "right_foot"]:
        pts = limb_dict[limb_name]
        if pts is None:
            limb_exploration[f"exploration_{limb_name}_distance_mean"] = np.nan
            limb_exploration[f"exploration_{limb_name}_distance_mean_norm_body"] = np.nan
            continue

        d = limb_distance_series(pts)

        limb_exploration[f"exploration_{limb_name}_distance_mean"] = float(np.mean(d))
        limb_exploration[f"exploration_{limb_name}_distance_mean_norm_body"] = float(np.mean(d) / body_size)

    # =====================================================================
    # 3) STABILITY (HIP + Limbs)
    # =====================================================================
    stability = {
        "stability_hip_velocity_sd": float(np.std(hip_v)),
        "stability_hip_acc_sd": float(np.std(hip_a)),
        "stability_hip_jerk_sd": float(np.std(hip_j)),

        "stability_hip_velocity_sd_norm_body": float(np.std(hip_v) / body_size),
        "stability_hip_acc_sd_norm_body": float(np.std(hip_a) / body_size),
        "stability_hip_jerk_sd_norm_body": float(np.std(hip_j) / body_size),
    }

    for limb_name in ["left_hand", "right_hand", "left_foot", "right_foot"]:
        pts = limb_dict[limb_name]

        if pts is None:
            stability[f"stability_{limb_name}_velocity_sd"] = np.nan
            stability[f"stability_{limb_name}_acc_sd"] = np.nan
            stability[f"stability_{limb_name}_jerk_sd"] = np.nan

            stability[f"stability_{limb_name}_velocity_sd_norm_body"] = np.nan
            stability[f"stability_{limb_name}_acc_sd_norm_body"] = np.nan
            stability[f"stability_{limb_name}_jerk_sd_norm_body"] = np.nan
            continue

        v_l = velocity_series(pts, dt_eff)
        a_l = acc_series(v_l, dt_eff)
        j_l = jerk_series(a_l, dt_eff)

        stability[f"stability_{limb_name}_velocity_sd"] = float(np.std(v_l))
        stability[f"stability_{limb_name}_acc_sd"] = float(np.std(a_l))
        stability[f"stability_{limb_name}_jerk_sd"] = float(np.std(j_l))

        stability[f"stability_{limb_name}_velocity_sd_norm_body"] = float(np.std(v_l) / body_size)
        stability[f"stability_{limb_name}_acc_sd_norm_body"] = float(np.std(a_l) / body_size)
        stability[f"stability_{limb_name}_jerk_sd_norm_body"] = float(np.std(j_l) / body_size)

    # =====================================================================
    # 4) SPEED (HIP)
    # =====================================================================
    ascent = (hip_y[0] - hip_y[-1]) / total_time

    speed = {
        "speed_hip_ascent_speed": float(ascent),
        "speed_hip_ascent_speed_norm_body": float(ascent / body_size),
    }

    # =====================================================================
    # 5) 통합
    # =====================================================================
    feats = {}
    feats["id"] = os.path.splitext(os.path.basename(video_path))[0].split("_")[0]
    feats["label"] = extract_id_and_label(video_path)[1]

    feats.update(fluency)
    feats.update(limb_exploration)
    feats.update(stability)
    feats.update(speed)

    return feats


# ==========================================================
# 6. 메인: 폴더 내 모든 영상 분석 + XLSX 저장
# ==========================================================
def main():
    """
    VIDEO_DIR 내의 .mp4, .mov 파일을 모두 분석하고,
    각 비디오마다 1개의 .xlsx 파일로 feature를 저장.
    """
    files = glob.glob(os.path.join(VIDEO_DIR, "*.mp4")) \
          + glob.glob(os.path.join(VIDEO_DIR, "*.mov"))

    if len(files) == 0:
        print("❌ 분석할 비디오가 없습니다.")
        return

    print(f"📁 총 {len(files)}개 비디오 분석 시작")

    for video_path in files:
        base = os.path.splitext(os.path.basename(video_path))[0]
        out_path = os.path.join(OUTPUT_DIR, f"{base}.xlsx")

        print(f"▶ 분석 중: {video_path}")
        feats = extract_features(video_path)

        if feats is None:
            print(f"⚠️ Feature 추출 실패 → {video_path} (결측 과다 또는 유효 프레임 부족)")
            continue

        df = pd.DataFrame([feats])
        df.to_excel(out_path, index=False)
        print(f"✅ 저장 완료: {out_path}")

    print("🎉 모든 비디오 분석 완료!")


if __name__ == "__main__":
    main()
