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

    # Label 추출 (언더바 사이의 숫자)
    m = re.search(r'_(\d)_', stem)
    label = int(m.group(1)) if m else None

    # ID 생성: 앞 숫자 + "_" + label
    parts = stem.split('_')
    if len(parts) >= 1 and label is not None:
        # 첫 번째 숫자 부분 + label
        video_id = f"{parts[0]}_{label}"
    else:
        # 파싱 실패 시 전체 이름 사용
        video_id = stem

    return video_id, label


# ==========================================================
# 2. Missing 값 처리
# ==========================================================
def fill_missing(arr):
    """보간 후 forward/backward fill로 완전히 채우기"""
    s = pd.Series(arr, dtype=float)
    s = s.interpolate(limit_direction="both")
    s = s.ffill().bfill()
    return s.to_numpy()

def nan_ratio(arr):
    """배열에서 NaN 비율 계산"""
    arr = np.asarray(arr, dtype=float)
    return np.mean(np.isnan(arr))


# ==========================================================
# 3. Kinematics 계산 함수
# ==========================================================
def center_point(p1, p2):
    """두 점의 중심점 계산"""
    return ((p1[0] + p2[0]) / 2.0, (p1[1] + p2[1]) / 2.0)


def velocity_series(pts, dt):
    """속도 시계열 계산 (첫 프레임 속도는 0)"""
    v = [0.0]
    for i in range(1, len(pts)):
        dx = pts[i][0] - pts[i-1][0]
        dy = pts[i][1] - pts[i-1][1]
        v.append(np.sqrt(dx**2 + dy**2) / dt)
    return np.array(v)

def acc_series(v, dt):
    return np.gradient(v, dt)

def jerk_series(a, dt):
    return np.gradient(a, dt)

def body_size_from_landmarks(lm):
    def dist(i, j):
        return np.sqrt((lm[i].x - lm[j].x) ** 2 + (lm[i].y - lm[j].y) ** 2)

    # 어깨, 엉덩이의 4개 거리 평균
    pairs = [(11, 12), (23, 24), (11, 23), (12, 24)]
    return np.mean([dist(i, j) for i, j in pairs])

def limb_distance_series(pts):
    """프레임 간 사지의 이동 거리 시계열"""
    d = [0.0]
    for i in range(1, len(pts)):
        dx = pts[i][0] - pts[i-1][0]
        dy = pts[i][1] - pts[i-1][1]
        d.append(np.sqrt(dx**2 + dy**2))
    return np.array(d)


# ==========================================================
# 4. Feature Extraction (27개 변수)
# ==========================================================
def extract_features(video_path):
    """
    비디오에서 운동학적 특징 추출
    Returns: 27개 특징 변수 또는 None (실패 시)
    """
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    fps = 30.0 if fps <= 0 else fps
    dt = 1.0 / fps

    hip_pts = []
    lh_pts, rh_pts = [], []
    lf_pts, rf_pts = [], []
    body_sizes = []  # 초반 5초 동안 수집
    body_size = None
    max_frames_for_size = int(fps * 5 / FRAME_INTERVAL)  # 초반 5초
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

                # Body size는 초반 5초 동안 수집
                if len(body_sizes) < max_frames_for_size:
                    body_sizes.append(body_size_from_landmarks(lm))

                # 5초 수집 완료 시 평균 계산
                if body_size is None and len(body_sizes) >= max_frames_for_size:
                    body_size = np.mean(body_sizes)

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

    # 데이터 유효성 검증
    if len(hip_pts) < 2:
        print(f"  ⚠️ 프레임 수 부족: {len(hip_pts)}개")
        return None

    # Body size 확인
    if body_size is None:
        # 5초가 안 되었지만 수집된 데이터가 있으면 사용
        if len(body_sizes) > 0:
            body_size = np.mean(body_sizes)
            print(f"  ℹ️ Body size를 {len(body_sizes)}개 프레임으로 계산")
        else:
            print(f"  ⚠️ Body size 계산 불가 (pose detection 실패)")
            return None

    # Hip missing ratio 체크
    hip_x = np.array([p[0] for p in hip_pts])
    hip_y = np.array([p[1] for p in hip_pts])
    if nan_ratio(hip_x) > HIP_MISSING_RATIO_MAX or nan_ratio(hip_y) > HIP_MISSING_RATIO_MAX:
        print(f"  ⚠️ Hip missing ratio 초과: {nan_ratio(hip_x):.2%}, {nan_ratio(hip_y):.2%}")
        return None

    # Hip 보간 (이후 NaN 없음 보장)
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

    for k, pts in limb_dict.items():
        xs = np.array([p[0] for p in pts])
        ys = np.array([p[1] for p in pts])

        # Missing ratio가 높으면 해당 limb는 None 처리
        if nan_ratio(xs) > LIMB_MISSING_RATIO_MAX or nan_ratio(ys) > LIMB_MISSING_RATIO_MAX:
            limb_dict[k] = None
        else:
            # 보간하여 사용
            limb_dict[k] = list(zip(fill_missing(xs), fill_missing(ys)))

    dt_eff = dt * FRAME_INTERVAL
    total_time = len(hip_xy) * dt_eff

    # ============================
    # Fluency (9개)
    # ============================
    hip_v = velocity_series(hip_xy, dt_eff)
    hip_a = acc_series(hip_v, dt_eff)
    hip_j = jerk_series(hip_a, dt_eff)
    path = np.sum(hip_v * dt_eff)

    fluency = {
        "fluency_hip_velocity_mean_norm": np.mean(hip_v) / body_size,
        "fluency_hip_velocity_max_norm": np.max(hip_v) / body_size,
        "fluency_hip_acc_mean_norm": np.mean(np.abs(hip_a)) / body_size,
        "fluency_hip_acc_max_norm": np.max(np.abs(hip_a)) / body_size,
        "fluency_hip_jerk_mean_norm": np.mean(np.abs(hip_j)) / body_size,
        "fluency_hip_jerk_max_norm": np.max(np.abs(hip_j)) / body_size,
        "fluency_hip_jerk_rms_norm": np.sqrt(np.mean(hip_j ** 2)) / body_size,
        "fluency_hip_path_length_norm": path / body_size,
        "fluency_hip_path_per_sec_norm": path / total_time / body_size,
    }

    # ============================
    # Stability (7개)
    # ============================
    stability = {
        "stability_hip_velocity_sd_norm": np.std(hip_v) / body_size,
        "stability_hip_acc_sd_norm": np.std(hip_a) / body_size,
        "stability_hip_jerk_sd_norm": np.std(hip_j) / body_size,
    }

    for limb in limb_dict:
        pts = limb_dict[limb]
        if pts is None:
            # 인식 안된 관절은 NaN 처리
            stability[f"stability_{limb}_velocity_sd_norm"] = np.nan
        else:
            v = velocity_series(pts, dt_eff)
            stability[f"stability_{limb}_velocity_sd_norm"] = np.std(v) / body_size

    # ============================
    # Exploration (8개 - 각 limb당 2개씩)
    # ============================
    exploration = {}
    for limb in limb_dict:
        pts = limb_dict[limb]
        if pts is None:
            # 인식 안된 관절은 NaN 처리
            exploration[f"exploration_{limb}_velocity_mean_norm"] = np.nan
            exploration[f"exploration_{limb}_path_length_norm"] = np.nan
        else:
            v = velocity_series(pts, dt_eff)
            d = limb_distance_series(pts)
            # 평균 속도 (활동성)
            exploration[f"exploration_{limb}_velocity_mean_norm"] = np.mean(v) / body_size
            # 총 이동 거리 (탐색 범위)
            exploration[f"exploration_{limb}_path_length_norm"] = np.sum(d) / body_size

    # 최종 특징 딕셔너리 구성
    video_id, video_label = extract_id_and_label(video_path)

    feats = {
        "id": video_id,
        "label": video_label,
        "total_time": total_time
    }

    feats.update(fluency)
    feats.update(stability)
    feats.update(exploration)

    return feats


# ==========================================================
# 5. MAIN
# ==========================================================
def main():
    files = glob.glob(os.path.join(VIDEO_DIR, "*.mp4")) + \
            glob.glob(os.path.join(VIDEO_DIR, "*.mov"))

    if not files:
        print("❌ 분석할 비디오가 없습니다.")
        return

    print(f"📁 총 {len(files)}개 비디오 분석 시작\n")

    success_count = 0
    fail_count = 0

    for idx, video_path in enumerate(files, 1):
        base = os.path.splitext(os.path.basename(video_path))[0]
        out_path = os.path.join(OUTPUT_DIR, f"{base}.xlsx")

        print(f"[{idx}/{len(files)}] {os.path.basename(video_path)}")
        feats = extract_features(video_path)

        if feats is None:
            print(f"  ❌ Feature 추출 실패 → {video_path}\n")
            fail_count += 1
            continue

        df = pd.DataFrame([feats])
        df.to_excel(out_path, index=False)
        print(f"  ✅ 저장 완료: {base}.xlsx\n")
        success_count += 1

    print("=" * 60)
    print(f"🎉 분석 완료!")
    print(f"   성공: {success_count}개 | 실패: {fail_count}개")
    print("=" * 60)


if __name__ == "__main__":
    main()