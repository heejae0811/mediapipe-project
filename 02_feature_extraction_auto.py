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
VIDEO_DIR = "./v/"
OUTPUT_DIR = "./features_xlsx/"
os.makedirs(OUTPUT_DIR, exist_ok=True)

FRAME_INTERVAL = 1
HIP_MISSING_RATIO_MAX = 0.30
BODYSIZE_MISSING_RATIO_MAX = 0.40

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
# 2. Missing 처리
# ==========================================================
def fill_missing(arr):
    """보간 후 forward/backward fill로 완전히 채우기"""
    s = pd.Series(arr, dtype=float)
    s = s.interpolate(limit_direction="both").ffill().bfill()
    return s.to_numpy()

def nan_ratio(arr):
    return float(np.mean(np.isnan(np.asarray(arr, dtype=float))))


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

def robust_body_size_from_landmarks(lm):
    def dist(i, j):
        dx = lm[i].x - lm[j].x
        dy = lm[i].y - lm[j].y
        return np.sqrt(dx*dx + dy*dy)
    pairs = [(11,12), (23,24), (11,23), (12,24)]
    bs = float(np.mean([dist(i,j) for i,j in pairs]))
    return np.nan if (not np.isfinite(bs) or bs <= 1e-6) else bs

def linear_slope(y, dt):
    y = np.asarray(y, dtype=float)
    if len(y) < 3:
        return np.nan
    t = np.arange(len(y)) * dt
    mask = np.isfinite(y)
    if np.sum(mask) < 3:
        return np.nan
    t = t[mask] - np.mean(t[mask])
    y = y[mask] - np.mean(y[mask])
    denom = np.sum(t*t)
    return np.nan if denom <= 0 else float(np.sum(t*y) / denom)


# ==========================================================
# 4. Feature Extraction
# ==========================================================
def extract_features(video_path):
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    fps = 30.0 if fps <= 0 else fps
    dt = 1.0 / fps

    frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    total_time = (frame_count / fps) if (frame_count and fps > 0) else np.nan

    hip_pts, body_sizes = [], []
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

            result = pose.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            if result.pose_landmarks:
                lm = result.pose_landmarks.landmark
                hip_pts.append(center_point((lm[23].x, lm[23].y), (lm[24].x, lm[24].y)))
                body_sizes.append(robust_body_size_from_landmarks(lm))
            else:
                hip_pts.append((np.nan, np.nan))
                body_sizes.append(np.nan)
            frame_idx += 1

    cap.release()

    hip_x = np.array([p[0] for p in hip_pts])
    hip_y = np.array([p[1] for p in hip_pts])
    bs = np.array(body_sizes)

    if nan_ratio(hip_x) > HIP_MISSING_RATIO_MAX or nan_ratio(hip_y) > HIP_MISSING_RATIO_MAX:
        print(f"❌ Hip NaN 비율 초과 → {os.path.basename(video_path)}")
        return None
    if nan_ratio(bs) > BODYSIZE_MISSING_RATIO_MAX:
        print(f"❌ BodySize NaN 비율 초과 → {os.path.basename(video_path)}")
        return None

    hip_x = fill_missing(hip_x)
    hip_y = fill_missing(hip_y)
    bs = fill_missing(bs)

    bs_med = np.median(bs[np.isfinite(bs)])
    hip_xy = list(zip(hip_x, hip_y))

    hip_v = velocity_series(hip_xy, dt) / bs_med
    hip_a = acc_series(hip_v, dt)
    hip_j = jerk_series(hip_a, dt)
    path = np.sum(hip_v * dt)

    feats = {
        "id": extract_id_and_label(video_path)[0],
        "label": extract_id_and_label(video_path)[1],
        "total_time": total_time,
        "body_size_median": bs_med,

        # Fluency
        "fluency_hip_velocity_mean": float(np.mean(hip_v)),
        "fluency_hip_velocity_max": float(np.max(hip_v)),
        "fluency_hip_acc_mean": float(np.mean(np.abs(hip_a))),
        "fluency_hip_acc_max": float(np.max(np.abs(hip_a))),
        "fluency_hip_jerk_mean": float(np.mean(np.abs(hip_j))),
        "fluency_hip_jerk_max": float(np.max(np.abs(hip_j))),
        "fluency_hip_jerk_rms": float(np.sqrt(np.mean(hip_j ** 2))),
        "fluency_hip_path_length": float(path),
        "fluency_hip_path_per_sec": float(path / total_time) if total_time > 0 else np.nan,

        # Stability
        "stability_hip_velocity_sd": float(np.std(hip_v)),
        "stability_hip_acc_sd": float(np.std(hip_a)),
        "stability_hip_jerk_sd": float(np.std(hip_j)),

        # Trend
        "trend_jerk_abs_slope": linear_slope(np.abs(hip_j), dt),
        "trend_speed_slope": linear_slope(hip_v, dt),
    }

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
