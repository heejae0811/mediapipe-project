import os
import re
import cv2
import joblib
import numpy as np
import pandas as pd
import mediapipe as mp
from flask import Flask, request, jsonify
from flask_cors import CORS  # 1. CORS 라이브러리 임포트


# ... (특징 추출 코드는 이전과 동일하므로 생략) ...
# ------ Helper Functions (그대로 사용) ------
def safe_divide(numerator, denominator, default=np.nan):
    if denominator == 0 or np.isnan(denominator) or denominator is None:
        return default
    return numerator / denominator


def compute_stats(arr):
    arr = np.array(arr)
    valid_arr = arr[~np.isnan(arr)]
    if len(valid_arr) > 3:
        return {'mean': np.mean(valid_arr), 'max': np.max(valid_arr), 'std': np.std(valid_arr)}
    return {k: np.nan for k in ['mean', 'max', 'std']}


def compute_derivatives(position_series, fps):
    position_series = np.array(position_series)
    if len(position_series) < 2: return np.array([]), np.array([]), np.array([])
    speed = np.diff(position_series) * fps
    if len(speed) < 2: return speed, np.array([]), np.array([])
    acceleration = np.diff(speed) * fps
    if len(acceleration) < 2: return speed, acceleration, np.array([])
    jerk = np.diff(acceleration) * fps
    return speed, acceleration, jerk


def compute_xy_distance_safe(x, y):
    min_len = min(len(x), len(y))
    if min_len < 2: return 0
    x_trimmed, y_trimmed = np.array(x[:min_len]), np.array(y[:min_len])
    try:
        return np.sum(np.linalg.norm(np.diff(np.stack([x_trimmed, y_trimmed], axis=1), axis=0), axis=1))
    except Exception:
        return 0


def compute_body_size(trajectory, threshold=0.5):
    def get_valid_distance(idx1, idx2):
        x1, y1, v1 = np.array(trajectory[idx1]['x']), np.array(trajectory[idx1]['y']), np.array(
            trajectory[idx1]['visibility'])
        x2, y2, v2 = np.array(trajectory[idx2]['x']), np.array(trajectory[idx2]['y']), np.array(
            trajectory[idx2]['visibility'])
        valid = (v1 > threshold) & (v2 > threshold)
        if np.any(valid):
            return np.mean(np.sqrt((x1[valid] - x2[valid]) ** 2 + (y1[valid] - y2[valid]) ** 2))
        return None

    left = get_valid_distance(11, 23)
    right = get_valid_distance(12, 24)
    if left and right: return (left + right) / 2
    return left or right or 1.0


def flatten_rows(rows):
    flat = {}
    for row in rows:
        landmark = row['landmark']
        for k, v in row.items():
            if k not in ['id', 'label', 'landmark']:
                flat[f'landmark{landmark}_{k}'] = v
    return flat


# ------ Main Feature Extraction Function (서버용으로 수정) ------
def extract_features_from_video(VIDEO_PATH):
    # ... (함수 내용은 이전과 동일) ...
    FILE_ID = os.path.splitext(os.path.basename(VIDEO_PATH))[0]
    LABEL = 0  # 예측 시에는 라벨이 필요 없으므로 0으로 고정

    cap = None
    pose = None

    try:
        cap = cv2.VideoCapture(VIDEO_PATH)
        if not cap.isOpened(): raise IOError(f"Cannot open video: {VIDEO_PATH}")

        fps = cap.get(cv2.CAP_PROP_FPS)
        if fps <= 0: fps = 30

        mp_pose = mp.solutions.pose
        pose = mp_pose.Pose(static_image_mode=False, model_complexity=1, min_detection_confidence=0.5,
                            min_tracking_confidence=0.5)

        trajectory = {i: {'x': [], 'y': [], 'visibility': []} for i in range(33)}
        frame_idx = 0

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret: break

            image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = pose.process(image_rgb)
            if results.pose_landmarks:
                for i, lm in enumerate(results.pose_landmarks.landmark):
                    trajectory[i]['x'].append(lm.x)
                    trajectory[i]['y'].append(lm.y)
                    trajectory[i]['visibility'].append(lm.visibility)
            else:
                for i in range(33):
                    trajectory[i]['x'].append(np.nan)
                    trajectory[i]['y'].append(np.nan)
                    trajectory[i]['visibility'].append(0.0)
            frame_idx += 1

        body_size = compute_body_size(trajectory)
        total_time = frame_idx / fps

        distance_rows, speed_rows, acceleration_rows, jerk_rows = [], [], [], []
        for i in range(33):
            x, y = np.array(trajectory[i]['x']), np.array(trajectory[i]['y'])
            valid_x, valid_y = x[~np.isnan(x)], y[~np.isnan(y)]
            dx, dy = np.sum(np.abs(np.diff(valid_x))) if len(valid_x) > 1 else 0, np.sum(
                np.abs(np.diff(valid_y))) if len(valid_y) > 1 else 0
            dxy = compute_xy_distance_safe(trajectory[i]['x'], trajectory[i]['y'])
            sx, ax, jx = compute_derivatives(valid_x, fps)
            sy, ay, jy = compute_derivatives(valid_y, fps)
            min_len = min(len(valid_x), len(valid_y))
            sxy, axy, jxy = compute_derivatives(np.sqrt(valid_x[:min_len] ** 2 + valid_y[:min_len] ** 2),
                                                fps) if min_len > 1 else (np.array([]), np.array([]), np.array([]))

            distance_rows.append(
                {'id': FILE_ID, 'label': LABEL, 'landmark': i, 'distance_x_raw': dx, 'distance_y_raw': dy,
                 'distance_xy_raw': dxy,
                 'distance_x_timeBodyNorm': safe_divide(dx, total_time * body_size),
                 'distance_y_timeBodyNorm': safe_divide(dy, total_time * body_size),
                 'distance_xy_timeBodyNorm': safe_divide(dxy, total_time * body_size)})
            for axis_name, (spd, acc, jrk) in {'x': (sx, ax, jx), 'y': (sy, ay, jy), 'xy': (sxy, axy, jxy)}.items():
                spd_stats, acc_stats, jrk_stats = compute_stats(spd), compute_stats(acc), compute_stats(jrk)
                speed_rows.append({'id': FILE_ID, 'label': LABEL, 'landmark': i,
                                   **{f'speed_{axis_name}_{k}_raw': v for k, v in spd_stats.items()},
                                   **{f'speed_{axis_name}_{k}_bodyNorm': safe_divide(v, body_size) for k, v in
                                      spd_stats.items()}})
                acceleration_rows.append({'id': FILE_ID, 'label': LABEL, 'landmark': i,
                                          **{f'acceleration_{axis_name}_{k}_raw': v for k, v in acc_stats.items()},
                                          **{f'acceleration_{axis_name}_{k}_bodyNorm': safe_divide(v, body_size) for
                                             k, v in acc_stats.items()}})
                jerk_rows.append({'id': FILE_ID, 'label': LABEL, 'landmark': i,
                                  **{f'jerk_{axis_name}_{k}_raw': v for k, v in jrk_stats.items()},
                                  **{f'jerk_{axis_name}_{k}_bodyNorm': safe_divide(v, body_size) for k, v in
                                     jrk_stats.items()}})

        merged_flat = {'id': FILE_ID, 'label': LABEL, **flatten_rows(distance_rows), **flatten_rows(speed_rows),
                       **flatten_rows(acceleration_rows), **flatten_rows(jerk_rows)}

        return pd.DataFrame([merged_flat])

    except Exception as e:
        print(f"❌ 비디오 처리 중 심각한 오류 발생: {e}")
        return pd.DataFrame()
    finally:
        if cap is not None: cap.release()
        if pose is not None: pose.close()


# ====================================================================
#  2. Flask 서버 코드 (CORS 적용됨)
# ====================================================================
app = Flask(__name__)
CORS(app)  # 2. CORS(app)를 추가하여 모든 요청을 허용

# ------ 서버 시작 시 모델과 특징 리스트를 미리 로드 ------
MODEL_PATH = "/Users/ihuijae/mediapipe-project/result/best_climbing_model.pkl"
FEATURES_PATH = "/Users/ihuijae/mediapipe-project/result/selected_features.pkl"
# ... (이하 코드는 이전과 동일하므로 생략) ...
print("🧠 모델과 특징 리스트를 로드합니다...")
try:
    model = joblib.load(MODEL_PATH)
    selected_features = joblib.load(FEATURES_PATH)
    print("✅ 로드 성공!")
    print(f"   - 모델: {MODEL_PATH}")
    print(f"   - 필요한 특징 수: {len(selected_features)}개")
except FileNotFoundError:
    print(f"❌ 에러: 모델 또는 특징 파일을 찾을 수 없습니다. 경로를 확인하세요.")
    model = None
    selected_features = None


# ------ Flutter 앱이 호출할 API 엔드포인트 ------
@app.route("/predict", methods=["POST"])
def predict():
    if not model or not selected_features:
        return jsonify({"error": "모델이 로드되지 않았습니다. 서버 로그를 확인하세요."}), 500

    if 'video' not in request.files:
        return jsonify({"error": "요청에 'video' 파일이 없습니다."}), 400

    video_file = request.files['video']
    temp_video_path = f"temp_{video_file.filename}"
    video_file.save(temp_video_path)
    print(f"\n📥 '{video_file.filename}' 파일 수신 및 임시 저장 완료.")

    try:
        # 1. 비디오에서 모든 특징 추출
        print("🛠️ 영상 분석 및 특징 추출 시작...")
        all_features_df = extract_features_from_video(temp_video_path)
        print("✅ 특징 추출 완료.")

        if all_features_df.empty:
            return jsonify({"error": "영상 분석 중 특징을 추출하지 못했습니다."}), 500

        # 2. 학습에 사용된 'selected_features'만 선택하고 순서 맞추기
        print(f"✨ {len(selected_features)}개의 주요 특징 선택 중...")
        # 모델이 학습한 특징 순서와 이름에 정확히 맞추기
        # 없는 컬럼은 NaN으로 채워짐
        all_features_reordered = all_features_df.reindex(columns=selected_features)

        # 3. 모델로 예측 수행
        print("🤖 모델 예측 수행...")
        # NaN 값을 0으로 채움
        prediction_result = model.predict(all_features_reordered.fillna(0))[0]
        prediction_proba = model.predict_proba(all_features_reordered.fillna(0))[0]

        result_label = 'Good' if prediction_result == 1 else 'Bad'
        confidence = prediction_proba[prediction_result]
        print(f"👍 예측 결과: {result_label} (신뢰도: {confidence:.2f})")

        # 4. Flutter 앱으로 결과 전송
        return jsonify({"prediction": result_label, "confidence": float(confidence)})

    except Exception as e:
        print(f"🔥 예측 처리 중 에러 발생: {e}")
        return jsonify({"error": f"예측 처리 중 서버 에러 발생: {e}"}), 500
    finally:
        # 임시 파일 삭제
        if os.path.exists(temp_video_path):
            os.remove(temp_video_path)
            print(f"🗑️ 임시 파일 '{temp_video_path}' 삭제 완료.")


# ------ 서버 실행 ------
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5001, debug=True)
