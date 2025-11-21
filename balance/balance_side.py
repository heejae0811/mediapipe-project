import cv2
import mediapipe as mp
import numpy as np
import pandas as pd
import os


mp_pose = mp.solutions.pose


# 좌표 변환
def get_xy(lm, w, h):
    return int(lm.x * w), int(lm.y * h)


# 벡터 각도 계산
def angle_between(v1, v2):
    v1 = np.array(v1, dtype=float)
    v2 = np.array(v2, dtype=float)
    denom = (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-7)
    cosang = np.dot(v1, v2) / denom
    cosang = np.clip(cosang, -1.0, 1.0)
    return np.degrees(np.arccos(cosang))


# 각도에 따른 색상 결정
def get_color_for_angle(angle):
    if angle >= 160:
        return (0, 255, 0)     # 초록
    elif angle >= 140:
        return (0, 255, 255)   # 노랑
    else:
        return (0, 0, 255)     # 빨강


# 메인 함수
def analyze_side_video(video_path):

    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    base = os.path.splitext(video_path)[0]
    excel_path = base + "_results.xlsx"
    output_video_path = base + "_tracked.mp4"

    # 출력 영상 생성
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (w, h))

    pose = mp_pose.Pose(
        static_image_mode=False,
        model_complexity=1,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    )

    traj_canvas = np.zeros((h, w, 3), dtype=np.uint8)

    toe_rel_list = []        # 발-힙 상대 X값
    trunk_angle_list = []    # 몸통 각도

    frame_idx = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_idx += 1
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = pose.process(rgb)

        if result.pose_landmarks:
            lm = result.pose_landmarks.landmark

            # 왼쪽 엄지발가락 + 왼쪽 고관절
            toe = get_xy(lm[31], w, h)   # LEFT_FOOT_INDEX
            hip = get_xy(lm[23], w, h)   # LEFT_HIP

            toe_x = toe[0]
            hip_x = hip[0]

            toe_rel_x = toe_x - hip_x
            toe_rel_list.append(toe_rel_x)

            # 발 궤적 누적
            cv2.circle(traj_canvas, toe, 3, (0, 255, 255), -1)

            # 몸통 각도 (어깨-고관절-무릎)
            shoulder = get_xy(lm[11], w, h)
            knee = get_xy(lm[25], w, h)

            v1 = (shoulder[0] - hip[0], shoulder[1] - hip[1])
            v2 = (knee[0] - hip[0], knee[1] - hip[1])

            angle = angle_between(v1, v2)
            trunk_angle_list.append(angle)

            # 시각화 (색 변화)
            color = get_color_for_angle(angle)
            overlay = cv2.addWeighted(frame, 0.7, traj_canvas, 1.0, 0)

            # 선 색상
            cv2.line(overlay, hip, shoulder, color, 4)
            cv2.line(overlay, hip, knee, color, 4)

            # 각도 표시
            cv2.putText(
                overlay,
                f"Trunk angle: {angle:.1f} deg",
                (30, 60),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.3,
                color,
                3,
                cv2.LINE_AA
            )

            out.write(overlay)

        else:
            overlay = cv2.addWeighted(frame, 0.7, traj_canvas, 1.0, 0)
            out.write(overlay)

    cap.release()
    out.release()
    pose.close()

    # 발 앞뒤 이동 (hip-relative)
    toe_rel_arr = np.array(toe_rel_list)
    start_rel = toe_rel_arr[0]
    movement = toe_rel_arr - start_rel

    forward_max = movement.max()
    backward_max = movement.min()

    # 각도 통계
    trunk_angles = np.array(trunk_angle_list)
    trunk_angle_max = trunk_angles.max()
    trunk_angle_min = trunk_angles.min()
    trunk_angle_mean = trunk_angles.mean()

    # 엑셀 저장
    df = pd.DataFrame([[
        forward_max,
        backward_max,
        trunk_angle_max,
        trunk_angle_min,
        trunk_angle_mean
    ]], columns=[
        "forward_max_rel_x",
        "backward_max_rel_x",
        "trunk_angle_max",
        "trunk_angle_min",
        "trunk_angle_mean"
    ])

    df.to_excel(excel_path, index=False)

    print("엑셀 저장 완료 →", excel_path)
    print("시각화 영상 저장 완료 →", output_video_path)

    return excel_path, output_video_path


# 실행
if __name__ == "__main__":
    analyze_side_video("side3.MOV")
