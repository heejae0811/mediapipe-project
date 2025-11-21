import cv2
import mediapipe as mp
import numpy as np
import pandas as pd
import os


mp_pose = mp.solutions.pose


# 랜드마크 → (x, y) 픽셀 좌표 변환
def get_xy(landmark, width, height):
    return int(landmark.x * width), int(landmark.y * height)


# 메인 함수
def track_body_parts(video_path):
    cap = cv2.VideoCapture(video_path)

    fps = cap.get(cv2.CAP_PROP_FPS)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    base = os.path.splitext(video_path)[0]
    output_path = base + "_tracked.mp4"
    excel_path  = base + "_results.xlsx"

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(output_path, fourcc, fps, (w, h))

    pose = mp_pose.Pose(
        static_image_mode=False,
        model_complexity=1,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    )

    traj_canvas = np.zeros((h, w, 3), dtype=np.uint8)

    ls_pts, rs_pts, lh_pts, rh_pts = [], [], [], []

    # 누적 거리 변수
    ls_dx_sum = ls_dy_sum = ls_total_sum = 0
    rs_dx_sum = rs_dy_sum = rs_total_sum = 0
    lh_dx_sum = lh_dy_sum = lh_total_sum = 0
    rh_dx_sum = rh_dy_sum = rh_total_sum = 0

    frame_idx = 0

    # 프레임 루프
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_idx += 1
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = pose.process(rgb)

        if result.pose_landmarks:
            lm = result.pose_landmarks.landmark

            ls = get_xy(lm[11], w, h)
            rs = get_xy(lm[12], w, h)
            lh = get_xy(lm[23], w, h)
            rh = get_xy(lm[24], w, h)

            ls_pts.append(ls)
            rs_pts.append(rs)
            lh_pts.append(lh)
            rh_pts.append(rh)

            if frame_idx > 1:
                # 이동량 계산
                ls_dx = abs(ls_pts[-1][0] - ls_pts[-2][0])
                rs_dx = abs(rs_pts[-1][0] - rs_pts[-2][0])
                lh_dx = abs(lh_pts[-1][0] - lh_pts[-2][0])
                rh_dx = abs(rh_pts[-1][0] - rh_pts[-2][0])

                ls_dy = abs(ls_pts[-1][1] - ls_pts[-2][1])
                rs_dy = abs(rs_pts[-1][1] - rs_pts[-2][1])
                lh_dy = abs(lh_pts[-1][1] - lh_pts[-2][1])
                rh_dy = abs(rh_pts[-1][1] - rh_pts[-2][1])

                ls_total = np.sqrt(ls_dx**2 + ls_dy**2)
                rs_total = np.sqrt(rs_dx**2 + rs_dy**2)
                lh_total = np.sqrt(lh_dx**2 + lh_dy**2)
                rh_total = np.sqrt(rh_dx**2 + rh_dy**2)

                # 누적 합계
                ls_dx_sum += ls_dx
                ls_dy_sum += ls_dy
                ls_total_sum += ls_total

                rs_dx_sum += rs_dx
                rs_dy_sum += rs_dy
                rs_total_sum += rs_total

                lh_dx_sum += lh_dx
                lh_dy_sum += lh_dy
                lh_total_sum += lh_total

                rh_dx_sum += rh_dx
                rh_dy_sum += rh_dy
                rh_total_sum += rh_total

            # 궤적 그리기
            cv2.circle(traj_canvas, ls, 2, (0, 255, 0), -1)
            cv2.circle(traj_canvas, rs, 2, (0, 0, 255), -1)
            cv2.circle(traj_canvas, lh, 2, (255, 255, 0), -1)
            cv2.circle(traj_canvas, rh, 2, (255, 0, 255), -1)

        overlay = cv2.addWeighted(frame, 0.7, traj_canvas, 1.0, 0)
        out.write(overlay)

    # 종료
    cap.release()
    out.release()
    pose.close()


    # 최종 데이터 저장
    df = pd.DataFrame([[
        ls_dx_sum, ls_dy_sum, ls_total_sum,
        rs_dx_sum, rs_dy_sum, rs_total_sum,
        lh_dx_sum, lh_dy_sum, lh_total_sum,
        rh_dx_sum, rh_dy_sum, rh_total_sum
    ]], columns=[
        "ls_dx", "ls_dy", "ls_total",
        "rs_dx", "rs_dy", "rs_total",
        "lh_dx", "lh_dy", "lh_total",
        "rh_dx", "rh_dy", "rh_total"
    ])

    df.to_excel(excel_path, index=False)

    print("궤적 영상 저장 완료:", output_path)
    print("총 이동거리 엑셀 저장 완료:", excel_path)

    return output_path, excel_path


if __name__ == "__main__":
    track_body_parts("front1.MOV")
