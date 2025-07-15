import pandas as pd
import glob

# 1. 파일 경로 설정
file_paths = glob.glob("./csv_features/*.csv")

# 2. visibility 컬럼만 추출해서 합치기
df_list = []
for file in file_paths:
    df = pd.read_csv(file)
    # visibility 컬럼만 선택
    vis_cols = [col for col in df.columns if 'visibility' in col]
    df_vis = df[vis_cols]
    df_list.append(df_vis)

# 3. 모든 파일의 데이터프레임을 하나로 합치기
df_all = pd.concat(df_list, ignore_index=True)

# 4. 각 관절별 전체 평균 계산
mean_visibilities = df_all.mean().sort_values(ascending=False)

# 5. Series → DataFrame
df_mean = mean_visibilities.reset_index()
df_mean.columns = ['landmark', 'mean_visibility']

# 6. landmark 번호 추출
df_mean['landmark_index'] = df_mean['landmark'].str.extract(r'(\d+)').astype(int)

# 7. Mediapipe landmark 이름 리스트
landmark_names = [
    "nose", "left_eye_inner", "left_eye", "left_eye_outer", "right_eye_inner", "right_eye",
    "right_eye_outer", "left_ear", "right_ear", "mouth_left", "mouth_right", "left_shoulder",
    "right_shoulder", "left_elbow", "right_elbow", "left_wrist", "right_wrist", "left_pinky",
    "right_pinky", "left_index", "right_index", "left_thumb", "right_thumb", "left_hip",
    "right_hip", "left_knee", "right_knee", "left_ankle", "right_ankle", "left_heel",
    "right_heel", "left_foot_index", "right_foot_index"
]

# 8. 이름 매칭
df_mean['landmark_name'] = df_mean['landmark_index'].apply(lambda x: landmark_names[x])

# 9. 열 순서 정리
df_mean = df_mean[['landmark', 'landmark_name', 'mean_visibility']]

# 10. CSV로 저장
df_mean.to_csv("joint_landmark_visibility_mean.csv", index=False)

print("\n✅ joint_landmark_visibility_mean.csv 저장 완료")
