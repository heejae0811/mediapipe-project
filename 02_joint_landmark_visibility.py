import os, glob
import pandas as pd

# 설정
excel_files = [f for f in glob.glob('./features_xlsx/*.xlsx') if '~$' not in f]
output_dir = './result'
os.makedirs(output_dir, exist_ok=True)

print(f"{len(excel_files)}개의 파일을 분석합니다.")

df_list = []

# 엑셀 시트별로 visibility 추출
for file in excel_files:
    try:
        df = pd.read_excel(file, sheet_name=2, engine='openpyxl')
        vis_cols = [col for col in df.columns if 'visibility' in col]
        if not vis_cols:
            print(f"⚠️ {file} → visibility 컬럼 없습니다.")
            continue
        df_vis = df[vis_cols]
        df_list.append(df_vis)
    except Exception as e:
        print(f"⚠️ {file} → {e}")

if not df_list:
    raise ValueError("❌ visibility 컬럼을 가진 데이터가 없습니다.")

# 관절별 평균 계산
df_all = pd.concat(df_list, ignore_index=True)
mean_vis = df_all.mean().sort_values(ascending=False).reset_index()
mean_vis.columns = ['landmark', 'mean_visibility']

# Mediapipe 관절 이름
landmark_names = [
    'nose', 'left_eye_inner', 'left_eye', 'left_eye_outer', 'right_eye_inner', 'right_eye',
    'right_eye_outer', 'left_ear', 'right_ear', 'mouth_left', 'mouth_right', 'left_shoulder',
    'right_shoulder', 'left_elbow', 'right_elbow', 'left_wrist', 'right_wrist', 'left_pinky',
    'right_pinky', 'left_index', 'right_index', 'left_thumb', 'right_thumb', 'left_hip',
    'right_hip', 'left_knee', 'right_knee', 'left_ankle', 'right_ankle', 'left_heel',
    'right_heel', 'left_foot_index', 'right_foot_index'
]

# landmark 번호 → 이름 매핑
mean_vis['landmark_index'] = mean_vis['landmark'].str.extract(r'(\d+)').astype(int)
mean_vis['landmark_name'] = mean_vis['landmark_index'].apply(lambda x: landmark_names[x])

# 엑셀 저장
excel_path = os.path.join(output_dir, 'joint_landmark_visibility.xlsx')
df_result = mean_vis[['landmark', 'landmark_name', 'mean_visibility']]
df_result.to_excel(excel_path, index=False)

print(f"✅ 엑셀 저장 완료: {excel_path}")
