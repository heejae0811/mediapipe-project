import os
import glob
import pandas as pd

# Mediapipe 관절 이름
landmark_names = [
    'nose', 'left_eye_inner', 'left_eye', 'left_eye_outer', 'right_eye_inner', 'right_eye',
    'right_eye_outer', 'left_ear', 'right_ear', 'mouth_left', 'mouth_right', 'left_shoulder',
    'right_shoulder', 'left_elbow', 'right_elbow', 'left_wrist', 'right_wrist', 'left_pinky',
    'right_pinky', 'left_index', 'right_index', 'left_thumb', 'right_thumb', 'left_hip',
    'right_hip', 'left_knee', 'right_knee', 'left_ankle', 'right_ankle', 'left_heel',
    'right_heel', 'left_foot_index', 'right_foot_index'
]

# 경로 설정
input_folder = './features_xlsx'
output_folder = './result'
os.makedirs(output_folder, exist_ok=True)

# 모든 엑셀 파일 불러오기
excel_files = [f for f in glob.glob(os.path.join(input_folder, '*.xlsx')) if '~$' not in f]
if not excel_files:
    raise FileNotFoundError(f"❌ '{input_folder}' 폴더에 엑셀 파일이 없습니다.")

print(f"{len(excel_files)}개의 파일을 찾았습니다.")

df_list = []

# 파일별로 3번째 시트의 visibility 컬럼만 추출
for file in excel_files:
    try:
        df = pd.read_excel(file, sheet_name=2, engine='openpyxl')
        vis_cols = [col for col in df.columns if 'visibility' in col]
        if not vis_cols:
            print(f"⚠️ {file} → visibility 컬럼 없음")
            continue
        df_vis = df[vis_cols]
        df_list.append(df_vis)
    except Exception as e:
        print(f"⚠️ {file} → {e}")

if not df_list:
    raise ValueError("❌ visibility 컬럼을 가진 데이터가 없습니다.")

# 하나로 합치기
df_all = pd.concat(df_list, ignore_index=True)

# 관절별 평균 계산
mean_vis = df_all.mean().sort_values(ascending=False).reset_index()
mean_vis.columns = ['landmark', 'mean_visibility']

# landmark 번호 → 이름 매핑
mean_vis['landmark_index'] = mean_vis['landmark'].str.extract(r'(\d+)').astype(int)
mean_vis['landmark_name'] = mean_vis['landmark_index'].apply(lambda x: landmark_names[x])

# 결과 정리
df_result = mean_vis[['landmark', 'landmark_name', 'mean_visibility']]

# 엑셀로 저장
excel_path = os.path.join(output_folder, 'joint_landmark_visibility.xlsx')
df_result.to_excel(excel_path, index=False)

print(f"✅ 엑셀로 저장 완료: {excel_path}")
