import os, glob
import numpy as np
import pandas as pd
from scipy.stats import ttest_ind, mannwhitneyu, shapiro, levene

# 설정
excel_files = [f for f in glob.glob('./features_xlsx/*.xlsx') if '~$' not in f]
os.makedirs('./result', exist_ok=True)
excel_path = './result/features_ttest.xlsx'

# 엑셀 시트 번호 선택
sheet_index = 2
sheet_name_map = {2: 'Jerk'}
sheet_name = sheet_name_map.get(sheet_index, f'Sheet{sheet_index}')

# Mediapipe 관절 이름
landmark_names = [
    'nose', 'left_eye_inner', 'left_eye', 'left_eye_outer', 'right_eye_inner', 'right_eye',
    'right_eye_outer', 'left_ear', 'right_ear', 'mouth_left', 'mouth_right', 'left_shoulder',
    'right_shoulder', 'left_elbow', 'right_elbow', 'left_wrist', 'right_wrist', 'left_pinky',
    'right_pinky', 'left_index', 'right_index', 'left_thumb', 'right_thumb', 'left_hip',
    'right_hip', 'left_knee', 'right_knee', 'left_ankle', 'right_ankle', 'left_heel',
    'right_heel', 'left_foot_index', 'right_foot_index'
]

# t-test 함수
def t_test(df, sheet_name):
    g0 = df[df['label'] == 0] # 초보자
    g1 = df[df['label'] == 1] # 숙련자
    print(f'[{sheet_name}] 그룹0: {len(g0)}, 그룹1: {len(g1)}')

    features = [col for col in df.columns if col not in ['id', 'label'] and pd.api.types.is_numeric_dtype(df[col])]
    results = []

    for feat in features:
        x0 = g0[feat].dropna()
        x1 = g1[feat].dropna()
        if len(x0) < 3 or len(x1) < 3:
            continue

        # 정규성 검사
        p_norm0 = shapiro(x0).pvalue
        p_norm1 = shapiro(x1).pvalue
        normal = p_norm0 > 0.05 and p_norm1 > 0.05

        # 등분산 검사
        p_levene = levene(x0, x1).pvalue
        equal_var = p_levene > 0.05

        # cohen's d, 효과크기
        mean_diff = x0.mean() - x1.mean()
        pooled_sd = np.sqrt(((x0.std()**2) + (x1.std()**2)) / 2)
        cohen_d = mean_diff / pooled_sd if pooled_sd > 0 else np.nan

        p_val = np.nan
        ci_low, ci_high = np.nan, np.nan
        test_method = ''

        # 검정법 선택
        if normal:
            if equal_var:
                stat, p_val = ttest_ind(x0, x1, equal_var=True)
                test_method = 'Independent t-test'
            else:
                stat, p_val = ttest_ind(x0, x1, equal_var=False)
                test_method = 'Welch t-test'

            n0, n1 = len(x0), len(x1)
            se_diff = np.sqrt(x0.var(ddof=1)/n0 + x1.var(ddof=1)/n1)
            ci_low = mean_diff - 1.96 * se_diff
            ci_high = mean_diff + 1.96 * se_diff
        else:
            stat, p_val = mannwhitneyu(x0, x1, alternative='two-sided')
            test_method = 'Mann–Whitney U' # 비모수 검정

        ci_str = f'[{ci_low:.3f}, {ci_high:.3f}]' if normal else 'N/A'

        try:
            idx = int(feat.replace('landmark', '').split('_')[0])
            landmark_name = landmark_names[idx] if 0 <= idx < len(landmark_names) else 'unknown'
        except:
            landmark_name = 'unknown'

        results.append({
            'feature': feat,
            'landmark_name': landmark_name,
            'p_value': p_val,
            'cohen_d': cohen_d,
            '95%_CI': ci_str,
            'test_method': test_method
        })

    return pd.DataFrame(results).sort_values('p_value') if results else pd.DataFrame()

# 엑셀 저장
df_all = []

for file in excel_files:
    try:
        df = pd.read_excel(file, sheet_name=sheet_index)
        df_all.append(df)
    except Exception as e:
        print(f"⚠️ {file} (sheet {sheet_index}) → {e}")

if not df_all:
    raise ValueError("❌ 유효한 시트 데이터가 없습니다.")

df_concat = pd.concat(df_all, ignore_index=True)
res = t_test(df_concat, sheet_name)

with pd.ExcelWriter(excel_path) as writer:
    if not res.empty:
        res.to_excel(writer, sheet_name=f'{sheet_name}_ttest_all', index=False)
        sig = res[res['p_value'] < 0.05]
        if not sig.empty:
            sig.to_excel(writer, sheet_name=f'{sheet_name}_ttest_sig', index=False)
        else:
            pd.DataFrame({'message': ['No significant results (p<0.05)']}).to_excel(writer, sheet_name=f'{sheet_name}_ttest_sig', index=False)
    else:
        pd.DataFrame({'message': ['No valid results']}).to_excel(writer, sheet_name=f'{sheet_name}_ttest_all', index=False)
        pd.DataFrame({'message': ['No significant results (p<0.05)']}).to_excel(writer, sheet_name=f'{sheet_name}_ttest_sig', index=False)

print(f"✅ 엑셀 저장 완료: {excel_path}")
