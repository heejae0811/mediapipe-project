import os, glob
import numpy as np
import pandas as pd
from scipy.stats import ttest_ind, mannwhitneyu, shapiro, levene

# Mediapipe landmark 이름
landmark_names = [
    'nose', 'left_eye_inner', 'left_eye', 'left_eye_outer', 'right_eye_inner', 'right_eye',
    'right_eye_outer', 'left_ear', 'right_ear', 'mouth_left', 'mouth_right', 'left_shoulder',
    'right_shoulder', 'left_elbow', 'right_elbow', 'left_wrist', 'right_wrist', 'left_pinky',
    'right_pinky', 'left_index', 'right_index', 'left_thumb', 'right_thumb', 'left_hip',
    'right_hip', 'left_knee', 'right_knee', 'left_ankle', 'right_ankle', 'left_heel',
    'right_heel', 'left_foot_index', 'right_foot_index'
]

def t_test(df, sheet_name):
    # 그룹 분리
    g0 = df[df['label'] == 0]
    g1 = df[df['label'] == 1]
    print(f'[{sheet_name}] 그룹0: {len(g0)}, 그룹1: {len(g1)}')

    # 분석할 피처
    features = [
        col for col in df.columns
        if col not in ['id', 'label'] and pd.api.types.is_numeric_dtype(df[col])
    ]

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

        # cohen's d
        mean_diff = x0.mean() - x1.mean()
        pooled_sd = np.sqrt(((x0.std()**2) + (x1.std()**2)) / 2)
        cohen_d = mean_diff / pooled_sd if pooled_sd > 0 else np.nan

        p_val = np.nan
        ci_low, ci_high = np.nan, np.nan
        test_method = ''

        # 테스트 선택
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
            test_method = 'Mann–Whitney U'

        ci_str = f'[{ci_low:.3f}, {ci_high:.3f}]' if normal else 'N/A'

        # landmark 이름 매핑
        try:
            idx = int(feat.replace('landmark', '').split('_')[0])
            if 0 <= idx < len(landmark_names):
                landmark_name = landmark_names[idx]
            else:
                landmark_name = 'unknown'
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

    if results:
        return pd.DataFrame(results).sort_values('p_value')
    else:
        print(f'⚠️ [{sheet_name}] 유효한 feature가 없습니다.')
        return pd.DataFrame()

# 모든 파일의 시트1/시트2를 모으기
xlsx_files = glob.glob('./features_xlsx/*.xlsx')
df_sheet1_all = []
df_sheet2_all = []

for file in xlsx_files:
    df1 = pd.read_excel(file, sheet_name=0)
    df2 = pd.read_excel(file, sheet_name=1)
    df_sheet1_all.append(df1)
    df_sheet2_all.append(df2)

df_sheet1 = pd.concat(df_sheet1_all, ignore_index=True)
df_sheet2 = pd.concat(df_sheet2_all, ignore_index=True)

# t_test 함수 실행
res1 = t_test(df_sheet1, 'Sheet1')
res2 = t_test(df_sheet2, 'Sheet2')

# 엑셀 저장
os.makedirs('./result', exist_ok=True)
save_path = './result/features_ttest.xlsx'

with pd.ExcelWriter(save_path) as writer:
    if not res1.empty:
        # 전체
        res1.to_excel(writer, sheet_name='position_ttest', index=False)
        # p-value < 0.05
        res1_sig = res1[res1['p_value'] < 0.05]
        if not res1_sig.empty:
            res1_sig.to_excel(writer, sheet_name='position_ttest_sig', index=False)
        else:
            pd.DataFrame({'message': ['No significant results (p<0.05)']}).to_excel(writer, sheet_name='position_ttest_sig', index=False)
    else:
        pd.DataFrame({'message': ['No valid results']}).to_excel(writer, sheet_name='position_ttest', index=False)
        pd.DataFrame({'message': ['No significant results (p<0.05)']}).to_excel(writer, sheet_name='position_ttest_sig', index=False)

    if not res2.empty:
        # 전체
        res2.to_excel(writer, sheet_name='normalized_position_ttest', index=False)
        # p-value < 0.05
        res2_sig = res2[res2['p_value'] < 0.05]
        if not res2_sig.empty:
            res2_sig.to_excel(writer, sheet_name='normalized_position_ttest_sig', index=False)
        else:
            pd.DataFrame({'message': ['No significant results (p<0.05)']}).to_excel(writer, sheet_name='normalized_position_ttest_sig', index=False)
    else:
        pd.DataFrame({'message': ['No valid results']}).to_excel(writer, sheet_name='normalized_position_ttest', index=False)
        pd.DataFrame({'message': ['No significant results (p<0.05)']}).to_excel(writer, sheet_name='normalized_position_ttest_sig', index=False)

print(f'✅ 분석 완료: {save_path}')
