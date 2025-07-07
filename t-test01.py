import glob
import pandas as pd
from scipy.stats import ttest_ind, mannwhitneyu, shapiro, levene
import numpy as np

# 파일 불러오기
file_paths = glob.glob('./csv_nose_pelvis/features_climbing*.csv')
df_list = [pd.read_csv(f) for f in file_paths]
df = pd.concat(df_list, ignore_index=True)

# 그룹 분리
g0 = df[df['label'] == 0]
g1 = df[df['label'] == 1]
print(f"✔ 그룹0 샘플 수: {len(g0)}, 그룹1 샘플 수: {len(g1)}")

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
        continue  # 샘플이 너무 적으면 건너뜀

    # 정규성 검사
    p_norm0 = shapiro(x0).pvalue
    p_norm1 = shapiro(x1).pvalue
    normal = p_norm0 > 0.05 and p_norm1 > 0.05

    # 등분산 검사
    p_levene = levene(x0, x1).pvalue
    equal_var = p_levene > 0.05

    # 테스트 선택
    if normal:
        if equal_var:
            test_name = 'Independent t-test'
            stat, p_val = ttest_ind(x0, x1, equal_var=True)
        else:
            test_name = 'Welch t-test'
            stat, p_val = ttest_ind(x0, x1, equal_var=False)
    else:
        test_name = 'Mann–Whitney U'
        stat, p_val = mannwhitneyu(x0, x1, alternative='two-sided')

    # 효과 크기
    mean_diff = x0.mean() - x1.mean()
    pooled_sd = np.sqrt(((x0.std() ** 2) + (x1.std() ** 2)) / 2)
    cohen_d = mean_diff / pooled_sd if pooled_sd > 0 else np.nan

    # 기존 r_effect (비모수 효과 크기 근사)
    r_raw = abs(stat)
    n_total = len(x0) + len(x1)
    r_norm = r_raw / np.sqrt(n_total)  # 정규화된 r_effect

    results.append({
        'feature': feat,
        'test': test_name,
        'statistic': stat,
        'p_value': p_val,
        'normal_g0_p': p_norm0,
        'normal_g1_p': p_norm1,
        'equal_var_p': p_levene,
        'cohen_d': cohen_d,
        'r_effect_raw': r_raw,
        'r_effect_normalized': r_norm
    })

res_df = pd.DataFrame(results).sort_values('p_value')

# 결과 출력
print(res_df)

# CSV로 저장
res_df.to_csv('./t_test_nose_pelvis_results.csv', index=False)
print("✅ 결과가 './t_test_nose_pelvis_results.csv' 에 저장되었습니다.")
