import glob
import pandas as pd
from scipy.stats import ttest_ind, shapiro, levene, mannwhitneyu
import numpy as np

novice_files   = glob.glob('./csv_variability/novice/*.csv')
advanced_files = glob.glob('./csv_variability/advanced/*.csv')

df_list = []

# 파일 읽기
for fp in novice_files:
    tmp = pd.read_csv(fp)
    tmp['group'] = 'novice'
    tmp['participant'] = fp.split('/')[-1]
    df_list.append(tmp)

for fp in advanced_files:
    tmp = pd.read_csv(fp)
    tmp['group'] = 'advanced'
    tmp['participant'] = fp.split('/')[-1]
    df_list.append(tmp)

df_all = pd.concat(df_list, ignore_index=True)

# 데이터를 wide-format으로 변환
df_wide = df_all.pivot_table(
    index=['participant', 'group'],
    columns='landmark_index',
    values=['x_variability', 'y_variability', 'z_variability']
)

# 컬럼 이름 정리
df_wide.columns = [f"{var}_landmark{lm}" for var, lm in df_wide.columns]
df_wide = df_wide.reset_index()

print(f"✅ 데이터 shape: {df_wide.shape}")

# 그룹 분리
g0 = df_wide[df_wide['group'] == 'novice']
g1 = df_wide[df_wide['group'] == 'advanced']

features = [
    col for col in df_wide.columns
    if col not in ['participant', 'group']
]

results = []

for feat in features:
    x0 = g0[feat].dropna()
    x1 = g1[feat].dropna()

    if len(x0) < 3 or len(x1) < 3:
        continue

    # 정규성
    p_norm0 = shapiro(x0).pvalue
    p_norm1 = shapiro(x1).pvalue
    normal = p_norm0 > 0.05 and p_norm1 > 0.05

    # 등분산
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

    results.append({
        'feature': feat,
        'test': test_name,
        'statistic': stat,
        'p_value': p_val,
        'normal_g0_p': p_norm0,
        'normal_g1_p': p_norm1,
        'equal_var_p': p_levene
    })

res_df = pd.DataFrame(results).sort_values('p_value')
print(res_df)

res_df.to_csv('./t_test_variability_results.csv', index=False)
print("✅ 결과가 './t_test_variability_results.csv' 에 저장되었습니다.")
