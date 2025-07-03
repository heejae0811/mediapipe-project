import glob
import pandas as pd
from scipy.stats import shapiro, levene, ttest_ind, mannwhitneyu

# 1) 파일 목록
novice_files   = glob.glob('./csv_variability/novice/*.csv')
advanced_files = glob.glob('./csv_variability/advanced/*.csv')

# 2) 데이터 불러오기
df_list = []
for fp in novice_files:
    df = pd.read_csv(fp); df['group']='novice'; df_list.append(df)
for fp in advanced_files:
    df = pd.read_csv(fp); df['group']='advanced'; df_list.append(df)
df_all = pd.concat(df_list, ignore_index=True)

metrics   = ['x_variability', 'y_variability', 'z_variability']
landmarks = sorted(df_all['landmark_index'].unique())

results = []
for lm in landmarks:
    sub = df_all[df_all['landmark_index']==lm]
    nov = sub[sub['group']=='novice']
    adv = sub[sub['group']=='advanced']

    for m in metrics:
        data1 = nov[m].dropna()
        data2 = adv[m].dropna()
        # 1) 정규성 검정
        p1 = shapiro(data1).pvalue
        p2 = shapiro(data2).pvalue
        normal = (p1>0.05 and p2>0.05)

        # 2) 등분산 검정
        p_levene = levene(data1, data2).pvalue
        equal_var = (p_levene>0.05)

        # 3) 적절한 그룹 비교
        if normal:
            # 정규분포이면 t-검정
            stat, p = ttest_ind(data1, data2, equal_var=equal_var)
            test_name = 't-test (Welch)' if not equal_var else 't-test'
        else:
            # 비정규 분포이면 비모수 검정
            stat, p = mannwhitneyu(data1, data2)
            test_name = 'Mann-Whitney U'

        results.append({
            'landmark_index': lm,
            'metric': m,
            'test': test_name,
            'statistic': stat,
            'p_value': p,
            'normal_p1': p1,
            'normal_p2': p2,
            'levene_p': p_levene
        })

res_df = pd.DataFrame(results)
res_df.to_csv('ttest_with_checks.csv', index=False, encoding='utf-8-sig')
print(res_df)
