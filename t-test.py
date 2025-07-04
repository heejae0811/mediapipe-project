import glob
import pandas as pd
from scipy.stats import ttest_ind

# 파일 불러오기
file_paths = glob.glob('./data/features_climbing*.csv')  # ← 폴더 경로에 맞게 수정
df_list = [pd.read_csv(f) for f in file_paths]
df = pd.concat(df_list, ignore_index=True)

# 라벨 나누기
g0 = df[df['label'] == 0]
g1 = df[df['label'] == 1]
print(f"✔ 그룹0 샘플 수: {len(g0)}, 그룹1 샘플 수: {len(g1)}")

# 변수 추출
features = [
    col for col in df.columns
    if col not in ['id', 'label'] and pd.api.types.is_numeric_dtype(df[col])
]
print(f"✔ 분석할 피처 {len(features)}개: {features}")

# t-test
results = []
for feat in features:
    x0 = g0[feat].dropna()
    x1 = g1[feat].dropna()
    if len(x0) < 2 or len(x1) < 2:
        continue  # 샘플 수 부족하면 건너뜀

    t_stat, p_val = ttest_ind(x0, x1, equal_var=False)
    results.append({
        'feature': feat,
        't_statistic': t_stat,
        'p_value': p_val
    })

# 결과 출력
res_df = pd.DataFrame(results).sort_values('p_value')
print(res_df)

# CSV 저장
res_df.to_csv('./t_test_results.csv', index=False)
print("✅ t-test 결과를 './t_test_results.csv' 에 저장했습니다.")
