import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import shapiro, levene, pearsonr, spearmanr

# CSV 불러오기
csv_files = glob.glob('./data/features_climbing*.csv')
df = pd.concat([pd.read_csv(file) for file in csv_files], ignore_index=True)
metrics = df.columns.drop(['id', 'label'])
correlation_matrix = pd.DataFrame(index=metrics, columns=metrics, dtype=float)
results = []

# 정규성, 등분산, 상관관계 분석
for i in range(len(metrics)):
    for j in range(i + 1, len(metrics)):
        var1, var2 = metrics[i], metrics[j]
        data = df[[var1, var2]].dropna()
        if len(data) < 3:
            continue

        x, y = data[var1], data[var2]
        normal = shapiro(x).pvalue > 0.05 and shapiro(y).pvalue > 0.05
        equal_var = levene(x, y).pvalue > 0.05

        if normal and equal_var:
            corr, p_val = pearsonr(x, y)
            method = 'Pearson'
        else:
            corr, p_val = spearmanr(x, y)
            method = 'Spearman'

        correlation_matrix.loc[var1, var2] = corr
        correlation_matrix.loc[var2, var1] = corr

        results.append({
            '변수1': var1,
            '변수2': var2,
            '상관계수': round(corr, 4),
            'p값': round(p_val, 4),
            '사용된 방법': method
        })

# 히트맵 그래프
np.fill_diagonal(correlation_matrix.values, 1.0)
plt.figure(figsize=(14, 10))
sns.heatmap(
    correlation_matrix.astype(float),
    annot=True,
    fmt='.2f',
    cmap='coolwarm',
    square=True,
    cbar=True,
    linewidths=0.5,
    annot_kws={"size": 8}
)
plt.title("Feature Correlation Matrix")
plt.tight_layout()
plt.show()

# 상관계수 0.95 이상만 그래프
results_df = pd.DataFrame(results)
high_corrs = results_df[
    (results_df['상관계수'].abs() >= 0.95) &
    (results_df['변수1'] != results_df['변수2'])
]

plt.figure(figsize=(12, 6))
plt.barh(
    [f"{row['변수1']} ↔ {row['변수2']}" for _, row in high_corrs.iterrows()],
    high_corrs['상관계수'],
)
plt.xlabel("Correlation")
plt.title("Correlation ≥ 0.95")
plt.grid(axis='x')
plt.tight_layout()
plt.show()
