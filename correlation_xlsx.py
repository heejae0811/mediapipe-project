import glob
import pandas as pd
from scipy.stats import shapiro, levene, pearsonr, spearmanr
import seaborn as sns
import matplotlib.pyplot as plt
import os

# 결과 저장 폴더
os.makedirs('./result', exist_ok=True)

# 제외할 변수
exclude_keywords = [
]

def correlation_analysis(df, sheet_name):
    # 분석할 변수
    metrics = [
        col for col in df.columns
        if col not in ['id', 'label'] and
           not any(keyword in col for keyword in exclude_keywords)
    ]

    results = []

    # 정규성, 등분산, 상관계수 계산
    for i in range(len(metrics)):
        for j in range(i + 1, len(metrics)):
            var1, var2 = metrics[i], metrics[j]
            data = df[[var1, var2]].dropna()
            if len(data) < 3:
                continue

            x, y = data[var1], data[var2]

            # 정규성 검정
            p_norm_x = shapiro(x).pvalue
            p_norm_y = shapiro(y).pvalue
            normal = p_norm_x > 0.05 and p_norm_y > 0.05

            # 등분산 검정
            p_levene = levene(x, y).pvalue
            equal_var = p_levene > 0.05

            # 상관계수 선택
            if normal:
                corr, p_val = pearsonr(x, y)
                method = 'Pearson'
            else:
                corr, p_val = spearmanr(x, y)
                method = 'Spearman'

            results.append({
                '변수1': var1,
                '변수2': var2,
                '상관계수': round(corr, 4),
                'p값': round(p_val, 4),
                '사용된 방법': method,
                'var1_정규성_p': round(p_norm_x, 4),
                'var2_정규성_p': round(p_norm_y, 4),
                '등분산성_p': round(p_levene, 4)
            })

    res_df = pd.DataFrame(results)

    # 상관행렬 생성 (피어슨 기준)
    corr_matrix = df[metrics].corr(method='pearson')

    # |r| >= 0.9인 변수 쌍 추출
    high_corr = res_df[(res_df['상관계수'].abs() >= 0.95) & (res_df['변수1'] != res_df['변수2'])]

    # Heatmap
    plt.figure(figsize=(12, 10))
    sns.heatmap(corr_matrix, annot=False, cmap='coolwarm')
    plt.title(f'Correlation Heatmap (Pearson) - {sheet_name}')
    plt.tight_layout()
    plt.savefig(f'./result/{sheet_name}_correlation_heatmap.png')
    plt.close()

    return res_df, high_corr, corr_matrix

# 파일 불러오기
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

# 분석
res1, high_corr1, corr_matrix1 = correlation_analysis(df_sheet1, 'Sheet1')
res2, high_corr2, corr_matrix2 = correlation_analysis(df_sheet2, 'Sheet2')

# 엑셀 저장
with pd.ExcelWriter('./result/features_correlation.xlsx') as writer:
    res1.to_excel(writer, sheet_name='position_correlation', index=False)
    high_corr1.to_excel(writer, sheet_name='position_correlation(>0.95)', index=False)
    corr_matrix1.to_excel(writer, sheet_name='position_correlation')

    res2.to_excel(writer, sheet_name='normalized_position_correlation', index=False)
    high_corr2.to_excel(writer, sheet_name='normalized_position_correlation(>0.95)', index=False)
    corr_matrix2.to_excel(writer, sheet_name='normalized_position_correlation')

print(f'🎉 총 {len(xlsx_files)}개의 파일을 분석했고, 시트1/시트2 결과를 각각 저장했습니다.')
