import glob
import pandas as pd
from scipy.stats import shapiro, levene, pearsonr, spearmanr
import seaborn as sns
import matplotlib.pyplot as plt

# 파일 불러오기
csv_files = glob.glob('./csv_features/*.csv')
df = pd.concat([pd.read_csv(f) for f in csv_files], ignore_index=True)

# 제외할 변수 (얼굴 관절에서 코만 선택)
exclude_keywords = [
    'left_eye_inner', 'left_eye', 'left_outer', 'right_eye_inner', 'right_eye', 'right_eye_outer',
    'left_ear', 'right_ear', 'mouth_left', 'mouth_right'
]

# 분석할 변수
metrics = [
    col for col in df.columns
    if col not in ['id', 'label'] and
       not any(keyword in col for keyword in exclude_keywords)
]

# 결과 저장용
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

# 결과를 데이터프레임으로 변환
res_df = pd.DataFrame(results)

# 상관행렬 생성 (피어슨 기준)
corr_matrix = df[metrics].corr(method='pearson')

# Heatmap
plt.figure(figsize=(12, 10))
sns.heatmap(corr_matrix, annot=False, cmap='coolwarm')
plt.title('Correlation Heatmap (Pearson)')
plt.tight_layout()
plt.show()

# |r| >= 0.9인 변수 쌍 추출
high_corr = res_df[(res_df['상관계수'].abs() >= 0.9) & (res_df['변수1'] != res_df['변수2'])]
print(f'✅ 총 {len(csv_files)}개의 파일을 분석했습니다.')
print(f'\n✅ |r| >= 0.9인 변수쌍 {len(high_corr)}개 발견')

# 결과를 엑셀로 저장
with pd.ExcelWriter('./result/correlation_analysis.xlsx') as writer:
    res_df.to_excel(writer, sheet_name='상관분석결과', index=False)
    high_corr.to_excel(writer, sheet_name='높은상관(>0.9)', index=False)
    corr_matrix.to_excel(writer, sheet_name='상관행렬')

print('✅ 엑셀 파일 저장 완료')
