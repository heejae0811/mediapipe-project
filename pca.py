import glob
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import os

# 📂 1. 파일 모으기
files = glob.glob('./features_xlsx/*.xlsx')  # 파일 경로에 맞게 수정
df_list = []

for f in files:
    df = pd.read_excel(f, sheet_name=0)  # 📄 시트1 (Position Metrics)
    df_list.append(df)

df_all = pd.concat(df_list, ignore_index=True)
print(f'✅ 데이터 모양: {df_all.shape}')  # (샘플수, 변수수)

# 📋 2. 숫자형 변수만 선택
X = df_all.select_dtypes(include=['float64', 'int64'])
print(f'✅ PCA 대상 변수 개수: {X.shape[1]}')

# 📏 3. 표준화
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 📉 4. PCA 수행
pca = PCA()
X_pca = pca.fit_transform(X_scaled)

explained_var = pca.explained_variance_ratio_
cum_explained = explained_var.cumsum()

# 📈 5. Scree Plot
plt.figure(figsize=(8,6))
plt.plot(range(1, len(cum_explained)+1), cum_explained, marker='o')
plt.xlabel('Number of Principal Components')
plt.ylabel('Cumulative Explained Variance')
plt.title('Scree Plot - PCA')
plt.grid()
plt.tight_layout()
os.makedirs('./result', exist_ok=True)
plt.savefig('./result/pca_scree_plot.png', dpi=300)
plt.show()

# 📊 6. 결과 요약 출력
for i, (var, cum) in enumerate(zip(explained_var, cum_explained), 1):
    print(f'PC{i}: 설명분산비율={var:.4f}, 누적설명비율={cum:.4f}')

# 🔷 7. 원하는 차원으로 축소 (예: 누적설명비율 ≥ 90%인 차원)
n_components = next(i for i, cum in enumerate(cum_explained) if cum >= 0.9) + 1
print(f'✅ 추천 주성분 개수: {n_components}')

pca_final = PCA(n_components=n_components)
X_pca_reduced = pca_final.fit_transform(X_scaled)

df_pca = pd.DataFrame(X_pca_reduced, columns=[f'PC{i+1}' for i in range(n_components)])
df_pca.index = df_all.index  # 인덱스 맞추기

# 🔷 8. 엑셀로 저장
with pd.ExcelWriter('./result/pca_result.xlsx') as writer:
    df_pca.to_excel(writer, sheet_name='PCA_Reduced', index=False)
    pd.DataFrame({
        'PC': [f'PC{i+1}' for i in range(len(explained_var))],
        'Explained Variance Ratio': explained_var,
        'Cumulative Variance Ratio': cum_explained
    }).to_excel(writer, sheet_name='PCA_Explained_Variance', index=False)

print('🎉 PCA 분석 완료! 결과는 ./result 폴더에 저장되었습니다.')
