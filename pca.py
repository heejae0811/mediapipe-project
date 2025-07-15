import glob
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

# 1️⃣ 파일 불러오기
file_paths = glob.glob('./csv_features/*.csv')
df = pd.concat([pd.read_csv(f) for f in file_paths], ignore_index=True)

# 2️⃣ 피처 선택 (id, label 제외)
features = [col for col in df.columns if col not in ['id', 'label']]
X = df[features]

# 3️⃣ 스케일링 (PCA는 분산 기반이므로 표준화 필요)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 4️⃣ PCA 수행
pca = PCA()  # 모든 주성분 계산
X_pca = pca.fit_transform(X_scaled)

# 5️⃣ 결과를 데이터프레임으로 저장
pca_df = pd.DataFrame(X_pca, columns=[f'PC{i+1}' for i in range(X_pca.shape[1])])
explained_var_ratio = pca.explained_variance_ratio_

# 6️⃣ 주성분 기여율 출력
print("🔷 주성분별 설명된 분산 비율:")
for i, ratio in enumerate(explained_var_ratio):
    print(f"PC{i+1}: {ratio:.4f}")

# 7️⃣ Scree Plot
plt.figure(figsize=(8,5))
plt.plot(range(1, len(explained_var_ratio)+1), explained_var_ratio, marker='o')
plt.xlabel('Principal Component')
plt.ylabel('Explained Variance Ratio')
plt.title('Scree Plot')
plt.grid(True)
plt.show()

# 8️⃣ 데이터 + 주성분을 저장
result = pd.concat([df[['id', 'label']], pca_df], axis=1)
result.to_excel("pca_results.xlsx", index=False)
print("✅ PCA 결과가 'pca_results.xlsx'로 저장되었습니다.")
