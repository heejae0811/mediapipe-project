import glob, os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler

# 파일 불러오기
excel_files = glob.glob('./features_xlsx/*.xlsx')
output_dir = './result'
os.makedirs(output_dir, exist_ok=True)

# 시트별 데이터 결합
sheet1_dfs, sheet2_dfs = [], []

for file_path in excel_files:
    try:
        sheet1_dfs.append(pd.read_excel(file_path, sheet_name=0))
        sheet2_dfs.append(pd.read_excel(file_path, sheet_name=1))
    except Exception as e:
        print(f"⚠️ {file_path} → {e}")

df_sheet1 = pd.concat(sheet1_dfs, ignore_index=True)
df_sheet2 = pd.concat(sheet2_dfs, ignore_index=True)

# Feature Importance 함수
def compute_feature_importance(df, sheet_name):
    print(f"\n{sheet_name}")

    X_raw = df.drop(['id', 'label'], axis=1)
    y = LabelEncoder().fit_transform(df['label'])
    X = StandardScaler().fit_transform(X_raw)

    clf = RandomForestClassifier(random_state=42).fit(X, y)

    importances = clf.feature_importances_
    indices = np.argsort(importances)[::-1]
    features = X_raw.columns[indices]

    df_out = pd.DataFrame({
        'Feature': features,
        'Importance': importances[indices]
    })
    top10_df = df_out.head(10)
    df_out.to_csv(f"{output_dir}/{sheet_name}.csv", index=False)

    plt.figure(figsize=(12, 6))
    plt.bar(range(len(top10_df)), top10_df['Importance'], align='center')
    plt.xticks(range(len(top10_df)), top10_df['Feature'], rotation=45, ha='right', fontsize=8)
    plt.ylabel('Importance')
    plt.title(f'{sheet_name} (Top 10)')
    plt.grid(True, alpha=0.5)
    plt.tight_layout()
    plt.savefig(f"{output_dir}/{sheet_name}.png")
    plt.close()
    print(f"저장 완료: {sheet_name}")

# 함수 실행
compute_feature_importance(df_sheet1, 'Position Feature Importance')
compute_feature_importance(df_sheet2, 'Normalized Position Feature Importance')
