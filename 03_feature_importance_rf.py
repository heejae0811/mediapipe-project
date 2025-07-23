import glob, os
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import StratifiedKFold


# 파일 설정
excel_files = [f for f in glob.glob('./features_xlsx/*.xlsx') if '~$' not in f]
output_dir = './result'
os.makedirs(output_dir, exist_ok=True)

# 시트별 데이터 결합
sheet1_dfs, sheet2_dfs = [], []

for file_path in excel_files:
    try:
        sheet1_dfs.append(pd.read_excel(file_path, sheet_name=0, engine='openpyxl'))
        sheet2_dfs.append(pd.read_excel(file_path, sheet_name=1, engine='openpyxl'))
    except Exception as e:
        print(f"⚠️ {file_path} → {e}")

df_sheet1 = pd.concat(sheet1_dfs, ignore_index=True)
df_sheet2 = pd.concat(sheet2_dfs, ignore_index=True)

# Feature Selection 함수 (CV 기반)
def feature_selection_cv(df, sheet_name, top_n=10, n_splits=5):
    print(f"\n{sheet_name} - Cross-Validation 기반 Feature Selection")

    X_raw = df.drop(columns=['id', 'label'])
    y = LabelEncoder().fit_transform(df['label'])
    feature_names = X_raw.columns.tolist()

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_raw)

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    importances_accum = np.zeros(len(feature_names))

    for fold, (train_idx, val_idx) in enumerate(skf.split(X_scaled, y), 1):
        X_train, y_train = X_scaled[train_idx], y[train_idx]
        clf = RandomForestClassifier(random_state=42)
        clf.fit(X_train, y_train)
        importances_accum += clf.feature_importances_
        print(f"Fold {fold} 완료")

    importances_mean = importances_accum / n_splits

    df_importance = pd.DataFrame({
        'Feature': feature_names,
        'Importance': importances_mean
    }).sort_values(by='Importance', ascending=False)

    top_features = df_importance.head(top_n)['Feature'].tolist()
    print(f"\n✅ 선택된 상위 {top_n} feature:\n", top_features)

    # 저장
    df_importance.to_csv(f"{output_dir}/{sheet_name}_importance.csv", index=False)

    plt.figure(figsize=(8, 5))
    sns.barplot(x='Importance', y='Feature', data=df_importance.head(top_n))
    plt.title(f'{sheet_name} - Top {top_n} Features (CV 기반)')
    plt.tight_layout()
    plt.savefig(f"{output_dir}/{sheet_name}_top{top_n}.png")
    plt.close()

    print(f"{sheet_name} 결과 저장 완료")

# 실행
feature_selection_cv(df_sheet1, 'Position Feature Importance')
feature_selection_cv(df_sheet2, 'Normalized Position Feature Importance')

print("\n✅ 모든 작업 완료!")
