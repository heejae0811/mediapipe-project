import glob, os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler

# 설정
excel_files = [f for f in glob.glob('./features_xlsx/*.xlsx') if '~$' not in f]
output_dir = './result'
os.makedirs(output_dir, exist_ok=True)

# 관절 인슉률이 좋은  feature들
selected_features = [
    # 'landmark24_min',
    # 'landmark24_max',
    # 'landmark24_mean',
    # 'landmark24_median',
    # 'landmark24_std'
    #
    # 'landmark23_min',
    # 'landmark23_max',
    # 'landmark23_mean',
    # 'landmark23_median',
    # 'landmark23_std'
    #
    # 'landmark12_min',
    # 'landmark12_max',
    # 'landmark12_mean',
    # 'landmark12_median',
    # 'landmark12_std'
    #
    # 'landmark11_min',
    # 'landmark11_max',
    # 'landmark11_mean',
    # 'landmark11_median',
    # 'landmark11_std'
    #
    # 'landmark0_min',
    # 'landmark0_max',
    # 'landmark0_mean',
    # 'landmark0_median',
    # 'landmark0_std'
]

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

# Feature Importance 함수
def compute_feature_importance(df, sheet_name, selected_features=None):
    print(f"\n📋 {sheet_name}")

    if selected_features:
        # 선택한 feature가 실제 데이터프레임에 있는지 확인
        valid_features = [f for f in selected_features if f in df.columns]
        if not valid_features:
            raise ValueError("❌ 선택한 feature가 데이터에 없습니다.")
        X_raw = df[valid_features]
    else:
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
    print(f"✅ 저장 완료: {sheet_name}")

# 실행
compute_feature_importance(df_sheet1, 'Position Feature Importance', selected_features)
compute_feature_importance(df_sheet2, 'Normalized Position Feature Importance', selected_features)

print("\n🎯 모든 작업 완료!")
