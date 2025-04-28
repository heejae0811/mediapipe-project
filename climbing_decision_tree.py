import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import glob
import os
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc, ConfusionMatrixDisplay

# 1. 여러 CSV 파일 읽어오기
csv_files = glob.glob('./data/features_*.csv')  # 파일명 패턴 지정

df_list = []
for file in csv_files:
    df = pd.read_csv(file)
    df_list.append(df)

# 파일 합치기
df = pd.concat(df_list, ignore_index=True)
print(f"[정보] 총 데이터 수: {len(df)}개 샘플")

# 2. Feature(X)와 Label(y) 분리
X = df.drop(['id', 'label'], axis=1)
y = df['label']

# 3. 레이블 인코딩
le = LabelEncoder()
y = le.fit_transform(y)

# 4. 학습/테스트 데이터 분할
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

# 5. 결정 트리 모델 + 하이퍼파라미터 튜닝
param_grid = {
    'criterion': ['gini', 'entropy'],
    'max_depth': [3, 5, 10, None],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'class_weight': [None, 'balanced']
}

dt = DecisionTreeClassifier(random_state=42)

grid_search = GridSearchCV(
    estimator=dt,
    param_grid=param_grid,
    cv=3,
    scoring='f1',
    n_jobs=-1,
    verbose=1
)

# 6. 학습
grid_search.fit(X_train, y_train)

# 7. 최적 모델로 예측
y_pred = grid_search.best_estimator_.predict(X_test)
y_prob = grid_search.best_estimator_.predict_proba(X_test)[:, 1]

# 8. 성능 출력
print('Best Parameters:', grid_search.best_params_)
print('Best f1-score (CV 평균): ', round(grid_search.best_score_, 5))
print('Confusion Matrix:\n', confusion_matrix(y_test, y_pred))
print('Classification Report:\n', classification_report(y_test, y_pred, digits=5))

# 9. GridSearchCV 결과 요약
cv_results = pd.DataFrame(grid_search.cv_results_)

summary_df = cv_results[[
    'mean_test_score', 'std_test_score', 'rank_test_score',
    'param_criterion', 'param_max_depth',
    'param_min_samples_split', 'param_min_samples_leaf',
    'param_class_weight'
]].sort_values(by='mean_test_score', ascending=False)

print('Top 10 parameter combinations by mean_test_score:')
print(summary_df.head(10).to_string(index=False))

# 10. 결정 트리 시각화
plt.figure(figsize=(8, 6))
plot_tree(
    grid_search.best_estimator_,
    feature_names=X.columns,
    class_names=['0', '1'],
    filled=True,
    rounded=True
)
plt.title('Best Decision Tree')
plt.tight_layout()
plt.show()

# 11. Feature Importance 시각화
importances = grid_search.best_estimator_.feature_importances_
indices = np.argsort(importances)[::-1]

plt.figure(figsize=(14, 8))
plt.title('Feature Importance')
plt.bar(range(X.shape[1]), importances[indices])
plt.xticks(range(X.shape[1]), X.columns[indices], rotation=90)
plt.ylabel('Importance Score')
plt.tight_layout()
plt.show()

importance_df = pd.DataFrame({
    'Feature': X.columns,
    'Importance': importances
}).sort_values(by='Importance', ascending=False)

print(importance_df.head(10))

# 12. ROC Curve
fpr, tpr, thresholds = roc_curve(y_test, y_prob)
roc_auc = auc(fpr, tpr)

plt.figure(figsize=(6, 5))
plt.plot(fpr, tpr, color='blue', lw=2, label=f'ROC Curve (AUC = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color='gray', linestyle='--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.legend(loc='lower right')
plt.grid(True)
plt.tight_layout()
plt.show()

# 13. Confusion Matrix
ConfusionMatrixDisplay.from_estimator(
    grid_search.best_estimator_,
    X_test,
    y_test,
    display_labels=['Class 0', 'Class 1'],
    cmap='Blues',
    values_format='d'
)
plt.title('Confusion Matrix')
plt.tight_layout()
plt.show()
