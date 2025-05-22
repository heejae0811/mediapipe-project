import glob
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV, learning_curve
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc,ConfusionMatrixDisplay, balanced_accuracy_score

# 1. CSV 불러오기
csv_files = glob.glob('./data/features_*.csv')
df = pd.concat([pd.read_csv(file) for file in csv_files], ignore_index=True)
print(f"[정보] 총 데이터 수: {len(df)}개 샘플")

# 2. Feature / Label 분리 및 정규화
X_raw = df.drop(['id', 'label'], axis=1)
X = StandardScaler().fit_transform(X_raw)
y = LabelEncoder().fit_transform(df['label']) # 0: 비숙련자 / 1: 숙련자

# 3. 학습/테스트 분할
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2,
    stratify=y,     # 각 클래스의 비율을 동일하게 유지하기 위한 속성값
    random_state=42 # 데이터를 무작위로 섞을 때 사용하는 난수값
)

# 4. Decision Tree + GridSearchCV
param_grid_dt = {
    'criterion': ['gini', 'entropy'],   # 트리를 분할할 때 사용하는 기준: gini - 클래스가 얼마나 균등하게 섞여있는지 측정(분순도) / entropy - 데이터를 나누었을 때 얼마나 정보를 많이 줄 수 있는가
    'max_depth': [3, 5, 10, 15],        # 트리의 최대 깊이 제한
    'min_samples_split': [2, 5, 10],    # 노드를 나누기 위한 최소 샘플 수
    'min_samples_leaf': [1, 2, 5],      # 노드 끝의 최소 샘플 수
    'class_weight': [None, 'balanced']  # 불균형한 클래스의 가중치 조절
}

grid_search = GridSearchCV(
    estimator=DecisionTreeClassifier(random_state=42),
    param_grid=param_grid_dt,   # 실험할 하이퍼파라미터 조합
    cv=5,                       # 5-fold 교차 검증: 데이터를 5개로 나눠 학습-검증 5번 반복
    scoring='f1',               # 모델 성능 평가 기준: F1-score (정밀도 + 재현율의 조화 평균)
    n_jobs=-1,                  # 가능한 모든 CPU 코어를 사용해서 병렬로 계산
    verbose=1,                  # 학습 중 로그 출력
    return_train_score=True     # 파라미터 조합을 시도하면서 훈련/검증 점수 둘 다 기록
)
grid_search.fit(X_train, y_train)
best_model = grid_search.best_estimator_

# 5. 예측 및 평가
y_pred = best_model.predict(X_test)             # X_test에 대한 최종 예측 결과 (0 또는 1로 분류만 함)
y_prob = best_model.predict_proba(X_test)[:, 1] # X_test에 대해 클래스 1일 확률 (예측 확률값)

print('Best Parameters:', grid_search.best_params_)
print('Best f1-score (CV mean):', f"{grid_search.best_score_:.5f}")
print('Confusion Matrix:\n', confusion_matrix(y_test, y_pred))
print('Classification Report:\n', classification_report(y_test, y_pred, digits=5, zero_division=0))
print(f"[Decision Tree] Balanced Accuracy: {balanced_accuracy_score(y_test, y_pred):.5f}")

# 6. Best Decision Tree Structure
plt.figure()
plot_tree(
    best_model,
    feature_names=X_raw.columns,
    class_names=['Beginner', 'Advanced'],
    filled=True,
    rounded=True
)
plt.title(f"Best Decision Tree Structure (Max Depth: {best_model.get_depth()})")
plt.tight_layout()
plt.show()

# 7. Confusion Matrix
ConfusionMatrixDisplay.from_estimator(
    estimator=best_model,
    X=X_test,
    y=y_test,
    display_labels=['Beginner', 'Advanced'],
    cmap='Blues',
    values_format='d'
)
plt.title('Confusion Matrix - Decision Tree')
plt.tight_layout()
plt.show()

# 8. ROC Curve
fpr, tpr, _ = roc_curve(y_test, y_prob)
roc_auc = auc(fpr, tpr)

plt.figure()
plt.plot(fpr, tpr, lw=2, label=f'AUC = {roc_auc:.2f}')
plt.plot([0, 1], [0, 1], linestyle='--', color='gray')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve - Decision Tree')
plt.legend(loc='lower right')
plt.grid(True, alpha=0.5)
plt.tight_layout()
plt.show()

# 9. Feature Importance
importances = best_model.feature_importances_
indices = np.argsort(importances)[::-1]
features = X_raw.columns[indices]

plt.figure(figsize=(12, 6))
plt.bar(range(len(features)), importances[indices], align='center')
plt.xticks(range(len(features)), features, rotation=45, ha='right', fontsize=10)
plt.ylabel('Importance Score')
plt.title('Feature Importance - Decision Tree')
plt.grid(True, alpha=0.5)
plt.tight_layout()
plt.show()

importance_df = pd.DataFrame({'Feature': features, 'Importance': importances[indices]})
print("\nTop 10 Features Important")
print(importance_df.head(10))

# 10. Train/Test Accuracy
train_score = best_model.score(X_train, y_train)
test_score = best_model.score(X_test, y_test)
scores = [train_score, test_score]
labels = ['Train Accuracy', 'Test Accuracy']

print(f"\n[Decision Tree] Train Accuracy: {train_score:.5f}")
print(f"[Decision Tree] Test Accuracy : {test_score:.5f}")

colors = [cm.Blues(0.6), cm.Blues(0.9)]
plt.figure()
plt.bar(labels, scores, color=colors)
plt.ylim(0, 1.1)
plt.ylabel('Accuracy')
plt.title('Accuracy - Decision Tree (Train vs Test)')
plt.grid(axis='y', alpha=0.5)
for i, v in enumerate(scores):
    plt.text(i, v + 0.02, f'{v:.2f}', ha='center')
plt.tight_layout()
plt.show()

# 11. Learning Curve
train_sizes, train_scores, val_scores = learning_curve(
    estimator=best_model,
    X=X_test,
    y=y_test,
    train_sizes=np.linspace(0.1, 1.0, 5),
    cv=5,
    scoring='accuracy',
    shuffle=True,
    random_state=42
)
train_mean = np.mean(train_scores, axis=1)
val_mean = np.mean(val_scores, axis=1)

plt.figure()
plt.plot(train_sizes, train_mean, 'o--', label='Training Accuracy')
plt.plot(train_sizes, val_mean, 'o-', label='Validation Accuracy')
plt.title('Learning Curve - Decision Tree')
plt.xlabel('Training Set Size')
plt.ylabel('Accuracy')
plt.ylim(0, 1.1)
plt.grid(True)
plt.legend(loc='lower right')
plt.tight_layout()
plt.show()

# 12. GridSearch 결과 heatmap
cv_results = pd.DataFrame(grid_search.cv_results_)
pivot = cv_results.pivot_table(
    index='param_max_depth',
    columns='param_min_samples_split',
    values='mean_test_score'
)

plt.figure()
sns.heatmap(pivot, annot=True, fmt=".3f", cmap="Blues")
plt.title('Decision Tree GridSearchCV F1-score Heatmap')
plt.xlabel('min_samples_split')
plt.ylabel('max_depth')
plt.tight_layout()
plt.show()