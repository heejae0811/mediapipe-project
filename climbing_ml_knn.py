import glob
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV, learning_curve
from sklearn.neighbors import KNeighborsClassifier
from sklearn.inspection import permutation_importance
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc, ConfusionMatrixDisplay, balanced_accuracy_score

# 1. CSV 불러오기
csv_files = glob.glob('./data/features_*.csv')
df = pd.concat([pd.read_csv(file) for file in csv_files], ignore_index=True)
print(f"[정보] 총 데이터 수: {len(df)}개 샘플")

# 2. Feature / Label 분리 및 정규화
X_raw = df.drop(['id', 'label'], axis=1)
y = LabelEncoder().fit_transform(df['label']) # 0 비숙련자 / 1 숙련자
X = StandardScaler().fit_transform(X_raw)

# 3. 학습/테스트 분할
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

# 4. KNN + GridSearchCV
param_grid_knn = {
    'n_neighbors': [3, 5, 7, 9],
    'weights': ['uniform', 'distance'],
    'metric': ['euclidean', 'manhattan']
}

grid_search = GridSearchCV(
    estimator=KNeighborsClassifier(),
    param_grid=param_grid_knn,
    cv=5,
    scoring='f1',
    n_jobs=-1,
    verbose=1
)
grid_search.fit(X_train, y_train)
best_model = grid_search.best_estimator_

# 5. 예측 및 평가
y_pred = best_model.predict(X_test)
y_prob = best_model.predict_proba(X_test)[:, 1]

print('Best Parameters:', grid_search.best_params_)
print('Best f1-score (CV mean):', f"{grid_search.best_score_:.5f}")
print('Confusion Matrix:\n', confusion_matrix(y_test, y_pred))
print('Classification Report:\n', classification_report(y_test, y_pred, digits=5, zero_division=0))
print(f"[KNN] Balanced Accuracy: {balanced_accuracy_score(y_test, y_pred):.5f}")

# 6. Confusion Matrix
ConfusionMatrixDisplay.from_estimator(
    estimator=grid_search.best_estimator_,
    X=X_test,
    y=y_test,
    display_labels=['Beginner', 'Advanced'],
    cmap='Blues',
    values_format='d'
)
plt.title('Confusion Matrix - KNN')
plt.tight_layout()
plt.show()

# 7. ROC Curve
fpr, tpr, _ = roc_curve(y_test, y_prob)
roc_auc = auc(fpr, tpr)

plt.figure()
plt.plot(fpr, tpr, lw=2, label=f'AUC = {roc_auc:.2f}')
plt.plot([0, 1], [0, 1], linestyle='--', color='gray')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve - KNN')
plt.legend(loc='lower right')
plt.grid(True, alpha=0.5)
plt.tight_layout()
plt.show()

# 8. Permutation Importance
result = permutation_importance(
    estimator=grid_search.best_estimator_,
    X=X_test,
    y=y_test,
    scoring='roc_auc',
    n_repeats=30,
    random_state=42,
)

importances = result.importances_mean
stds = result.importances_std
features = X_raw.columns
indices = np.argsort(importances)[::-1]

plt.figure(figsize=(12, 6))
plt.bar(range(len(features)), importances[indices], yerr=stds[indices], align='center')
plt.xticks(range(len(features)), features[indices], rotation=45, ha='right', fontsize=10)
plt.ylabel("Decrease in AUC")
plt.title("Permutation Importance - KNN")
plt.grid(True, alpha=0.5)
plt.tight_layout()
plt.show()

importance_df = pd.DataFrame({'Feature': features[indices], 'Importance': importances[indices]})
print("\nTop 10 Permutation Importance")
print(importance_df.head(10))

# 9. Train/Test Accuracy
train_score = best_model.score(X_train, y_train)
test_score = best_model.score(X_test, y_test)
scores = [train_score, test_score]
labels = ['Train Accuracy', 'Test Accuracy']

print(f"\n[KNN] Train Accuracy: {train_score:.5f}")
print(f"[KNN] Test Accuracy : {test_score:.5f}")

colors = [cm.Blues(0.6), cm.Blues(0.9)]
plt.figure()
plt.bar(labels, scores, color=colors)
plt.ylim(0, 1.1)
plt.ylabel('Accuracy')
plt.title('Accuracy - KNN (Train vs Test)')
plt.grid(axis='y', alpha=0.5)
for i, v in enumerate(scores):
    plt.text(i, v + 0.02, f'{v:.2f}', ha='center')
plt.tight_layout()
plt.show()

# 10. Learning Curve
train_sizes, train_scores, val_scores = learning_curve(
    estimator=best_model,
    X=X_train,
    y=y_train,
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
plt.title('Learning Curve - KNN')
plt.xlabel('Training Set Size')
plt.ylabel('Accuracy')
plt.ylim(0, 1.1)
plt.grid(True, alpha=0.5)
plt.legend(loc='lower right')
plt.tight_layout()
plt.show()

# 11. GridSearch 결과 heatmap
cv_results = pd.DataFrame(grid_search.cv_results_)
pivot = cv_results.pivot_table(
    index='param_n_neighbors',
    columns='param_weights',
    values='mean_test_score'
)

plt.figure()
sns.heatmap(pivot, annot=True, fmt=".3f", cmap="Blues")
plt.title('KNN GridSearchCV F1-score Heatmap')
plt.xlabel('weights')
plt.ylabel('n_neighbors')
plt.tight_layout()
plt.show()
