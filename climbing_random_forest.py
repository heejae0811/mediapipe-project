import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.cm as cm
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV, learning_curve
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc, ConfusionMatrixDisplay, balanced_accuracy_score

# 1. CSV 불러오기
csv_files = glob.glob('./data/features_*.csv')
df = pd.concat([pd.read_csv(file) for file in csv_files], ignore_index=True)
print(f"[정보] 총 데이터 수: {len(df)}개 샘플")

# 2. Feature / Label 분리 및 정규화
X_raw = df.drop(['id', 'label'], axis=1)
y = LabelEncoder().fit_transform(df['label'])
X = StandardScaler().fit_transform(X_raw)

# 3. 학습/테스트 분할
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

# 5. Random Forest + GridSearchCV
param_grid_rf = {
    'n_estimators': [100, 200],
    'criterion': ['gini', 'entropy'],
    'max_depth': [None, 5, 10, 20],
    'min_samples_split': [2, 5],
    'min_samples_leaf': [1, 2],
    'class_weight': [None, 'balanced']
}

grid_search = GridSearchCV(
    estimator=RandomForestClassifier(random_state=42),
    param_grid=param_grid_rf,
    cv=3,
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
print('Best f1-score (CV 평균):', f"{grid_search.best_score_:.5f}")
print('Confusion Matrix:\n', confusion_matrix(y_test, y_pred))
print('Classification Report:\n', classification_report(y_test, y_pred, digits=5, zero_division=0))
print(f"[Random Forest] Balanced Accuracy: {balanced_accuracy_score(y_test, y_pred):.5f}")

# 6. Confusion Matrix 시각화
ConfusionMatrixDisplay.from_estimator(
    best_model, X_test, y_test,
    display_labels=['Class 0', 'Class 1'],
    cmap='Blues', values_format='d'
)
plt.title('Random Forest Confusion Matrix')
plt.tight_layout()
plt.show()

# 7. ROC Curve
fpr, tpr, _ = roc_curve(y_test, y_prob)
roc_auc = auc(fpr, tpr)

plt.figure(figsize=(6, 4))
plt.plot(fpr, tpr, color=cm.Blues(0.6), lw=2, label=f'AUC = {roc_auc:.2f}')
plt.plot([0, 1], [0, 1], linestyle='--', color='gray')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Random Forest ROC Curve')
plt.legend(loc='lower right')
plt.grid(True)
plt.tight_layout()
plt.show()

# 8. Feature Importance
importances = best_model.feature_importances_
indices = np.argsort(importances)[::-1]
features = X_raw.columns[indices]

plt.figure(figsize=(10, 5))
plt.title('Random Forest Feature Importance')
plt.bar(range(X.shape[1]), importances[indices], color=cm.Blues(0.6))
plt.xticks(range(X.shape[1]), features, rotation=90)
plt.ylabel('Importance Score')
plt.tight_layout()
plt.show()

importance_df = pd.DataFrame({'Feature': features, 'Importance': importances[indices]})
print("\nTop 10 Features Important:")
print(importance_df.head(10))

# 10. Train/Test Accuracy 시각화
train_score = best_model.score(X_train, y_train)
test_score = best_model.score(X_test, y_test)
scores = [train_score, test_score]
labels = ['Train Accuracy', 'Test Accuracy']

print(f"\n[Random Forest] Train Accuracy: {train_score:.5f}")
print(f"[Random Forest] Test Accuracy : {test_score:.5f}")

colors = [cm.Blues(0.6), cm.Blues(0.9)]
plt.figure(figsize=(4.5, 3.5))
plt.bar(labels, scores, color=colors)
plt.ylim(0, 1.05)
plt.ylabel('Accuracy')
plt.title('Random Forest Accuracy (Train vs Test)')
plt.grid(axis='y')
for i, v in enumerate(scores):
    plt.text(i, v + 0.02, f'{v:.2f}', ha='center')
plt.tight_layout()
plt.show()

# 10. Learning Curve
train_sizes, train_scores, val_scores = learning_curve(
    best_model, X_train, y_train,
    train_sizes=np.linspace(0.1, 1.0, 5),
    cv=5,
    scoring='accuracy',
    shuffle=True,
    random_state=42
)
train_mean = np.mean(train_scores, axis=1)
val_mean = np.mean(val_scores, axis=1)

plt.figure(figsize=(6, 4))
plt.plot(train_sizes, train_mean, 'o--', label='Training Accuracy', color=cm.Blues(0.6))
plt.plot(train_sizes, val_mean, 'o-', label='Validation Accuracy', color=cm.Blues(0.9))
plt.title('Random Forest Learning Curve')
plt.xlabel('Training Set Size')
plt.ylabel('Accuracy')
plt.ylim(0, 1.1)
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()

# 11. GridSearch 결과 heatmap
cv_results = pd.DataFrame(grid_search.cv_results_)
pivot = cv_results.pivot_table(
    index='param_max_depth', columns='param_n_estimators', values='mean_test_score'
)

plt.figure(figsize=(6, 4))
sns.heatmap(pivot, annot=True, fmt=".3f", cmap="Blues")
plt.title('Random Forest GridSearchCV F1-score Heatmap')
plt.xlabel('n_estimators')
plt.ylabel('max_depth')
plt.tight_layout()
plt.show()
