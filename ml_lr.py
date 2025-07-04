import glob
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV, learning_curve
from sklearn.metrics import confusion_matrix, classification_report, ConfusionMatrixDisplay, roc_curve, auc, balanced_accuracy_score

# 1. CSV 불러오기
csv_files = glob.glob('./data/features_*.csv')
df = pd.concat([pd.read_csv(file) for file in csv_files], ignore_index=True)
print(f"[정보] 총 데이터 수: {len(df)}개 샘플")

# 2. Feature / Label 분리 및 정규화
X_raw = df.drop(['id', 'label'], axis=1)
X = StandardScaler().fit_transform(X_raw)
y = LabelEncoder().fit_transform(df['label'])

# 3. 학습/테스트 분할
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

# 4. Logistic Regression + GridSearchCV
param_grid_lr = {
    'C': [0.01, 0.1, 1, 10, 100],     # 정규화 강도: 값이 작을수록 규제가 강해 과소적합 가능성이 있음 / 과적합
    'penalty': ['l2'],                # 어떤 방식으로 규제를 적용할지
    'solver': ['liblinear', 'lbfgs']  # liblinear: 작은 데이터 또는 이진 분류 / lbfgs:큰 데이터 또는 다중 클래스
}

grid_search = GridSearchCV(
    estimator=LogisticRegression(max_iter=1000, random_state=42), # max_iter: 모델이 수렴할 때까지 반복할 최대 횟수
    param_grid=param_grid_lr,
    cv=5,
    scoring='f1',
    n_jobs=-1,
    verbose=1,
    return_train_score=True
)
grid_search.fit(X_train, y_train)
best_model = grid_search.best_estimator_

# 5. 예측 및 평가
y_pred = best_model.predict(X_test)
y_proba = best_model.predict_proba(X_test)[:, 1]

print('Best Parameters:', grid_search.best_params_)
print('Best f1-score (CV mean):', f"{grid_search.best_score_:.5f}")
print('Confusion Matrix:\n', confusion_matrix(y_test, y_pred))
print('Classification Report:\n', classification_report(y_test, y_pred, digits=5, zero_division=0))
print(f"[Logistic Regression] Balanced Accuracy: {balanced_accuracy_score(y_test, y_pred):.5f}")

# 6. Confusion Matrix
ConfusionMatrixDisplay.from_estimator(
    estimator=best_model,
    X=X_test,
    y=y_test,
    display_labels=['Beginner', 'Advanced'],
    cmap='Blues',
    values_format='d'
)
plt.title('Confusion Matrix - Logistic Regression')
plt.tight_layout()
plt.show()

# 7. ROC Curve
fpr, tpr, _ = roc_curve(y_test, y_proba)
roc_auc = auc(fpr, tpr)

plt.figure()
plt.plot(fpr, tpr, lw=2, label=f'AUC = {roc_auc:.2f}')
plt.plot([0, 1], [0, 1], linestyle='--', color='gray')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve - Logistic Regression')
plt.legend(loc='lower right')
plt.grid(True, alpha=0.5)
plt.tight_layout()
plt.show()

# 8. Train/Test Accuracy
train_score = best_model.score(X_train, y_train)
test_score = best_model.score(X_test, y_test)
scores = [train_score, test_score]
labels = ['Train Accuracy', 'Test Accuracy']

print(f"\n[Logistic Regression] Train Accuracy: {train_score:.5f}")
print(f"[Logistic Regression] Test Accuracy : {test_score:.5f}")

colors = [cm.Blues(0.6), cm.Blues(0.9)]
plt.figure()
plt.bar(labels, scores, color=colors)
plt.ylim(0, 1.1)
plt.ylabel('Accuracy')
plt.title('Accuracy - Logistic Regression (Train vs Test)')
plt.grid(axis='y', alpha=0.5)
for i, v in enumerate(scores):
    plt.text(i, v + 0.02, f'{v:.2f}', ha='center')
plt.tight_layout()
plt.show()

# 9. Learning Curve
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
plt.title('Learning Curve - Logistic Regression')
plt.xlabel('Training Set Size')
plt.ylabel('Accuracy')
plt.ylim(0, 1.1)
plt.grid(True, alpha=0.5)
plt.legend(loc='lower right')
plt.tight_layout()
plt.show()

# 10. GridSearchCV 결과 heatmap
cv_results = pd.DataFrame(grid_search.cv_results_)
pivot = cv_results.pivot_table(
    index='param_C',
    columns='param_solver',
    values='mean_test_score'
)

plt.figure()
sns.heatmap(pivot, annot=True, fmt=".3f", cmap="Blues")
plt.title('Logistic Regression GridSearchCV Heatmap')
plt.xlabel('solver')
plt.ylabel('C')
plt.tight_layout()
plt.show()
