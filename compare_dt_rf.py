import glob
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc, balanced_accuracy_score

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

# 4. 모델 정의
param_dt = {
    'criterion': ['gini', 'entropy'],   # 불순도 측정 기준
    'max_depth': [3, 5, 10, 15],        # 과적합 방지용 깊이 제한
    'min_samples_split': [2, 5, 10],    # 분할 최소 샘플 수 (작을수록 과적합 위험)
    'min_samples_leaf': [1, 2, 5],      # 리프 노드 최소 샘플 수
    'class_weight': [None, 'balanced']  # 클래스 불균형 대응
}

param_rf = {
    'n_estimators': [100, 200],         # 트리 개수 (많을수록 일반화 좋음, 시간 증가)
    'criterion': ['gini', 'entropy'],   # 분할 기준
    'max_depth': [None, 5, 10, 20],     # None은 완전 분기, 숫자는 과적합 방지
    'min_samples_split': [2, 5],        # 내부 노드 최소 샘플 수
    'min_samples_leaf': [1, 2],         # 리프 노드 최소 샘플 수
    'class_weight': [None, 'balanced']  # 클래스 불균형 처리
}

# 5. GridSearch 학습
grid_dt = GridSearchCV(DecisionTreeClassifier(random_state=42), param_dt, cv=3, scoring='f1', n_jobs=-1)
grid_rf = GridSearchCV(RandomForestClassifier(random_state=42), param_rf, cv=3, scoring='f1', n_jobs=-1)

grid_dt.fit(X_train, y_train)
grid_rf.fit(X_train, y_train)

best_dt = grid_dt.best_estimator_
best_rf = grid_rf.best_estimator_

# 6. 예측
y_pred_dt = best_dt.predict(X_test)
y_pred_rf = best_rf.predict(X_test)
y_prob_dt = best_dt.predict_proba(X_test)[:, 1]
y_prob_rf = best_rf.predict_proba(X_test)[:, 1]

# 7. 평가 출력
def report(model_name, estimator, y_true, y_pred):
    print(f"\n[{model_name}]")
    print("Best Params:", estimator.best_params_)
    print("F1-score (CV 평균):", f"{estimator.best_score_:.5f}")
    print("Balanced Accuracy:", f"{balanced_accuracy_score(y_true, y_pred):.5f}")
    print("Confusion Matrix:\n", confusion_matrix(y_true, y_pred))
    print("Classification Report:\n", classification_report(y_true, y_pred, digits=5, zero_division=0))

report("Decision Tree", grid_dt, y_test, y_pred_dt)
report("Random Forest", grid_rf, y_test, y_pred_rf)

# 8. ROC Curve
fpr_dt, tpr_dt, _ = roc_curve(y_test, y_prob_dt)
fpr_rf, tpr_rf, _ = roc_curve(y_test, y_prob_rf)
auc_dt = auc(fpr_dt, tpr_dt)
auc_rf = auc(fpr_rf, tpr_rf)

plt.figure(figsize=(6, 5))
plt.plot(fpr_dt, tpr_dt, label=f'DT AUC = {auc_dt:.2f}', color='darkorange')
plt.plot(fpr_rf, tpr_rf, label=f'RF AUC = {auc_rf:.2f}', color='royalblue')
plt.plot([0, 1], [0, 1], linestyle='--', color='gray')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve: Decision Tree vs Random Forest')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# 9. Feature Importance 비교
features = X_raw.columns
fi_dt = best_dt.feature_importances_
fi_rf = best_rf.feature_importances_
fi_df = pd.DataFrame({
    'Feature': features,
    'Decision Tree': fi_dt,
    'Random Forest': fi_rf
}).set_index('Feature')

fi_df = fi_df[(fi_df > 0).any(axis=1)]
colors = ['darkorange', 'royalblue']
fi_df.plot(kind='barh', figsize=(10, 6), color=colors)
plt.title('Feature Importance Comparison')
plt.xlabel('Importance Score')
plt.tight_layout()
plt.show()

# 10. Train/Test Accuracy 비교
acc_df = pd.DataFrame({
    'Model': ['Decision Tree', 'Random Forest'],
    'Train Accuracy': [best_dt.score(X_train, y_train), best_rf.score(X_train, y_train)],
    'Test Accuracy': [best_dt.score(X_test, y_test), best_rf.score(X_test, y_test)]
})
acc_df.set_index('Model').T.plot(kind='bar', figsize=(6, 4), color=[cm.Blues(0.6), cm.Blues(0.9)])

plt.title('Train/Test Accuracy Comparison')
plt.ylabel('Accuracy')
plt.ylim(0, 1.05)
plt.grid(axis='y')
plt.tight_layout()
plt.show()
