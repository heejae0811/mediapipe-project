import glob
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, f1_score, balanced_accuracy_score, roc_curve, auc
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier

# 1. CSV 불러오기 및 전처리
csv_files = glob.glob('./data/features_*.csv')
df = pd.concat([pd.read_csv(file) for file in csv_files], ignore_index=True)

X_raw = df.drop(['id', 'label'], axis=1)
y = LabelEncoder().fit_transform(df['label'])  # 0 = 비숙련자, 1 = 숙련자
X = StandardScaler().fit_transform(X_raw)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

# 2. 모델 설정
models = {
    "Decision Tree": {
        "estimator": DecisionTreeClassifier(random_state=42),
        "param_grid": {
            'criterion': ['gini', 'entropy'],
            'max_depth': [3, 5, 10, 15],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 5],
            'class_weight': [None, 'balanced']
        }
    },
    "Random Forest": {
        "estimator": RandomForestClassifier(random_state=42),
        "param_grid": {
            'n_estimators': [100, 200],
            'criterion': ['gini', 'entropy'],
            'max_depth': [None, 5, 10, 20],
            'min_samples_split': [2, 5],
            'min_samples_leaf': [1, 2],
            'class_weight': [None, 'balanced']
        }
    },
    "KNN": {
        "estimator": KNeighborsClassifier(),
        "param_grid": {
            'n_neighbors': [3, 5, 7, 9],
            'weights': ['uniform', 'distance'],
            'metric': ['euclidean', 'manhattan']
        }
    }
}

model_order = ["Decision Tree", "Random Forest", "KNN"]
model_colors = {
    "Decision Tree": "#2ca02c",   # Green
    "Random Forest": "#d62728",   # Red
    "KNN": "#1f77b4"              # Blue
}

# 3. 학습 및 평가
results = {
    "Model": [],
    "Accuracy": [],
    "F1-score": [],
    "Balanced Accuracy": [],
    "AUC": []
}
roc_data = {}

for name in model_order:
    config = models[name]
    grid = GridSearchCV(config["estimator"], config["param_grid"], cv=3, scoring='f1', n_jobs=-1)
    grid.fit(X_train, y_train)
    best_model = grid.best_estimator_

    y_pred = best_model.predict(X_test)
    y_prob = best_model.predict_proba(X_test)[:, 1]

    fpr, tpr, _ = roc_curve(y_test, y_prob)
    auc_score = auc(fpr, tpr)
    roc_data[name] = (fpr, tpr, auc_score)

    results["Model"].append(name)
    results["Accuracy"].append(accuracy_score(y_test, y_pred))
    results["F1-score"].append(f1_score(y_test, y_pred))
    results["Balanced Accuracy"].append(balanced_accuracy_score(y_test, y_pred))
    results["AUC"].append(auc_score)

# 4. 결과 정리
df_result = pd.DataFrame(results).set_index("Model").loc[model_order]
print("\n[모델 성능 비교 요약]")
print(df_result.round(5))

# 5. ROC Curve
plt.figure()
for name in model_order:
    fpr, tpr, auc_val = roc_data[name]
    plt.plot(fpr, tpr, lw=2, label=f"{name} (AUC = {auc_val:.2f})", color=model_colors[name])
plt.plot([0, 1], [0, 1], linestyle='--', color='gray', lw=1)
plt.title("ROC Curves", fontsize=14)
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.grid(True, linestyle='--', linewidth=0.5)
plt.legend(loc="lower right")
plt.tight_layout()
plt.show()

# 6. 성능 비교 바 그래프
df_result.plot(
    kind='barh',
    figsize=(9, 5)
)
plt.title("Model Performance Comparison (Higher is Better)", fontsize=14)
plt.xlabel("Score")
plt.grid(axis='x', linestyle='--', linewidth=0.5)
plt.legend(
    bbox_to_anchor=(1.05, 1),  # 범례를 오른쪽 바깥으로 이동
    loc='upper left',
    borderaxespad=0.
)
plt.tight_layout()
plt.show()