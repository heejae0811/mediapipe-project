import os, glob
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import ConfusionMatrixDisplay, classification_report, confusion_matrix, f1_score, matthews_corrcoef, precision_score, recall_score, roc_auc_score, roc_curve, balanced_accuracy_score, auc
from sklearn.model_selection import GridSearchCV, learning_curve, train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier
from catboost import CatBoostClassifier

# 설정
os.makedirs('./result_ml', exist_ok=True)
RANDOM_STATE = 42

# 데이터 로딩 및 전처리
def load_and_split_data():
    csv_files = glob.glob('./features_xlsx/*.xlsx')
    print(f"분석할 파일 수: {len(csv_files)}개")

    df_all = pd.concat([pd.read_excel(file, sheet_name=2) for file in csv_files], ignore_index=True)
    y_all = LabelEncoder().fit_transform(df_all['label'])

    class_0 = np.sum(y_all == 0)
    class_1 = np.sum(y_all == 1)
    print(f"라벨 분포 - 0: {class_0}개, 1: {class_1}개")

    feature_cols = df_all.select_dtypes(include=['float64', 'int64']).columns.drop('label')
    X_raw_all = df_all[feature_cols]
    X_scaled_all = StandardScaler().fit_transform(X_raw_all)

    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled_all, y_all, test_size=0.2, stratify=y_all, random_state=RANDOM_STATE
    )

    raw_train = X_raw_all.iloc[X_train.shape[0] * -1:]
    raw_test = X_raw_all.iloc[:X_test.shape[0]]

    return df_all, X_train, X_test, y_train, y_test, raw_train, raw_test, feature_cols

# 랜덤 포레스트 변수 선택
def select_top_features_by_rf(X_train, y_train, feature_cols, top_n=10):
    rf = RandomForestClassifier(n_estimators=100, random_state=RANDOM_STATE)
    rf.fit(X_train, y_train)
    importances = rf.feature_importances_
    importance_df = pd.DataFrame({'feature': feature_cols, 'importance': importances})
    top_features = importance_df.sort_values(by='importance', ascending=False).head(top_n)['feature'].tolist()

    plt.figure(figsize=(10, 6))
    sns.barplot(x='importance', y='feature', data=importance_df.sort_values(by='importance', ascending=False).head(top_n))
    plt.title(f'Random Forest Top {top_n} Features')
    plt.tight_layout()
    plt.show()

    return top_features

# 모델 정의
def get_models():
    return {
        'Logistic Regression': {
            'estimator': LogisticRegression(max_iter=1000, random_state=RANDOM_STATE),
            'param_grid': {
                'C': [0.1, 1],
                'penalty': ['l2'],
                'solver': ['liblinear']
            }
        },
        'KNN': {
            'estimator': KNeighborsClassifier(),
            'param_grid': {
                'n_neighbors': [3, 5, 7],
                'weights': ['uniform']
            }
        },
        'SVM': {
            'estimator': SVC(probability=True, random_state=RANDOM_STATE),
            'param_grid': {
                'C': [1],
                'kernel': ['rbf'],
                'gamma': ['scale']
            }
        },
        'Decision Tree': {
            'estimator': DecisionTreeClassifier(random_state=RANDOM_STATE),
            'param_grid': {
                'max_depth': [3, 5],
                'min_samples_split': [2],
                'min_samples_leaf': [1, 2],
                'criterion': ['gini']
            }
        },
        'Random Forest': {
            'estimator': RandomForestClassifier(random_state=RANDOM_STATE),
            'param_grid': {
                'n_estimators': [50, 100],
                'max_depth': [3, 5],
                'min_samples_leaf': [1, 2],
                'criterion': ['gini']
            }
        },
        'LightGBM': {
            'estimator': LGBMClassifier(random_state=RANDOM_STATE),
            'param_grid': {
                'n_estimators': [50, 100],
                'max_depth': [3, 5],
                'learning_rate': [0.1]
            }
        },
        'XGBoost': {
            'estimator': XGBClassifier(random_state=RANDOM_STATE, eval_metric='logloss'),
            'param_grid': {
                'n_estimators': [50, 100],
                'max_depth': [3],
                'learning_rate': [0.1]
            }
        },
        'CatBoost': {
            'estimator': CatBoostClassifier(random_state=RANDOM_STATE, verbose=0),
            'param_grid': {
                'depth': [3, 5],
                'iterations': [100],
                'learning_rate': [0.1]
            }
        }
    }

# 성능 평가 지표 계산
def compute_metrics(y_true, y_pred, y_proba):
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()
    return {
        'Accuracy': (tn + tp) / (tn + fp + fn + tp),
        'Precision': precision_score(y_true, y_pred),
        'Recall': recall_score(y_true, y_pred),
        'F1': f1_score(y_true, y_pred),
        'Balanced_Accuracy': balanced_accuracy_score(y_true, y_pred),
        'Specificity': tn / (tn + fp),
        'Sensitivity': tp / (tp + fn),
        'MCC': matthews_corrcoef(y_true, y_pred),
        'AUC': roc_auc_score(y_true, y_proba) if y_proba is not None else None
    }

# 시각화 함수들
def plot_confusion_matrix(y_true, y_pred, model_name):
    ConfusionMatrixDisplay.from_predictions(y_true, y_pred, cmap='Blues')
    plt.title(f'Confusion Matrix - {model_name}')
    plt.tight_layout()
    plt.show()

def plot_roc_curve(y_true, y_proba, model_name):
    fpr, tpr, _ = roc_curve(y_true, y_proba)
    plt.plot(fpr, tpr, label=f'AUC = {auc(fpr, tpr):.2f}')
    plt.plot([0, 1], [0, 1], 'k--')
    plt.title(f'ROC Curve - {model_name}')
    plt.xlabel('FPR')
    plt.ylabel('TPR')
    plt.legend()
    plt.tight_layout()
    plt.show()

def plot_learning_curve(estimator, X, y, model_name):
    sizes, train_scores, val_scores = learning_curve(estimator, X, y, cv=5, scoring='f1')
    plt.plot(sizes, train_scores.mean(axis=1), label='Train')
    plt.plot(sizes, val_scores.mean(axis=1), label='Validation')
    plt.title(f'Learning Curve - {model_name}')
    plt.legend()
    plt.tight_layout()
    plt.show()

def plot_feature_importance(model, X, y, feature_names, model_name):
    if hasattr(model, 'feature_importances_'):
        imp = model.feature_importances_
        sorted_idx = np.argsort(imp)[::-1]
        sns.barplot(x=imp[sorted_idx][:10], y=np.array(feature_names)[sorted_idx][:10])
        plt.title(f'Feature Importance - {model_name}')
        plt.tight_layout()
        plt.show()

    result = permutation_importance(model, X, y, n_repeats=10, random_state=RANDOM_STATE)
    idx = result.importances_mean.argsort()[::-1]
    sns.barplot(x=result.importances_mean[idx][:10], y=np.array(feature_names)[idx][:10])
    plt.title(f'Permutation Importance - {model_name}')
    plt.tight_layout()
    plt.show()

# 모델 실행
def run_model(model_name, model_info, X_train, X_test, y_train, y_test, X_all, y_all, feature_names):
    estimator = model_info['estimator']
    param_grid = model_info['param_grid']

    grid = GridSearchCV(estimator, param_grid, cv=5, scoring='f1', n_jobs=-1)
    grid.fit(X_train, y_train)
    best_model = grid.best_estimator_

    y_pred = best_model.predict(X_test)
    y_proba = best_model.predict_proba(X_test)[:, 1] if hasattr(best_model, 'predict_proba') else None

    metrics = compute_metrics(y_test, y_pred, y_proba)
    metrics['Model'] = model_name
    metrics['Best Params'] = str(best_model.get_params())

    print(f"\n========== {model_name} ==========")
    print(classification_report(y_test, y_pred))

    # 시각화
    plot_confusion_matrix(y_test, y_pred, model_name)
    if y_proba is not None:
        plot_roc_curve(y_test, y_proba, model_name)
    plot_learning_curve(best_model, X_all, y_all, model_name)
    plot_feature_importance(best_model, X_train, y_train, feature_names, model_name)

    return metrics

# 실행
if __name__ == '__main__':
    df_all, X_train_full, X_test_full, y_train, y_test, raw_train, raw_test, feature_cols = load_and_split_data()
    top_features = select_top_features_by_rf(X_train_full, y_train, feature_cols, top_n=10)

    # 변수 재선택 후 스케일 재적용
    X_train = StandardScaler().fit_transform(raw_train[top_features])
    X_test = StandardScaler().fit_transform(raw_test[top_features])
    X_all = np.concatenate([X_train, X_test], axis=0)
    y_all = np.concatenate([y_train, y_test], axis=0)

    models = get_models()
    results = []
    for name, info in models.items():
        metrics = run_model(name, info, X_train, X_test, y_train, y_test, X_all, y_all, top_features)
        results.append(metrics)

    results_df = pd.DataFrame(results)
    results_df.sort_values(by='F1', ascending=False, inplace=True)
    print("\n전체 모델 성능 요약:")
    print(results_df[['Model', 'Accuracy', 'Precision', 'Recall', 'F1', 'AUC']].to_string(index=False))

    results_df.to_excel('./result/model_comparison.xlsx', index=False)
    print("\n✅ './result/model_comparison.xlsx' 에 결과 저장 완료.")
