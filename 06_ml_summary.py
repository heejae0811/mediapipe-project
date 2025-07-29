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

# 자동 변수 선택 함수
def select_top_features_by_rf(df, target_col='label', top_n=10):
    feature_cols = df.select_dtypes(include=['float64', 'int64']).columns.drop(target_col)
    X_raw = df[feature_cols]
    y = LabelEncoder().fit_transform(df[target_col])
    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf.fit(X_raw, y)

    importance_df = pd.DataFrame({
        'feature': feature_cols,
        'importance': rf.feature_importances_
    }).sort_values(by='importance', ascending=False)

    # 시각화
    plt.figure(figsize=(10, 6))
    sns.barplot(data=importance_df.head(top_n), x='importance', y='feature')
    plt.title(f'Random Forest Top {top_n} Features')
    plt.tight_layout()
    plt.show()

    return importance_df['feature'].head(top_n).tolist()

# 데이터 전처리 함수
def data_processing(selected_features):
    csv_files = glob.glob('./features_xlsx/*.xlsx')
    df = pd.concat([pd.read_excel(file, sheet_name=1) for file in csv_files], ignore_index=True)
    print(f'[정보] 총 데이터 수: {len(df)}개 샘플')

    X_raw = df[selected_features]
    X = StandardScaler().fit_transform(X_raw)
    y = LabelEncoder().fit_transform(df['label'])

    return df, X_raw, X, y

# 모델 정의
def get_models():
    return {
        'Logistic Regression': {
            'estimator': LogisticRegression(max_iter=1000, random_state=42),
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
            'estimator': SVC(probability=True, random_state=42),
            'param_grid': {
                'C': [1],
                'kernel': ['rbf'],
                'gamma': ['scale']
            }
        },
        'Decision Tree': {
            'estimator': DecisionTreeClassifier(random_state=42),
            'param_grid': {
                'max_depth': [3, 5],
                'min_samples_split': [2],
                'min_samples_leaf': [1, 2],
                'criterion': ['gini']
            }
        },
        'Random Forest': {
            'estimator': RandomForestClassifier(random_state=42),
            'param_grid': {
                'n_estimators': [50, 100],
                'max_depth': [3, 5],
                'min_samples_leaf': [1, 2],
                'criterion': ['gini']
            }
        },
        'LightGBM': {
            'estimator': LGBMClassifier(random_state=42),
            'param_grid': {
                'n_estimators': [50, 100],
                'max_depth': [3, 5],
                'learning_rate': [0.1]
            }
        },
        'XGBoost': {
            'estimator': XGBClassifier(random_state=42, eval_metric='logloss'),
            'param_grid': {
                'n_estimators': [50, 100],
                'max_depth': [3],
                'learning_rate': [0.1]
            }
        },
        'CatBoost': {
            'estimator': CatBoostClassifier(random_state=42, verbose=0),
            'param_grid': {
                'depth': [3, 5],
                'iterations': [100],
                'learning_rate': [0.1]
            }
        }
    }

# 지표 계산
def compute_metrics(y_true, y_pred, y_proba):
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()
    return {
        'Accuracy': (tn+tp)/(tn+fp+fn+tp),
        'Precision': precision_score(y_true, y_pred),
        'Recall': recall_score(y_true, y_pred),
        'F1': f1_score(y_true, y_pred),
        'Balanced_Accuracy': balanced_accuracy_score(y_true, y_pred),
        'Specificity': tn/(tn+fp),
        'Sensitivity': tp/(tp+fn),
        'MCC': matthews_corrcoef(y_true, y_pred),
        'AUC': roc_auc_score(y_true, y_proba) if y_proba is not None else None
    }

# 시각화
def plot_confusion_matrix(y_true, y_pred, model_name):
    plt.figure(figsize=(5, 4))
    cm_display = ConfusionMatrixDisplay.from_predictions(y_true, y_pred, cmap='Blues')
    plt.title(f'Confusion Matrix - {model_name}')
    plt.tight_layout()
    plt.close()

def plot_roc_curve(y_true, y_proba, model_name):
    fpr, tpr, _ = roc_curve(y_true, y_proba)
    auc_score = auc(fpr, tpr)
    plt.figure(figsize=(5, 4))
    plt.plot(fpr, tpr, label=f'AUC = {auc_score:.2f}')
    plt.plot([0, 1], [0, 1], 'k--')
    plt.title(f'ROC Curve - {model_name}')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.legend()
    plt.tight_layout()
    plt.close()

def plot_learning_curve(estimator, X, y, model_name):
    train_sizes, train_scores, test_scores = learning_curve(
        estimator, X, y, cv=5, scoring='f1', n_jobs=-1, train_sizes=np.linspace(0.1, 1.0, 5)
    )
    plt.figure(figsize=(6, 4))
    plt.plot(train_sizes, train_scores.mean(axis=1), label='Train')
    plt.plot(train_sizes, test_scores.mean(axis=1), label='CV')
    plt.title(f'Learning Curve - {model_name}')
    plt.xlabel('Training Size')
    plt.ylabel('F1 Score')
    plt.legend()
    plt.tight_layout()
    plt.close()

def plot_feature_importance(model, X_raw, y, feature_names, model_name):
    if hasattr(model, 'feature_importances_'):
        imp = model.feature_importances_
        idx = np.argsort(imp)[::-1]
        plt.figure(figsize=(8, 5))
        sns.barplot(x=imp[idx][:10], y=np.array(feature_names)[idx][:10])
        plt.title(f'Feature Importance - {model_name}')
        plt.tight_layout()
        plt.close()

    # Permutation importance
    result = permutation_importance(model, X_raw, y, n_repeats=10, random_state=42)
    idx = result.importances_mean.argsort()[::-1]
    plt.figure(figsize=(8, 5))
    sns.barplot(x=result.importances_mean[idx][:10], y=np.array(feature_names)[idx][:10])
    plt.title(f'Permutation Importance - {model_name}')
    plt.tight_layout()
    plt.close()

def plot_accuracy_bar(results_df):
    plt.figure(figsize=(8, 5))
    sns.barplot(data=results_df.sort_values('Accuracy', ascending=False), x='Accuracy', y='Model')
    plt.title('Model Accuracy Comparison')
    plt.tight_layout()
    plt.close()

# 모델 실행
def run_model(model_name, model_info, X_train, X_test, y_train, y_test, X_raw):
    estimator = model_info['estimator']
    param_grid = model_info['param_grid']

    if param_grid:
        grid = GridSearchCV(estimator, param_grid, cv=5, scoring='f1', n_jobs=-1)
        grid.fit(X_train, y_train)
        best_model = grid.best_estimator_
    else:
        best_model = estimator.fit(X_train, y_train)

    y_pred = best_model.predict(X_test)
    y_proba = best_model.predict_proba(X_test)[:,1] if hasattr(best_model, 'predict_proba') else None

    metrics = compute_metrics(y_test, y_pred, y_proba)
    metrics['Model'] = model_name
    metrics['Best Params'] = str(best_model.get_params())

    print(f'\n========== {model_name} ==========')
    print(classification_report(y_test, y_pred))

    return metrics, best_model, y_pred, y_proba

# 함수 실행
if __name__ == '__main__':
    # 전체 데이터프레임 로딩
    csv_files = glob.glob('./features_xlsx/*.xlsx')
    df_all = pd.concat([pd.read_excel(file, sheet_name=1) for file in csv_files], ignore_index=True)

    # 변수 자동 선택
    selected_features = select_top_features_by_rf(df_all, target_col='label', top_n=10)
    print(f"\n 랜덤 포레스트 자동 선택된 변수 목록:\n{selected_features}")

    # 데이터 전처리
    df, X_raw, X, y = data_processing(selected_features)

    # 학습/테스트 분할
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

    # 모델 학습 및 평가
    models = get_models()
    results = []
    for model_name, model_info in models.items():
        metrics, best_model, y_pred, y_proba = run_model(model_name, model_info, X_train, X_test, y_train, y_test, X_raw)
        results.append(metrics)

        # 시각화
        plot_confusion_matrix(y_test, y_pred, model_name)
        if y_proba is not None:
            plot_roc_curve(y_test, y_proba, model_name)
        plot_learning_curve(best_model, X, y, model_name)
        plot_feature_importance(best_model, X_raw, y, selected_features, model_name)

    results_df = pd.DataFrame(results)
    results_df.sort_values(by='F1', ascending=False, inplace=True)

    print("\n 전체 모델 성능 요약:")
    print(
        results_df[['Model', 'Accuracy', 'Precision', 'Recall', 'F1', 'Balanced_Accuracy', 'MCC', 'AUC']]
        .to_string(index=False, float_format='{:0.5f}'.format)
    )

    results_df.to_excel('./result_ml/model_comparison.xlsx', index=False)
    print("\n✅ 결과가 './result_ml/model_comparison.xlsx' 에 저장되었습니다.")
