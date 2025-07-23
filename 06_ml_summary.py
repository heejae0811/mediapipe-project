import glob, os
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split, GridSearchCV, learning_curve
from sklearn.metrics import (
    confusion_matrix, classification_report, roc_curve, auc, roc_auc_score,
    balanced_accuracy_score, precision_score, recall_score, f1_score, matthews_corrcoef
)
from sklearn.inspection import permutation_importance
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from xgboost import XGBClassifier
from catboost import CatBoostClassifier

os.makedirs('./result', exist_ok=True)

# 데이터 전처리
def data_processing():
    csv_files = glob.glob('./features_xlsx/*.xlsx')
    df = pd.concat([pd.read_excel(file, sheet_name=0) for file in csv_files], ignore_index=True)
    print(f'[정보] 총 데이터 수: {len(df)}개 샘플')

    selected_features = [
        'landmark4_max',
        'landmark8_max',
        'landmark9_max',
        'landmark7_max',
        'landmark6_max',
        'landmark5_max',
        'landmark28_min',
        'landmark3_max',
        'landmark8_std',
        'landmark27_median',
        'landmark13_max',
        'landmark2_max',
        'landmark24_std'
    ]

    X_raw = df[selected_features]
    X = StandardScaler().fit_transform(X_raw)
    y = LabelEncoder().fit_transform(df['label'])

    return df, X_raw, X, y


# 모델 정의
def get_models():
    return {
        'Decision Tree': {
            'estimator': DecisionTreeClassifier(random_state=42),
            'param_grid': {
                'max_depth': [None, 3, 5, 7],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 2, 4],
                'criterion': ['gini', 'entropy']
            }
        },
        'Random Forest': {
            'estimator': RandomForestClassifier(random_state=42),
            'param_grid': {
                'n_estimators': [50, 100, 200],
                'max_depth': [None, 3, 5],
                'min_samples_split': [2, 5],
                'min_samples_leaf': [1, 2],
                'criterion': ['gini', 'entropy']
            }
        },
        'KNN': {
            'estimator': KNeighborsClassifier(),
            'param_grid': {
                'n_neighbors': [1, 3, 5],
                'weights': ['uniform', 'distance'],
                'metric': ['euclidean', 'manhattan']
            }
        },
        'SVM': {
            'estimator': SVC(probability=True, random_state=42),
            'param_grid': {
                'C': [0.1, 1, 10],
                'kernel': ['linear', 'rbf'],
                'gamma': ['scale', 0.01, 0.001]
            }
        },
        'Logistic Regression': {
            'estimator': LogisticRegression(max_iter=1000, random_state=42),
            'param_grid': {
                'C': [0.1, 1, 10],
                'penalty': ['l1', 'l2'],
                'solver': ['liblinear', 'saga']
            }
        },
        'Naive Bayes': {
            'estimator': GaussianNB(),
            'param_grid': {}
        },
        'XGBoost': {
            'estimator': XGBClassifier(random_state=42, eval_metric='logloss'),
            'param_grid': {
                'n_estimators': [50, 100, 200],
                'max_depth': [3, 5, 7],
                'learning_rate': [0.01, 0.1, 0.2],
                'subsample': [0.8, 1.0]
            }
        },
        'CatBoost': {
            'estimator': CatBoostClassifier(random_state=42, verbose=0),
            'param_grid': {
                'depth': [3, 5, 7],
                'learning_rate': [0.01, 0.1],
                'iterations': [50, 100, 200]
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
def plot_confusion_matrix(model_name, y_test, y_pred):
    cm = confusion_matrix(y_test, y_pred)

    plt.figure(figsize=(6, 5))
    sns.heatmap(
        cm,
        annot=True,
        fmt='d',
        cmap='Blues',
        xticklabels=['Beginner', 'Advanced'],
        yticklabels=['Beginner', 'Advanced']
    )
    plt.title(f'Confusion Matrix - {model_name}')
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.tight_layout()
    plt.show()


def plot_roc_curve(model_name, y_test, y_proba):
    fpr, tpr, _ = roc_curve(y_test, y_proba)
    roc_auc = auc(fpr, tpr)

    plt.figure()
    plt.plot(fpr, tpr, lw=2, label=f'AUC = {roc_auc:.2f}')
    plt.plot([0, 1], [0, 1], linestyle='--', color='gray')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'ROC Curve - {model_name}')
    plt.legend(loc='lower right')
    plt.grid(True, alpha=0.5)
    plt.tight_layout()
    plt.show()


def plot_learning_curve(model_name, estimator, X_train, y_train):
    train_sizes, train_scores, val_scores = learning_curve(
        estimator,
        X_train,
        y_train,
        train_sizes=np.linspace(0.1,1.0,5),
        cv=3
    )

    plt.figure()
    plt.plot(train_sizes, train_scores.mean(axis=1), 'o-', label='Train')
    plt.plot(train_sizes, val_scores.mean(axis=1), 'o-', label='Validation')
    plt.title(f'Learning Curve - {model_name}')
    plt.xlabel('Training Size')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(axis='y', alpha=0.5)
    plt.tight_layout()
    plt.show()


def plot_accuracy_bar(model_name, train_score, test_score):
    plt.figure()
    plt.bar(['Train Accuracy', 'Test Accuracy'], [train_score, test_score], color=['#1f77b4', '#ff7f0e'])
    plt.ylim(0, 1.1)
    plt.ylabel('Accuracy')
    plt.title(f'Accuracy - {model_name} (Train vs Test)')
    for i, v in enumerate([train_score, test_score]):
        plt.text(i, v + 0.02, f'{v:.2f}', ha='center')
    plt.grid(axis='y', alpha=0.5)
    plt.tight_layout()
    plt.show()


# 모델 실행
def run_model(model_name, model_info, X_train, X_test, y_train, y_test, X_raw):
    estimator = model_info['estimator']
    param_grid = model_info['param_grid']

    if param_grid:
        grid = GridSearchCV(estimator, param_grid, cv=3, scoring='f1', n_jobs=-1)
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

    # 시각화
    plot_confusion_matrix(model_name, y_test, y_pred)
    if y_proba is not None:
        plot_roc_curve(model_name, y_test, y_proba)
    plot_learning_curve(model_name, best_model, X_train, y_train)

    train_score = best_model.score(X_train, y_train)
    test_score = best_model.score(X_test, y_test)
    plot_accuracy_bar(model_name, train_score, test_score)

    # Feature Importance
    if hasattr(best_model, 'feature_importances_'):
        importances = best_model.feature_importances_
        indices = np.argsort(importances)[::-1]
        features = X_raw.columns[indices]

        plt.figure(figsize=(12, 6))
        plt.bar(range(len(features)), importances[indices], align='center')
        plt.xticks(range(len(features)), features, rotation=45, ha='right', fontsize=10)
        plt.ylabel('Importance Score')
        plt.title(f'Feature Importance - {model_name}')
        plt.grid(axis='y', alpha=0.5)
        plt.tight_layout()
        plt.show()
    else:
        result = permutation_importance(
            estimator=best_model,
            X=X_test,
            y=y_test,
            scoring='roc_auc',
            n_repeats=30,
            random_state=42,
            n_jobs=-1
        )
        importances = result.importances_mean
        stds = result.importances_std
        indices = np.argsort(importances)[::-1]
        features = X_raw.columns[indices]

        plt.figure(figsize=(12, 6))
        plt.bar(range(len(features)), importances[indices], yerr=stds[indices], align='center')
        plt.xticks(range(len(features)), features, rotation=45, ha='right', fontsize=10)
        plt.ylabel('Decrease in AUC')
        plt.title(f'Permutation Importance - {model_name}')
        plt.grid(True, alpha=0.5)
        plt.tight_layout()
        plt.show()

    return metrics


# 실행
if __name__ == '__main__':
    df, X_raw, X, y = data_processing()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

    models = get_models()
    results = []

    for model_name, model_info in models.items():
        metrics = run_model(model_name, model_info, X_train, X_test, y_train, y_test, X_raw)
        results.append(metrics)

    results_df = pd.DataFrame(results)
    results_df.sort_values(by='F1', ascending=False, inplace=True)

    # 📋 터미널에 예쁘게 출력
    print("\n📊 전체 모델 성능 요약:")
    print(
        results_df[['Model', 'Accuracy', 'Precision', 'Recall', 'F1', 'Balanced_Accuracy', 'MCC', 'AUC']]
        .to_string(index=False, float_format='{:0.4f}'.format)
    )

    results_df.to_excel('./result/model_comparison.xlsx', index=False)
    print("\n✅ 결과가 './result/model_comparison.xlsx' 에 저장되었습니다.")