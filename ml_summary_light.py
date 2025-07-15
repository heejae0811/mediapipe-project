import os
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split, GridSearchCV, learning_curve
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc, ConfusionMatrixDisplay, balanced_accuracy_score
from sklearn.inspection import permutation_importance
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression

result_summary = []

# 1. 데이터 전처리 함수
def data_processing():
    csv_files = glob.glob('./csv_features/*.csv')
    df = pd.concat([pd.read_csv(file) for file in csv_files], ignore_index=True)
    print(f'[정보] 총 데이터 수: {len(df)}개 샘플')

    X_raw = df.drop(['id', 'label'], axis=1)
    X = StandardScaler().fit_transform(X_raw)
    y = LabelEncoder().fit_transform(df['label'])

    return df, X_raw, X, y


# 2. 머신러닝 모델 정의 함수
def get_models():
    return {
        'Decision Tree': {
            'estimator': DecisionTreeClassifier(random_state=42),
            'param_grid': {
                'criterion': ['gini', 'entropy'],
                'max_depth': [3, 4, 5],
                'min_samples_leaf': [1, 2],
                'class_weight': ['balanced']
            }
        },
        'Random Forest': {
            'estimator': RandomForestClassifier(random_state=42),
            'param_grid': {
                'n_estimators': [50, 100],
                'max_depth': [5, 10],
                'min_samples_leaf': [1, 2],
                'class_weight': ['balanced']
            }
        },
        'KNN': {
            'estimator': KNeighborsClassifier(),
            'param_grid': {
                'n_neighbors': [3, 5],
                'weights': ['uniform', 'distance'],
                'metric': ['euclidean']
            }
        },
        'SVM': {
            'estimator': SVC(probability=True, random_state=42),
            'param_grid': {
                'C': [0.1, 1],
                'kernel': ['linear', 'rbf'],
                'gamma': ['scale', 'auto']
            }
        },
        'Logistic Regression': {
            'estimator': LogisticRegression(max_iter=1000, random_state=42),
            'param_grid': {
                'C': [0.1, 1],
                'penalty': ['l2'],
                'solver': ['liblinear']
            }
        }
    }


# 3. 시각화 함수들
def plot_confusion_matrix(model_name, model_estimator, X_test, y_test):
    ConfusionMatrixDisplay.from_estimator(
        estimator=model_estimator,
        X=X_test,
        y=y_test,
        display_labels=['Beginner', 'Advanced'],
        cmap='Blues',
        values_format='d'
    )

    plt.title(f'Confusion Matrix - {model_name}')
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


def plot_learning_curve(model_name, model_estimator, X_train, y_train):
    train_sizes, train_scores, val_scores = learning_curve(
        estimator=model_estimator,
        X=X_train,
        y=y_train,
        train_sizes=np.linspace(0.1, 1.0, 5),
        cv=3,
        scoring='accuracy',
        shuffle=True,
        random_state=42
    )
    train_mean = np.mean(train_scores, axis=1)
    val_mean = np.mean(val_scores, axis=1)

    plt.figure()
    plt.plot(train_sizes, train_mean, 'o--', label='Training Accuracy')
    plt.plot(train_sizes, val_mean, 'o-', label='Validation Accuracy')
    plt.title(f'Learning Curve - {model_name}')
    plt.xlabel('Training Set Size')
    plt.ylabel('Accuracy')
    plt.ylim(0, 1.1)
    plt.grid(True, alpha=0.5)
    plt.legend(loc='lower right')
    plt.tight_layout()
    plt.show()


# 4. 모델 실행 함수
def run_model(model_name, model_info, X_train, X_test, y_train, y_test, X_raw):
    grid_search = GridSearchCV(
        estimator=model_info['estimator'],
        param_grid=model_info['param_grid'],
        cv=5,
        scoring='f1',
        n_jobs=-1,
        verbose=1,
        return_train_score=True
    )
    grid_search.fit(X_train, y_train)
    best_model = grid_search.best_estimator_

    y_pred = best_model.predict(X_test)
    y_proba = best_model.predict_proba(X_test)[:, 1]

    print(f'\n========== {model_name} 결과 ==========')
    print('Best Parameters:', grid_search.best_params_)
    print('Best f1-score (CV mean):', f'{grid_search.best_score_:.5f}')
    print('Confusion Matrix:\n', confusion_matrix(y_test, y_pred))
    print('Classification Report:\n', classification_report(y_test, y_pred, digits=5, zero_division=0))
    print(f'Balanced Accuracy: {balanced_accuracy_score(y_test, y_pred):.5f}')

    report = classification_report(y_test, y_pred, output_dict=True, digits=5, zero_division=0)
    result_summary.append({
        'Model': model_name,
        'Best Parameters': str(grid_search.best_params_),
        'Best F1 (CV Mean)': round(grid_search.best_score_, 5),
        'Balanced Accuracy': round(balanced_accuracy_score(y_test, y_pred), 5),
        'Precision': round(report['macro avg']['precision'], 5),
        'Recall': round(report['macro avg']['recall'], 5),
        'F1-score': round(report['macro avg']['f1-score'], 5),
        'Support': int(report['macro avg']['support'])
    })

    plot_confusion_matrix(model_name, best_model, X_test, y_test)
    plot_roc_curve(model_name, y_test, y_proba)

    if hasattr(best_model, 'feature_importances_'):
        importances = best_model.feature_importances_
        indices = np.argsort(importances)[::-1]
        features = X_raw.columns[indices]
        plt.figure(figsize=(12, 6))
        plt.bar(range(len(features)), importances[indices], align='center')
        plt.xticks(range(len(features)), features, rotation=45, ha='right', fontsize=10)
        plt.ylabel('Importance Score')
        plt.title(f'Feature Importance - {model_name}')
        plt.grid(True, alpha=0.5)
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

    train_score = best_model.score(X_train, y_train)
    test_score = best_model.score(X_test, y_test)
    plot_accuracy_bar(model_name, train_score, test_score)
    plot_learning_curve(model_name, best_model, X_train, y_train)


# 5. 실행
if __name__ == '__main__':
    df, X_raw, X, y = data_processing()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
    models = get_models()

    for model_name, model_info in models.items():
        run_model(model_name, model_info, X_train, X_test, y_train, y_test, X_raw)

    # 6. csv 저장
    os.makedirs('result', exist_ok=True)
    pd.DataFrame(result_summary).to_csv('./results/ml_evaluation.csv', index=False, encoding='utf-8-sig')
    print('\ncsv가 저장되었습니다.')
