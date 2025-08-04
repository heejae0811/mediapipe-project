import glob
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from sklearn.tree import plot_tree
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
RANDOM_STATE = 42

# 데이터 전처리
def data_processing():
    csv_files = glob.glob('./features_xlsx_일대일/*.xlsx')
    print(f"\n📂 분석할 파일 수 - {len(csv_files)}개")

    df_all = pd.concat([pd.read_excel(file, sheet_name=2) for file in csv_files], ignore_index=True)
    y_all = LabelEncoder().fit_transform(df_all['label'])
    print(f"라벨 분포 - 0: {(y_all == 0).sum()}개 / 1: {(y_all == 1).sum()}개")

    feature_cols = df_all.select_dtypes(include=['float64', 'int64']).columns.drop('label')
    raw_features = df_all[feature_cols]
    X_train_raw, X_test_raw, y_train, y_test = train_test_split(raw_features, y_all, test_size=0.2, stratify=y_all, random_state=RANDOM_STATE)

    return df_all, X_train_raw, X_test_raw, y_train, y_test, feature_cols

# 랜덤 포레스트 변수 선택
def select_top_features_by_rf(X_train, y_train, feature_cols, top_n=20):
    rf = RandomForestClassifier(
        n_estimators=500,
        max_depth=None,
        max_features='sqrt',
        class_weight='balanced',
        random_state=RANDOM_STATE,
        n_jobs=-1
    )
    rf.fit(X_train, y_train)
    importances = rf.feature_importances_
    importance_df = pd.DataFrame({'feature': feature_cols, 'importance': importances})
    importance_df = importance_df.sort_values(by='importance', ascending=False)
    print(f"\n🎯 Random Forest Top {top_n} 중요 변수:")

    for i, row in importance_df.head(top_n).iterrows():
        print(f"{i+1:2d}. {row['feature']} ({row['importance']:.5f})")

    plt.figure(figsize=(12, 6))
    sns.barplot(x='importance', y='feature', data=importance_df.head(top_n))
    plt.title(f'Random Forest Top {top_n} Features')
    plt.grid(True, linestyle='--', linewidth=0.4, alpha=0.6)
    plt.tight_layout()
    plt.show()

    return importance_df['feature'].tolist()

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
                'n_neighbors': [1, 3, 5],
                'weights': ['uniform']
            }
        },
        'SVM': {
            'estimator': SVC(probability=True, random_state=RANDOM_STATE),
            'param_grid': {
                'C': [0.1, 1, 10],
                'kernel': ['rbf'],
                'gamma': ['scale']
            }
        },
        'Decision Tree': {
            'estimator': DecisionTreeClassifier(random_state=RANDOM_STATE),
            'param_grid': {
                'max_depth': [2, 3, 4],
                'min_samples_split': [2, 3],
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
            'estimator': LGBMClassifier(random_state=RANDOM_STATE, verbose=-1),
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
                'learning_rate': [0.05, 0.1, 0.2]
            }
        },
        'CatBoost': {
            'estimator': CatBoostClassifier(random_state=RANDOM_STATE, verbose=0),
            'param_grid': {
                'depth': [3, 5],
                'iterations': [100],
                'learning_rate': [0.05, 0.1, 0.2]
            }
        }
    }

# 성능 평가 지표 계산
def compute_metrics(y_true, y_pred, y_proba):
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()

    return {
        'Accuracy': (tn + tp) / (tn + fp + fn + tp),    # 전체 중 맞춘 비율
        'Precision': precision_score(y_true, y_pred),   # 1 이라고 예측한 것 중 진짜 1의 비율
        'Recall': recall_score(y_true, y_pred),         # 실제 1 중에서 1 이라고 맞춘 비율
        'F1': f1_score(y_true, y_pred),
        'Balanced_Accuracy': balanced_accuracy_score(y_true, y_pred),
        'Specificity': tn / (tn + fp),
        'Sensitivity': tp / (tp + fn),
        'MCC': matthews_corrcoef(y_true, y_pred),
        'AUC': roc_auc_score(y_true, y_proba) if y_proba is not None else None
    }

# 시각화 함수들
def plot_confusion_matrix(y_true, y_pred, model_name):
    ConfusionMatrixDisplay.from_predictions(y_true, y_pred, display_labels=['Beginner', 'Trained'], cmap='Blues')
    plt.title(f'Confusion Matrix - {model_name}')
    plt.tight_layout()
    plt.show()

def plot_roc_curve(y_true, y_proba, model_name):
    fpr, tpr, _ = roc_curve(y_true, y_proba)
    plt.plot(fpr, tpr, label=f'AUC = {auc(fpr, tpr):.2f}')
    plt.plot([0, 1], [0, 1], 'k--')
    plt.title(f'ROC Curve - {model_name}')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.grid(True, axis='both', linestyle='-', linewidth=0.4, color='gray', alpha=0.4)
    plt.legend(loc='lower right')
    plt.tight_layout()
    plt.show()

def plot_combined_roc_curves(results, X_test, y_test):
    colors = ['pink', 'red', 'orange', 'green', 'blue', 'navy', 'purple', 'brown']
    plt.figure(figsize=(10, 6))

    for result, color in zip(results, colors):
        name = result['Model']
        model = result.get('Best Model')

        if model is None:
            print(f"⚠ {name}: Best Model 없음")
            continue

        try:
            if hasattr(model, "predict_proba"):
                y_prob = model.predict_proba(X_test)[:, 1]
            elif hasattr(model, "decision_function"):
                y_prob = model.decision_function(X_test)
            else:
                print(f"⚠ {name}는 확률 예측 불가")
                continue

            fpr, tpr, _ = roc_curve(y_test, y_prob)
            roc_auc = auc(fpr, tpr)
            plt.plot(fpr, tpr, label=f'{name} (AUC = {roc_auc:.2f})', color=color)

        except Exception as e:
            print(f"⚠ {name} 예측 실패: {e}")

    plt.plot([0, 1], [0, 1], linestyle='--', color='gray')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve (All Models)')
    plt.legend(loc='lower right')
    plt.grid(True, axis='both', linestyle='-', linewidth=0.4, color='gray', alpha=0.4)
    plt.tight_layout()
    plt.show()

def plot_learning_curve(estimator, X, y, model_name):
    sizes, train_scores, val_scores = learning_curve(estimator, X, y, cv=5, scoring='f1')
    train_mean = train_scores.mean(axis=1)
    train_std = train_scores.std(axis=1)
    val_mean = val_scores.mean(axis=1)
    val_std = val_scores.std(axis=1)
    plt.plot(sizes, train_mean, 'o-', label='Train')
    plt.plot(sizes, val_mean, 'o-', label='Validation')
    plt.fill_between(sizes, train_mean - train_std, train_mean + train_std, alpha=0.2)
    plt.fill_between(sizes, val_mean - val_std, val_mean + val_std, alpha=0.2)
    plt.title(f'Learning Curve - {model_name}')
    plt.xlabel('Training Set Size')
    plt.ylabel('F1-score')
    plt.grid(True, axis='both', linestyle='-', linewidth=0.4, color='gray', alpha=0.4)
    plt.legend(loc='lower right')
    plt.tight_layout()
    plt.show()

def plot_feature_importance(model, X, y, feature_names, model_name):
    if hasattr(model, 'feature_importances_'):
        imp = model.feature_importances_
        sorted_idx = np.argsort(imp)[::-1]
        plt.figure(figsize=(12, 6))
        sns.barplot(x=imp[sorted_idx][:10], y=np.array(feature_names)[sorted_idx][:10])
        plt.title(f'Feature Importance - {model_name}')
        plt.grid(True, axis='both', linestyle='-', linewidth=0.4, color='gray', alpha=0.4)
        plt.tight_layout()
        plt.show()

    result = permutation_importance(model, X, y, n_repeats=10, random_state=RANDOM_STATE)
    idx = result.importances_mean.argsort()[::-1]
    plt.figure(figsize=(12, 6))
    sns.barplot(x=result.importances_mean[idx][:10], y=np.array(feature_names)[idx][:10])
    plt.title(f'Permutation Importance - {model_name}')
    plt.grid(True, axis='both', linestyle='-', linewidth=0.4, color='gray', alpha=0.4)
    plt.tight_layout()
    plt.show()

def plot_decision_tree(model, feature_names, class_names=['Beginner', 'Trained'], model_name='Decision Tree'):
    if not hasattr(model, 'tree_'):
        print(f"❌ {model_name}는 결정트리 기반 모델이 아닙니다.")
        return

    plt.figure(figsize=(30, 10))
    plot_tree(
        model,
        feature_names=feature_names,
        class_names=class_names,
        filled=True,
        rounded=True,
        fontsize=10
    )
    plt.title(f"Decision Tree Structure - {model_name}")
    plt.tight_layout()
    plt.show()

    print(f"Max Depth of {model_name}: {model.get_depth()}")

# 모델 실행
def run_model(name, model_info, X_train, X_test, y_train, y_test):
    grid = GridSearchCV(model_info['estimator'], model_info['param_grid'], scoring='f1', cv=5, n_jobs=-1)
    grid.fit(X_train, y_train)
    best_model = grid.best_estimator_
    y_pred = best_model.predict(X_test)
    y_proba = best_model.predict_proba(X_test)[:, 1] if hasattr(best_model, "predict_proba") else None

    print(f"\n========== {name} ==========")
    print(confusion_matrix(y_test, y_pred))
    print(classification_report(y_test, y_pred))
    print(f"Best Params: {grid.best_params_}")
    print(f"Best F1 (CV): {grid.best_score_:.4f}")

    plot_confusion_matrix(y_test, y_pred, name)
    if y_proba is not None:
        plot_roc_curve(y_test, y_proba, name)

    return {
        'Model': name,
        'Best Params': str(grid.best_params_),
        'Best Model': best_model,
        **compute_metrics(y_test, y_pred, y_proba)
    }

if __name__ == '__main__':
    # 1. 데이터 로딩 및 분할
    df_all, X_train_raw, X_test_raw, y_train, y_test, feature_cols = data_processing()

    # 2. 랜덤포레스트로 Top 변수 추천
    top_features_all = select_top_features_by_rf(X_train_raw, y_train, feature_cols, top_n=20)

    # 3. 변수 선택 (터미널 입력)
    input_str = input("\n📥 사용할 변수명을 쉼표(,)로 입력하세요:\n")
    selected_features = [var.strip() for var in input_str.split(',') if var.strip()]
    for var in selected_features:
        if var not in X_train_raw.columns:
            raise ValueError(f"❌ 변수 '{var}'는 데이터에 존재하지 않습니다.")

    # 4. 표준화 (StandardScaler)
    scaler = StandardScaler()
    scaler.fit(X_train_raw[selected_features])
    X_train = pd.DataFrame(scaler.transform(X_train_raw[selected_features]), columns=selected_features)
    X_test = pd.DataFrame(scaler.transform(X_test_raw[selected_features]), columns=selected_features)

    # 5. 모델 학습 및 결과 저장
    results = []
    for model_name, model_info in get_models().items():
        metrics = run_model(model_name, model_info, X_train, X_test, y_train, y_test)
        results.append(metrics)

    # 6. 전체 모델 ROC 커브 통합 시각화
    plot_combined_roc_curves(results, X_test, y_test)

    # 7. 모델별 추가 시각화 (중요도, 학습곡선, DT 시각화)
    for res in results:
        model = res['Best Model']
        model_name = res['Model']

        plot_feature_importance(model, X_train, y_train, selected_features, model_name)
        plot_learning_curve(model, X_train, y_train, model_name)

        if model_name == 'Decision Tree':
            plot_decision_tree(model, selected_features, class_names=['Beginner', 'Trained'], model_name='Decision Tree')

    # 8. 결과 요약 및 엑셀 저장
    results_df = pd.DataFrame(results)
    results_df.sort_values(by='F1', ascending=False, inplace=True)
    print("\n✅ 전체 모델 성능 요약:")
    print(results_df[['Model', 'Accuracy', 'Precision', 'Recall', 'F1', 'AUC']].to_string(index=False))

    results_df.to_excel('./result/model_comparison.xlsx', index=False)
    print("\n✅ 결과 저장 완료: ./result/model_comparison.xlsx")
