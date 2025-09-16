import glob
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.ensemble import RandomForestClassifier
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier
from catboost import CatBoostClassifier
from sklearn.metrics import confusion_matrix, classification_report, f1_score, matthews_corrcoef, precision_score, recall_score, roc_auc_score, roc_curve, auc, balanced_accuracy_score, ConfusionMatrixDisplay
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold, learning_curve
from sklearn.feature_selection import RFECV
from sklearn.inspection import permutation_importance
from sklearn.preprocessing import LabelEncoder, StandardScaler

RANDOM_STATE = 42

# 데이터 불러오기
def data_processing():
    csv_files = glob.glob('./features_xlsx/*.xlsx')
    print(f"\n📂 분석할 파일 수 - {len(csv_files)}개")

    df_all = pd.concat([pd.read_excel(file, sheet_name=4) for file in csv_files], ignore_index=True)
    y_all = LabelEncoder().fit_transform(df_all['label'])
    print(f"라벨 분포 - 0: {(y_all == 0).sum()}개 / 1: {(y_all == 1).sum()}개")

    feature_cols = df_all.select_dtypes(include=['float64', 'int64']).columns.drop('label')
    raw_features = df_all[feature_cols]
    X_train_raw, X_test_raw, y_train, y_test = train_test_split(
        raw_features, y_all, test_size=0.2, stratify=y_all, random_state=RANDOM_STATE
    )

    return df_all, X_train_raw, X_test_raw, y_train, y_test, feature_cols

# 상관계수 0.9 이상
def find_highly_correlation(X, threshold=0.9):
    corr_matrix = X.corr().abs()
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))

    correlated_pairs = [
        (col1, col2, upper.loc[col1, col2])
        for col1 in upper.columns
        for col2 in upper.index
        if upper.loc[col1, col2] > threshold
    ]

    print(f"\n🔍 상관계수 {threshold} 이상인 변수쌍 {len(correlated_pairs)}개:")
    for col1, col2, corr in correlated_pairs:
        print(f" - {col1} ↔ {col2}: {corr:.2f}")

    return correlated_pairs

# 중요도 낮은 변수 제거
def drop_low_importance(X, y, correlated_pairs):
    rf = RandomForestClassifier(n_estimators=500, random_state=RANDOM_STATE, n_jobs=-1)
    rf.fit(X, y)
    importances = pd.Series(rf.feature_importances_, index=X.columns)

    to_drop = set()
    for col1, col2, _ in correlated_pairs:
        if col1 in to_drop or col2 in to_drop:
            continue
        if importances[col1] < importances[col2]:
            to_drop.add(col1)
        else:
            to_drop.add(col2)

    print(f"\n🚮 상관관계 변수 중 중요도 낮은 {len(to_drop)}개 제거:")
    for col in to_drop:
        print(f" - {col}")

    return list(to_drop)

# RFECV

def run_rfecv(X, y):
    estimator = LGBMClassifier(
        n_estimators=200,
        learning_rate=0.1,
        max_depth=5,
        random_state=RANDOM_STATE,
        n_jobs=-1,
        verbosity=-1
    )

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

    rfecv = RFECV(
        estimator=estimator,
        step=3,  # 3개씩 제거 → 속도 향상
        cv=cv,
        scoring='roc_auc',
        n_jobs=-1
    )

    rfecv.fit(X, y)
    selected = X.columns[rfecv.support_]

    # 그래프 출력
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, len(rfecv.cv_results_['mean_test_score']) + 1),
             rfecv.cv_results_['mean_test_score'], marker='o')
    plt.xlabel("Number of Selected Features")
    plt.ylabel("Cross-Validation Score (AUC)")
    plt.title("RFECV - LightGBM")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    print(f"\n🎯 RFECV 최종 선택 변수 {len(selected)}개:")
    print(selected.tolist())

    return selected

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
        'Accuracy': (tn + tp) / (tn + fp + fn + tp), # 전체 샘플 중에서 정답을 맞춘 비율
        'Precision': precision_score(y_true, y_pred), # 1 이라고 예측한 것 중에 실제 1인 비율
        'Recall': recall_score(y_true, y_pred), # 실제 1인 것 중에서 모델이 1 이라고 맞춘 비율
        'F1': f1_score(y_true, y_pred), # Precision과 Recall의 조화 평균
        'Balanced_Accuracy': balanced_accuracy_score(y_true, y_pred), # 클래스 불균형이 심할 떄 적합한 정확도 측정 지표
        'Specificity': tn / (tn + fp), # 0 이라고 예측한 것 중에 실제 0인 비율
        'Sensitivity': tp / (tp + fn), # = Recall
        'MCC': matthews_corrcoef(y_true, y_pred), # 균형 잡힌 분류 성능 평가
        'AUC': roc_auc_score(y_true, y_proba) if y_proba is not None else None # 모델이 무작위보다 얼마나 나은 분류자인지 측정
    }

# Confusion Matrix
def plot_confusion_matrix(y_true, y_pred, model_name):
    ConfusionMatrixDisplay.from_predictions(y_true, y_pred, display_labels=['Beginner', 'Trained'], cmap='Blues')
    plt.title(f'Confusion Matrix - {model_name}')
    plt.tight_layout()
    plt.show()

# Roc Curve
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

# Combined Roc Curve
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

# Learning Curve
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

# Feature Importance + Permutation Importance
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

# Decision Tree
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

# 시각화
def visualize_model_results(model, model_name, X_train, y_train, X_test, y_test, selected_features):
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1] if hasattr(model, "predict_proba") else None

    plot_confusion_matrix(y_test, y_pred, model_name)

    if y_proba is not None:
        plot_roc_curve(y_test, y_proba, model_name)

    plot_learning_curve(model, X_train, y_train, model_name)

    plot_feature_importance(model, X_train, y_train, selected_features, model_name)

    if model_name == 'Decision Tree':
        plot_decision_tree(model, selected_features, class_names=['Beginner', 'Trained'], model_name=model_name)

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
    print(f"Best F1 (CV): {grid.best_score_:.5f}")

    return {
        'Model': name,
        'Best Params': str(grid.best_params_),
        'Best Model': best_model,
        **compute_metrics(y_test, y_pred, y_proba)
    }

# main
if __name__ == '__main__':
    df_all, X_train_raw, X_test_raw, y_train, y_test, feature_cols = data_processing()

    # 1. 상관관계 쌍 찾기
    correlated_pairs = find_highly_correlation(X_train_raw, threshold=0.85)

    # 2. 중요도 낮은 변수 제거
    to_drop = drop_low_importance(X_train_raw, y_train, correlated_pairs)

    # 3. 남은 변수만 추출
    all_corr_vars = set([col for pair in correlated_pairs for col in pair[:2]])
    remaining_corr_vars = all_corr_vars - set(to_drop)
    uncorrelated_vars = set(X_train_raw.columns) - all_corr_vars
    final_candidate_vars = list(remaining_corr_vars | uncorrelated_vars)
    X_train_filtered = X_train_raw[final_candidate_vars]

    print(f"\n🔧 RFECV 대상 변수 수: {len(final_candidate_vars)}")

    # 4. RFECV 최종 변수 선택
    selected_features = run_rfecv(X_train_filtered, y_train)

    # 5. 표준화
    scaler = StandardScaler()
    scaler.fit(X_train_raw[selected_features])
    X_train = pd.DataFrame(scaler.transform(X_train_raw[selected_features]), columns=selected_features)
    X_test = pd.DataFrame(scaler.transform(X_test_raw[selected_features]), columns=selected_features)

    # 6. 모델 학습 및 평가
    results = []
    for model_name, model_info in get_models().items():
        metrics = run_model(model_name, model_info, X_train, X_test, y_train, y_test)
        results.append(metrics)

    # 7. 시각화
    for res in results:
        model = res['Best Model']
        model_name = res['Model']
        visualize_model_results(model, model_name, X_train, y_train, X_test, y_test, selected_features)

    # 8. ROC 통합
    plot_combined_roc_curves(results, X_test, y_test)

    # 9. 결과 저장
    results_df = pd.DataFrame(results)
    results_df.sort_values(by='F1', ascending=False, inplace=True)
    print("\n✅ 전체 모델 성능 요약:")
    print(results_df[['Model', 'Accuracy', 'Precision', 'Recall', 'F1', 'AUC']].to_string(index=False))

    results_df.to_excel('./result/model_comparison.xlsx', index=False)
    print("\n✅ 결과 저장 완료: ./result/model_comparison.xlsx")