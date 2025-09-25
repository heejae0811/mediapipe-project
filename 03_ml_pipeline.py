import os
import glob
import warnings
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score, GridSearchCV
from sklearn.feature_selection import SelectKBest, f_classif, mutual_info_classif
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier
from catboost import CatBoostClassifier
from sklearn.metrics import (
    confusion_matrix, classification_report, f1_score, matthews_corrcoef,
    precision_score, recall_score, roc_auc_score, roc_curve, balanced_accuracy_score
)
from sklearn.pipeline import Pipeline
import optuna
from optuna.samplers import TPESampler
from optuna.pruners import MedianPruner


# ================================
# Global Variables
# ================================
warnings.filterwarnings('ignore')
RANDOM_STATE = 42

class TuningMethod:
    DEFAULT = "default"
    GRID_SEARCH = "grid"
    OPTUNA = "optuna"


# ================================
# Data Processing
# ================================
def data_processing():
    files = glob.glob('./features_xlsx/*.xlsx')
    print(f"📂 분석할 파일 수 - {len(files)}개")

    if not files:
        raise FileNotFoundError("경로에 파일이 없습니다.")

    df_all = pd.concat([pd.read_excel(file, sheet_name=4) for file in files], ignore_index=True)
    print(f"총 데이터 수: {len(df_all)}개")
    print(f"컬럼 수: {df_all.shape[1]}개")
    print(f"결측치 개수: {df_all.isnull().sum().sum()}개")

    y_all = LabelEncoder().fit_transform(df_all['label'])
    print(f"라벨 분포: 0 - {(y_all == 0).sum()}개 / 1 - {(y_all == 1).sum()}개")

    feature_cols = df_all.select_dtypes(include=['float64', 'int64']).columns.drop('label', errors='ignore')
    raw_features = df_all[feature_cols]
    X_train, X_test, y_train, y_test = train_test_split(
        raw_features, y_all, test_size=0.2, stratify=y_all, random_state=RANDOM_STATE
    )

    return df_all, X_train, X_test, y_train, y_test, feature_cols


# ========================================
# Feature Selection: filter + embedded
# ========================================
def feature_selection(X, y, final_k=50):
    original_features = len(X.columns)

    # 1단계: 분산 필터링
    print("1. 분산 필터링 (Variance Threshold)")
    variances = X.var()
    low_var_threshold = 0.001  # 매우 낮은 임계값
    low_variance_features = variances[variances <= low_var_threshold].index.tolist()
    remaining_features = [col for col in X.columns if col not in low_variance_features]
    X_filtered = X[remaining_features]

    print(f"   제거된 낮은 분산 특성: {len(low_variance_features)}개")
    print(f"   남은 특성: {len(remaining_features)}개")

    # 2단계: 상관관계 필터링 (0.9 이상 제거)
    print("\n2. 상관관계 필터링 (Pearson Correlation)")
    corr_threshold = 0.90  # 높은 임계값으로 설정
    corr_matrix = X_filtered.corr().abs()
    upper_triangle = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    highly_corr_features = [column for column in upper_triangle.columns if any(upper_triangle[column] > corr_threshold)]
    remaining_features = [col for col in remaining_features if col not in highly_corr_features]
    X_filtered = X_filtered[remaining_features]
    print(f"   제거된 높은 상관관계 특성: {len(highly_corr_features)}개")
    print(f"   남은 특성: {len(remaining_features)}개")

    # 3단계: ANOVA F-test로 1차 선별 (final_k의 2배까지)
    intermediate_k = min(final_k * 2, len(remaining_features))
    if len(remaining_features) > intermediate_k:
        print(f"\n3. ANOVA F-test로 1차 선별 ({intermediate_k}개)")

        selector_anova = SelectKBest(score_func=f_classif, k=intermediate_k)
        selector_anova.fit(X_filtered, y)

        anova_features = X_filtered.columns[selector_anova.get_support()].tolist()
        anova_scores = selector_anova.scores_[selector_anova.get_support()]
        X_filtered = X_filtered[anova_features]

        print(f"   ANOVA F-test로 선택된 특성: {len(anova_features)}개")
        print(f"   평균 F-score: {anova_scores.mean():.2f}")
        remaining_features = anova_features

    # 4단계: Mutual Information으로 최종 선별
    if len(remaining_features) > final_k:
        print(f"\n4. Mutual Information으로 최종 선별 ({final_k}개)")

        selector_mi = SelectKBest(score_func=lambda X, y: mutual_info_classif(X, y, random_state=RANDOM_STATE), k=final_k)

        SelectKBest
        selector_mi.fit(X_filtered, y)

        final_features = X_filtered.columns[selector_mi.get_support()].tolist()
        mi_scores = selector_mi.scores_[selector_mi.get_support()]

        print(f"   Mutual Information으로 최종 선택: {len(final_features)}개")
        print(f"   평균 MI score: {mi_scores.mean():.3f}")
    else:
        final_features = remaining_features
        print(f"\n4️⃣ 이미 목표 특성 수 이하이므로 모든 특성 사용: {len(final_features)}개")

    # 결과 요약
    print("=" * 50)
    print(f"\n✅ 최적 Filter 조합 완료!")
    print(f"📊 특성 선택 요약:")
    print(f"   원본 특성: {original_features:4d}개")
    print(f"   분산 필터링: {len(X.columns) - len(low_variance_features):4d}개 (제거: {len(low_variance_features)}개)")
    print(f"   상관관계 필터링: {len(remaining_features) + len(highly_corr_features):4d}개 (제거: {len(highly_corr_features)}개)")
    if len(X_filtered.columns) != len(final_features):
        print(f"   ANOVA 1차: {len(X_filtered.columns):4d}개")
    print(f"   최종 선택: {len(final_features):4d}개")
    print(f"   감소율: {((original_features - len(final_features)) / original_features * 100):5.1f}%")
    print("🎯 선택 방법: 분산 → 상관관계 → ANOVA F-test → Mutual Information")
    print("=" * 50)

    return final_features


# ================================
# Pipeline
# ================================
def make_pipeline_with_scaler(model, scaling=True):
    scaler = StandardScaler() if scaling else "passthrough"
    return Pipeline([("scaler", scaler), ("model", model)])


# ================================
# ML Models
# ================================
def get_all_models():
    scaling_models = {
        'Logistic Regression': LogisticRegression(
            random_state=RANDOM_STATE,
            max_iter=1000
        ),
        'K-Neighbors': KNeighborsClassifier(
            n_jobs=-1
        ),
        'Support Vector Machine': SVC(
            random_state=RANDOM_STATE,
            probability=True
        )
    }
    non_scaling_models = {
        'Decision Tree': DecisionTreeClassifier(
            random_state=RANDOM_STATE
        ),
        'Random Forest': RandomForestClassifier(
            random_state=RANDOM_STATE,
            n_jobs=-1
        ),
        'LightGBM': LGBMClassifier(
            random_state=RANDOM_STATE,
            n_jobs=-1,
            verbosity=-1
        ),
        'XGBoost': XGBClassifier(
            random_state=RANDOM_STATE,
            eval_metric="logloss",
            n_jobs=-1,
            verbosity=0
        ),
        'CatBoost': CatBoostClassifier(
            random_state=RANDOM_STATE,
            verbose=False
        )
    }

    models = {}

    for name, model in scaling_models.items():
        models[name] = make_pipeline_with_scaler(model, scaling=True)

    for name, model in non_scaling_models.items():
        models[name] = make_pipeline_with_scaler(model, scaling=False)

    return models


# ================================
# GridSearch Parameters
# ================================
def get_grid_search_params():
    return {
        'Logistic Regression': {
            'C': [0.01, 0.1, 1, 10, 100],
            'solver': ['liblinear', 'lbfgs'],
        },
        'K-Neighbors': {
            'n_neighbors': [3, 5, 7, 11, 15],
            'weights': ['uniform', 'distance']
        },
        'Support Vector Machine': {
            'C': [0.1, 1, 10],
            'kernel': ['rbf', 'linear'],
            'gamma': ['scale', 'auto']
        },
        'Decision Tree': {
            'max_depth': [3, 5, 10, None],
            'min_samples_split': [2, 5, 10],
            'criterion': ['gini', 'entropy']
        },
        'Random Forest': {
            'n_estimators': [50, 100, 200],
            'max_depth': [5, 10, None],
            'min_samples_split': [2, 5]
        },
        'LightGBM': {
            'n_estimators': [50, 100, 200],
            'learning_rate': [0.05, 0.1, 0.2],
            'max_depth': [3, 5, 7]
        },
        'XGBoost': {
            'n_estimators': [50, 100, 200],
            'learning_rate': [0.05, 0.1, 0.2],
            'max_depth': [3, 5, 7]
        },
        'CatBoost': {
            'iterations': [50, 100, 200],
            'learning_rate': [0.05, 0.1, 0.2],
            'depth': [3, 4, 5]
        }
    }

def grid_search_tuning(X_train, y_train):
    print("\n🎯 GridSearch 하이퍼파라미터 튜닝 시작")

    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=RANDOM_STATE)
    tuned = {}
    params = get_grid_search_params()
    models = get_all_models()

    for name, model in models.items():
        if name in params:
            search = GridSearchCV(model, params[name], cv=cv, scoring='f1', n_jobs=-1)
            search.fit(X_train, y_train)
            tuned[name] = search.best_estimator_
        else:
            tuned[name] = model.fit(X_train, y_train)

    return tuned


# ================================
# Optuna
# ================================
def create_optuna(model_name, X_train, y_train):
    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=RANDOM_STATE)

    def run_cv(pipeline):
        return cross_val_score(pipeline, X_train, y_train, cv=cv, scoring='f1', n_jobs=-1).mean()

    def objective(trial):
        if model_name == "Logistic Regression":
            model = LogisticRegression(random_state=RANDOM_STATE, max_iter=1000)
            params = {
                "model__C": trial.suggest_float("model__C", 0.001, 100, log=True),
                "model__solver": trial.suggest_categorical("model__solver", ["liblinear", "lbfgs"])
            }
            pipeline = make_pipeline_with_scaler(model, scaling=True).set_params(**params)

        elif model_name == "K-Neighbors":
            model = KNeighborsClassifier(n_jobs=-1)
            params = {
                "model__n_neighbors": trial.suggest_int("model__n_neighbors", 3, 15),
                "model__weights": trial.suggest_categorical("model__weights", ["uniform", "distance"])
            }
            pipeline = make_pipeline_with_scaler(model, scaling=True).set_params(**params)

        elif model_name == "Support Vector Machine":
            model = SVC(random_state=RANDOM_STATE, probability=True)
            params = {
                "model__C": trial.suggest_float("model__C", 0.01, 100, log=True),
                "model__kernel": trial.suggest_categorical("model__kernel", ["linear", "rbf"])
            }
            pipeline = make_pipeline_with_scaler(model, scaling=True).set_params(**params)

        elif model_name == "Decision Tree":
            model = DecisionTreeClassifier(random_state=RANDOM_STATE)
            params = {
                "model__max_depth": trial.suggest_categorical("model__max_depth", [3, 5, 10, None]),
                "model__min_samples_split": trial.suggest_int("model__min_samples_split", 2, 10)
            }
            pipeline = make_pipeline_with_scaler(model, scaling=False).set_params(**params)

        elif model_name == "Random Forest":
            model = RandomForestClassifier(random_state=RANDOM_STATE, n_jobs=-1)
            params = {
                "model__n_estimators": trial.suggest_int("model__n_estimators", 50, 300),
                "model__max_depth": trial.suggest_categorical("model__max_depth", [5, 10, None])
            }
            pipeline = make_pipeline_with_scaler(model, scaling=False).set_params(**params)

        elif model_name == "LightGBM":
            model = LGBMClassifier(random_state=RANDOM_STATE, n_jobs=-1, verbosity=-1)
            params = {
                "model__n_estimators": trial.suggest_int("model__n_estimators", 50, 300),
                "model__num_leaves": trial.suggest_int("model__num_leaves", 15, 63)
            }
            pipeline = make_pipeline_with_scaler(model, scaling=False).set_params(**params)

        elif model_name == "XGBoost":
            model = XGBClassifier(random_state=RANDOM_STATE, eval_metric="logloss", n_jobs=-1, verbosity=0)
            params = {
                "model__n_estimators": trial.suggest_int("model__n_estimators", 50, 300),
                "model__max_depth": trial.suggest_int("model__max_depth", 3, 7)
            }
            pipeline = make_pipeline_with_scaler(model, scaling=False).set_params(**params)

        elif model_name == "CatBoost":
            model = CatBoostClassifier(random_state=RANDOM_STATE, verbose=False)
            params = {
                "model__depth": trial.suggest_int("model__depth", 4, 10),
                "model__iterations": trial.suggest_int("model__iterations", 100, 300)
            }
            pipeline = make_pipeline_with_scaler(model, scaling=False).set_params(**params)

        else:
            raise ValueError(f"지원하지 않는 모델: {model_name}")

        return run_cv(pipeline)

    return objective


def optuna_tuning(X_train, y_train, n_trials=30):
    print("\n🎯 Optuna 하이퍼파라미터 튜닝 시작")

    objectives = create_optuna(X_train, y_train)
    optimized_models = {}
    optuna_results = []

    for name, objective in objectives.items():
        study = optuna.create_study(direction="maximize", sampler=TPESampler(seed=RANDOM_STATE), pruner=MedianPruner())
        study.optimize(objective, n_trials=n_trials)
        best_params = study.best_params
        base_models = get_all_models()
        optimized_models[name] = base_models[name].set_params(**best_params)
        optuna_results.append(
            {'Model': name, 'Best Params': best_params, 'Best CV F1': study.best_value}
        )
        print(f"Best CV F1: {study.best_value:.4f}")

    pd.DataFrame(optuna_results).to_excel('./result/results_optuna.xlsx', index=False)
    print("Optuna 결과 저장: ./result/results_optuna.xlsx")

    return optimized_models


# ================================
# ML Evaluation
# ================================
def compute_metrics(y_true, y_pred, y_proba=None):
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel() if cm.size == 4 else (0, 0, 0, 0)
    metrics = {
        'Accuracy': (y_true == y_pred).mean(), # 전체 샘플 중 맞춘 비율
        'Precision': precision_score(y_true, y_pred, zero_division=0), # 모델이 1 이라고 예측한 것 중 실제로 1인 비율
        'Recall': recall_score(y_true, y_pred, zero_division=0), # 실제 1 중 모델이 올바르게 1 이라고 맞춘 비율
        'F1': f1_score(y_true, y_pred, zero_division=0), # Precision, Recall의 조화 평균
        'Balanced_Accuracy': balanced_accuracy_score(y_true, y_pred), # 각 클래스의 Recall 평균
        'Specificity': tn / (tn + fp) if (tn + fp) > 0 else 0, # 실제 0 중 모델이 올바르게 0 이라고 맞춘 비율
        'Sensitivity': tp / (tp + fn) if (tp + fn) > 0 else 0, # = Recall
        'MCC': matthews_corrcoef(y_true, y_pred), # 이진 분류의 전반적인 상관관계 지표
        'AUC': roc_auc_score(y_true, y_proba) if y_proba is not None else 0 # ROC Curve 면적
    }
    return metrics


# ================================
# Visualization
# ================================
def plot_f1_comparison(results_df, tuning_method):
    subset = results_df[results_df['Tuning'] == tuning_method]
    subset_sorted = subset.sort_values('F1', ascending=False)

    plt.figure(figsize=(10, 6))
    colors = sns.color_palette("Set2", len(subset_sorted))
    bars = plt.barh(subset_sorted['Model'], subset_sorted['F1'], color=colors)
    for bar, f1 in zip(bars, subset_sorted['F1']):
        plt.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height()/2, f"{f1:.3f}", va="center")
    plt.title('F1 Score Comparison - All Models', fontsize=16, weight='bold')
    plt.xlabel('F1 Score')
    plt.grid(True, axis='x', linestyle='-', alpha=0.5)
    plt.tight_layout()
    plt.show()

def plot_roc_comparison(results, y_test, tuning_method):
    plt.figure(figsize=(10, 6))
    subset = [r for r in results if r['Tuning'] == tuning_method]
    for r in subset:
        if r['y_proba'] is not None:
            fpr, tpr, _ = roc_curve(y_test, r['y_proba'])
            plt.plot(fpr, tpr, label=f"{r['Model']} (AUC={r['AUC']:.3f})")
    plt.plot([0, 1], [0, 1], 'k--', linewidth=1)
    plt.title('ROC Curves - All Models', fontsize=16, weight='bold')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.legend(loc='lower right', frameon=True)
    plt.grid(True, linestyle='-', alpha=0.5)
    plt.tight_layout()
    plt.show()


# ================================
# Save Results
# ================================
def save_results_by_tuning(results_df, y_test):
    os.makedirs('./result', exist_ok=True)
    # 전체 저장
    results_df.to_excel('./result/final_results.xlsx', index=False)

    # 튜닝별 저장 + Plot 저장
    for tuning in results_df['Tuning'].unique():
        sub = results_df[results_df['Tuning'] == tuning]
        sub.to_excel(f'./result/results_{tuning.upper()}.xlsx', index=False)
        plot_f1_comparison_per_tuning(results_df, tuning)
        plot_roc_comparison_per_tuning(results_df.to_dict('records'), y_test, tuning)

    # 최고 성능 모델 저장
    best = results_df.sort_values('F1', ascending=False).iloc[0]
    pd.DataFrame([best]).to_excel('./result/best_model.xlsx', index=False)
    print(f"🏆 최고 성능 모델: {best['Model']} ({best['Tuning']}) | F1={best['F1']:.3f}")

# ================================
# Main
# ================================
def run_all():
    df_all, X_train, X_test, y_train, y_test, feature_cols = data_processing()
    selected = feature_selection(X_train, y_train, final_k=50)
    all_results = []

    for tuning_method in [TuningMethod.DEFAULT, TuningMethod.GRID_SEARCH, TuningMethod.OPTUNA]:
        print(f"\n===== {tuning_method.upper()} 실행 =====")
        if tuning_method == TuningMethod.DEFAULT:
            models = get_all_models()
        elif tuning_method == TuningMethod.GRID_SEARCH:
            models = grid_search_tuning(X_train[selected], y_train)
        elif tuning_method == TuningMethod.OPTUNA:
            models = optuna_tuning(X_train[selected], y_train)
        else:
            continue

        for name, model in models.items():
            model.fit(X_train[selected], y_train)
            preds = model.predict(X_test[selected])
            proba = model.predict_proba(X_test[selected])[:, 1] if hasattr(model, "predict_proba") else None
            m = compute_metrics(y_test, preds, proba)
            all_results.append({'Tuning': tuning_method, 'Model': name, 'y_pred': preds, 'y_proba': proba, **m})

    results_df = pd.DataFrame(all_results).sort_values(['Tuning', 'F1'], ascending=[True, False])
    print(results_df[['Tuning', 'Model', 'F1', 'AUC', 'Accuracy']])

    save_results_by_tuning(results_df, y_test)

if __name__ == "__main__":
    run_all()
