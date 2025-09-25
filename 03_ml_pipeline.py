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
    print(f"📂 분석할 파일 수: {len(files)}개")

    if not files:
        raise FileNotFoundError("경로에 파일이 없습니다.")

    df_all = pd.concat([pd.read_excel(file, sheet_name=4) for file in files], ignore_index=True)
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
    print(f"\n{'=' * 60}")
    print(f"👉 Feature Selection 시작")
    print(f"{'=' * 60}")

    original_features = len(X.columns)

    # 1단계: 분산 필터링
    print("1. 분산 필터링 (Variance Threshold)")
    variances = X.var()
    low_var_threshold = 0.001
    low_variance_features = variances[variances <= low_var_threshold].index.tolist()
    remaining_features = [col for col in X.columns if col not in low_variance_features]
    X_filtered = X[remaining_features]

    print(f"   제거된 낮은 분산 특성: {len(low_variance_features)}개")
    print(f"   남은 특성: {len(remaining_features)}개")

    # 2단계: 상관관계 필터링 (0.9 이상 제거)
    print("\n2. 상관관계 필터링 (Pearson Correlation)")
    corr_threshold = 0.90
    corr_matrix = X_filtered.corr().abs()
    upper_triangle = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    highly_corr_features = [column for column in upper_triangle.columns if any(upper_triangle[column] > corr_threshold)]
    remaining_features = [col for col in remaining_features if col not in highly_corr_features]
    X_filtered = X_filtered[remaining_features]
    print(f"   제거된 높은 상관관계 특성: {len(highly_corr_features)}개")
    print(f"   남은 특성: {len(remaining_features)}개")

    # 3단계: ANOVA F-test로 1차 선별
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
        selector_mi.fit(X_filtered, y)
        final_features = X_filtered.columns[selector_mi.get_support()].tolist()
        mi_scores = selector_mi.scores_[selector_mi.get_support()]

        print(f"   Mutual Information으로 최종 선택: {len(final_features)}개")
        print(f"   평균 MI score: {mi_scores.mean():.3f}")
    else:
        final_features = remaining_features
        print(f"\n4. 이미 목표 특성 수 이하이므로 모든 특성 사용: {len(final_features)}개")

    # 결과 요약
    print(f"\n📊 특성 선택 요약")
    print(f"   원본 특성: {original_features:4d}개")
    print(f"   분산 필터링: {len(X.columns) - len(low_variance_features):4d}개 (제거: {len(low_variance_features)}개)")
    print(f"   상관관계 필터링: {len(remaining_features) + len(highly_corr_features):4d}개 (제거: {len(highly_corr_features)}개)")
    if len(X_filtered.columns) != len(final_features):
        print(f"   ANOVA 1차: {len(X_filtered.columns):4d}개")
    print(f"   최종 선택: {len(final_features):4d}개")
    print(f"   감소율: {((original_features - len(final_features)) / original_features * 100):5.1f}%")
    print("\n✅ 선택 방법: 분산 → 상관관계 → ANOVA F-test → Mutual Information")

    return final_features


# ================================
# Pipeline
# ================================
def make_pipeline_with_scaler(model, scaling=True):
    if scaling:
        scaler = StandardScaler()
    else:
        scaler = "passthrough"

    return Pipeline([("scaler", scaler), ("model", model)])


# ================================
# ML Models
# ================================
def get_all_models():
    scaling_models = {
        'Logistic Regression': LogisticRegression(
            random_state=RANDOM_STATE,
            max_iter=1000,
            solver='lbfgs',
            C=1.0
        ),
        'K-Neighbors': KNeighborsClassifier(
            n_neighbors=5,
            weights='uniform',
            metric='minkowski',
            n_jobs=-1
        ),
        'Support Vector Machine': SVC(
            random_state=RANDOM_STATE,
            probability=True,
            C=1.0,
            gamma='scale',
            kernel='rbf'
        )
    }
    non_scaling_models = {
        'Decision Tree': DecisionTreeClassifier(
            random_state=RANDOM_STATE,
            max_depth=None,
            min_samples_split=2,
            min_samples_leaf=1,
            criterion='gini'
        ),
        'Random Forest': RandomForestClassifier(
            random_state=RANDOM_STATE,
            n_estimators=100,
            max_depth=None,
            min_samples_split=2,
            min_samples_leaf=1,
            max_features='sqrt',
            bootstrap=True,
            n_jobs=-1
        ),
        'LightGBM': LGBMClassifier(
            random_state=RANDOM_STATE,
            n_estimators=100,
            learning_rate=0.1,
            max_depth=-1,
            num_leaves=31,
            subsample=1.0,
            colsample_bytree=1.0,
            reg_alpha=0.0,
            reg_lambda=0.0,
            n_jobs=-1,
            verbosity=-1,
            force_col_wise=True
        ),
        'XGBoost': XGBClassifier(
            random_state=RANDOM_STATE,
            n_estimators=100,
            learning_rate=0.1,
            max_depth=6,
            min_child_weight=1,
            subsample=1.0,
            colsample_bytree=1.0,
            reg_alpha=0,
            reg_lambda=1,
            eval_metric="logloss",
            n_jobs=-1,
            verbosity=0,
            use_label_encoder=False  # 경고 방지
        ),
        'CatBoost': CatBoostClassifier(
            random_state=RANDOM_STATE,
            iterations=100,
            learning_rate=0.1,
            depth=6,
            l2_leaf_reg=3.0,
            bootstrap_type='Bayesian',
            bagging_temperature=1.0,
            od_type='IncToDec',
            od_wait=20,
            verbose=False,
            allow_writing_files=False
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
            'model__C': [0.01, 0.1, 1, 10, 100],
            'model__solver': ['liblinear', 'lbfgs'],
        },
        'K-Neighbors': {
            'model__n_neighbors': [3, 5, 7, 11, 15],
            'model__weights': ['uniform', 'distance']
        },
        'Support Vector Machine': {
            'model__C': [0.1, 1, 10],
            'model__kernel': ['rbf', 'linear'],
            'model__gamma': ['scale', 'auto']
        },
        'Decision Tree': {
            'model__max_depth': [3, 5, 10, None],
            'model__min_samples_split': [2, 5, 10],
            'model__criterion': ['gini', 'entropy']
        },
        'Random Forest': {
            'model__n_estimators': [50, 100, 200],
            'model__max_depth': [5, 10, None],
            'model__min_samples_split': [2, 5]
        },
        'LightGBM': {
            'model__n_estimators': [50, 100, 200],
            'model__learning_rate': [0.05, 0.1, 0.2],
            'model__max_depth': [3, 5, 7]
        },
        'XGBoost': {
            'model__n_estimators': [50, 100, 200],
            'model__learning_rate': [0.05, 0.1, 0.2],
            'model__max_depth': [3, 5, 7]
        },
        'CatBoost': {
            'model__iterations': [50, 100, 200],
            'model__learning_rate': [0.05, 0.1, 0.2],
            'model__depth': [3, 4, 5]
        }
    }


def grid_search_tuning(X_train, y_train):
    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=RANDOM_STATE)
    tuned = {}
    params = get_grid_search_params()
    models = get_all_models()

    for name, model in models.items():
        print(f"- {name} 튜닝")
        if name in params:
            search = GridSearchCV(
                model, params[name],
                cv=cv, scoring='f1',
                n_jobs=-1, verbose=0
            )
            search.fit(X_train, y_train)
            tuned[name] = search.best_estimator_
            print(f"     Best params: {search.best_params_}")
            print(f"     Best CV F1: {search.best_score_:.4f}")
        else:
            tuned[name] = model.fit(X_train, y_train)

    return tuned


# ================================
# Optuna
# ================================
def create_optuna_objective(model_name, X_train, y_train):
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
            if params["model__kernel"] == "rbf":
                params["model__gamma"] = trial.suggest_categorical("model__gamma", ["scale", "auto"])
            pipeline = make_pipeline_with_scaler(model, scaling=True).set_params(**params)

        elif model_name == "Decision Tree":
            model = DecisionTreeClassifier(random_state=RANDOM_STATE)
            params = {
                "model__max_depth": trial.suggest_categorical("model__max_depth", [3, 5, 10, None]),
                "model__min_samples_split": trial.suggest_int("model__min_samples_split", 2, 10),
                "model__criterion": trial.suggest_categorical("model__criterion", ["gini", "entropy"])
            }
            pipeline = make_pipeline_with_scaler(model, scaling=False).set_params(**params)

        elif model_name == "Random Forest":
            model = RandomForestClassifier(random_state=RANDOM_STATE, n_jobs=-1)
            params = {
                "model__n_estimators": trial.suggest_int("model__n_estimators", 50, 300),
                "model__max_depth": trial.suggest_categorical("model__max_depth", [5, 10, None]),
                "model__min_samples_split": trial.suggest_int("model__min_samples_split", 2, 5)
            }
            pipeline = make_pipeline_with_scaler(model, scaling=False).set_params(**params)

        elif model_name == "LightGBM":
            model = LGBMClassifier(random_state=RANDOM_STATE, n_jobs=-1, verbosity=-1)
            params = {
                "model__n_estimators": trial.suggest_int("model__n_estimators", 50, 300),
                "model__learning_rate": trial.suggest_float("model__learning_rate", 0.01, 0.3),
                "model__num_leaves": trial.suggest_int("model__num_leaves", 15, 63),
                "model__max_depth": trial.suggest_int("model__max_depth", 3, 7)
            }
            pipeline = make_pipeline_with_scaler(model, scaling=False).set_params(**params)

        elif model_name == "XGBoost":
            model = XGBClassifier(random_state=RANDOM_STATE, eval_metric="logloss", n_jobs=-1, verbosity=0)
            params = {
                "model__n_estimators": trial.suggest_int("model__n_estimators", 50, 300),
                "model__learning_rate": trial.suggest_float("model__learning_rate", 0.01, 0.3),
                "model__max_depth": trial.suggest_int("model__max_depth", 3, 7),
                "model__subsample": trial.suggest_float("model__subsample", 0.6, 1.0)
            }
            pipeline = make_pipeline_with_scaler(model, scaling=False).set_params(**params)

        elif model_name == "CatBoost":
            model = CatBoostClassifier(random_state=RANDOM_STATE, verbose=False)
            params = {
                "model__depth": trial.suggest_int("model__depth", 4, 10),
                "model__iterations": trial.suggest_int("model__iterations", 100, 300),
                "model__learning_rate": trial.suggest_float("model__learning_rate", 0.01, 0.3)
            }
            pipeline = make_pipeline_with_scaler(model, scaling=False).set_params(**params)

        else:
            raise ValueError(f"지원하지 않는 모델: {model_name}")

        return run_cv(pipeline)

    return objective


def optuna_tuning(X_train, y_train, n_trials=30):
    optimized_models = {}
    optuna_results = []
    base_models = get_all_models()

    for name in base_models.keys():
        print(f"- {name} 최적화")
        objective = create_optuna_objective(name, X_train, y_train)
        study = optuna.create_study(
            direction="maximize",
            sampler=TPESampler(seed=RANDOM_STATE),
            pruner=MedianPruner()
        )
        study.optimize(objective, n_trials=n_trials, show_progress_bar=False)

        best_params = study.best_params
        optimized_models[name] = base_models[name].set_params(**best_params)
        optuna_results.append({
            'Model': name,
            'Best Params': str(best_params),
            'Best CV F1': study.best_value
        })
        print(f"     Best params: {study.best_params}")
        print(f"     Best CV F1: {study.best_value:.4f}")

    # Optuna 결과 저장
    os.makedirs('./result', exist_ok=True)
    pd.DataFrame(optuna_results).to_excel('./result/results_optuna_details.xlsx', index=False)
    print("   Optuna 세부 결과 저장: ./result/results_optuna_details.xlsx")

    return optimized_models


# ================================
# ML Evaluation
# ================================
def compute_metrics(y_true, y_pred, y_proba=None):
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel() if cm.size == 4 else (0, 0, 0, 0)

    metrics = {
        'Accuracy': (y_true == y_pred).mean(),
        'Precision': precision_score(y_true, y_pred, zero_division=0),
        'Recall': recall_score(y_true, y_pred, zero_division=0),
        'F1': f1_score(y_true, y_pred, zero_division=0),
        'Balanced_Accuracy': balanced_accuracy_score(y_true, y_pred),
        'Specificity': tn / (tn + fp) if (tn + fp) > 0 else 0,
        'Sensitivity': tp / (tp + fn) if (tp + fn) > 0 else 0,
        'MCC': matthews_corrcoef(y_true, y_pred),
        'AUC': roc_auc_score(y_true, y_proba) if y_proba is not None else 0
    }
    return metrics


# ================================
# Cross Validation for All Methods
# ================================
def evaluate_models_with_cv(models, X_train, y_train, cv_folds=5):
    """모든 모델에 대해 동일한 교차검증 수행"""
    cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=RANDOM_STATE)
    cv_results = {}

    for name, model in models.items():
        print(f"- {name} CV 평가")

        # 주요 메트릭들에 대해 교차검증 수행
        accuracy_scores = cross_val_score(model, X_train, y_train, cv=cv, scoring='accuracy', n_jobs=-1)
        f1_scores = cross_val_score(model, X_train, y_train, cv=cv, scoring='f1', n_jobs=-1)
        auc_scores = cross_val_score(model, X_train, y_train, cv=cv, scoring='roc_auc', n_jobs=-1)

        cv_results[name] = {
            'CV_Accuracy_Mean': accuracy_scores.mean(),
            'CV_Accuracy_Std': accuracy_scores.std(),
            'CV_F1_Mean': f1_scores.mean(),
            'CV_F1_Std': f1_scores.std(),
            'CV_AUC_Mean': auc_scores.mean(),
            'CV_AUC_Std': auc_scores.std()
        }

        print(f"     CV Accuracy: {accuracy_scores.mean():.4f} (±{accuracy_scores.std():.4f})")
        print(f"     CV F1: {f1_scores.mean():.4f} (±{f1_scores.std():.4f})")
        print(f"     CV AUC: {auc_scores.mean():.4f} (±{auc_scores.std():.4f})")

    return cv_results


# ================================
# Visualization
# ================================
def plot_f1_comparison(results_df, tuning_method):
    subset = results_df[results_df['Tuning'] == tuning_method]
    subset_sorted = subset.sort_values('F1', ascending=False)

    plt.figure(figsize=(10, 6))
    bars = plt.barh(subset_sorted['Model'], subset_sorted['F1'], color=sns.color_palette("Set2", len(subset_sorted)))

    for bar, f1 in zip(bars, subset_sorted['F1']):
        plt.text(bar.get_width() + 0.005, bar.get_y() + bar.get_height() / 2, f"{f1:.3f}", va="center", fontsize=10)

    plt.title(f"F1 Scores ({tuning_method.upper()})", fontsize=14, fontweight='bold')
    plt.xlabel("F1 Score", fontsize=12)
    plt.xlim(0, max(subset_sorted['F1']) * 1.1)
    plt.grid(True, axis='x', linestyle='-', alpha=0.5)
    plt.tight_layout()
    plt.show()


def plot_roc_comparison(results, y_test, tuning_method):
    plt.figure(figsize=(10, 6))
    subset = [r for r in results if r['Tuning'] == tuning_method]
    colors = sns.color_palette("Set2", len(subset))

    for i, r in enumerate(subset):
        if r['y_proba'] is not None and len(np.unique(r['y_proba'])) > 1:
            fpr, tpr, _ = roc_curve(y_test, r['y_proba'])
            plt.plot(fpr, tpr, label=f"{r['Model']} (AUC={r['AUC']:.3f})", color=colors[i], linewidth=2)

    plt.plot([0, 1], [0, 1], 'k--', linewidth=1)
    plt.title(f"ROC Curves ({tuning_method.upper()})", fontsize=14, fontweight='bold')
    plt.xlabel("False Positive Rate", fontsize=12)
    plt.ylabel("True Positive Rate", fontsize=12)
    plt.legend(loc="lower right")
    plt.grid(True, linestyle='-', alpha=0.5)
    plt.tight_layout()
    plt.show()


def plot_comprehensive_comparison(results_df):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))

    # F1 Score 비교
    tuning_methods = results_df['Tuning'].unique()
    models = results_df['Model'].unique()

    x = np.arange(len(models))
    width = 0.25

    for i, tuning in enumerate(tuning_methods):
        subset = results_df[results_df['Tuning'] == tuning].set_index('Model')
        f1_scores = [subset.loc[model, 'F1'] if model in subset.index else 0 for model in models]
        ax1.bar(x + i * width, f1_scores, width, label=tuning.upper(), alpha=0.8)

    ax1.set_ylabel('F1 Score')
    ax1.set_title('F1 Score Comparison by Tuning Method')
    ax1.set_xticks(x + width)
    ax1.set_xticklabels(models, rotation=45, ha='right')
    ax1.legend()
    ax1.grid(True, linestyle='-', alpha=0.5)

    # AUC 비교
    for i, tuning in enumerate(tuning_methods):
        subset = results_df[results_df['Tuning'] == tuning].set_index('Model')
        auc_scores = [subset.loc[model, 'AUC'] if model in subset.index else 0 for model in models]
        ax2.bar(x + i * width, auc_scores, width, label=tuning.upper(), alpha=0.8)

    ax2.set_ylabel('AUC Score')
    ax2.set_title('AUC Score Comparison by Tuning Method')
    ax2.set_xticks(x + width)
    ax2.set_xticklabels(models, rotation=45, ha='right')
    ax2.legend()
    ax2.grid(True, linestyle='-', alpha=0.5)

    plt.tight_layout()
    plt.show()


# ================================
# Save Results
# ================================
def save_results_by_tuning(results_df, y_test):
    os.makedirs('./result', exist_ok=True)

    # 전체 결과 저장
    results_df_save = results_df.drop(['y_pred', 'y_proba'], axis=1)
    results_df_save.to_excel('./result/final_results.xlsx', index=False)
    print("\n📊 전체 결과 저장: ./result/final_results.xlsx")

    # 튜닝별 저장 및 시각화
    all_results = results_df.to_dict('records')

    for tuning in results_df['Tuning'].unique():
        subset = results_df[results_df['Tuning'] == tuning].drop(['y_pred', 'y_proba'], axis=1)
        subset.to_excel(f'./result/results_{tuning.upper()}.xlsx', index=False)
        print(f"   {tuning.upper()} 결과 저장: ./result/results_{tuning.upper()}.xlsx")

        # f1, ROC 시각화
        plot_f1_comparison(results_df, tuning)
        plot_roc_comparison(all_results, y_test, tuning)

    # 통합 비교 시각화
    plot_comprehensive_comparison(results_df)

    # 최고 성능 모델 저장
    best = results_df.sort_values('F1', ascending=False).iloc[0]
    best_save = best.drop(['y_pred', 'y_proba'])
    pd.DataFrame([best_save]).to_excel('./result/best_model.xlsx', index=False)
    print(f"\n🏆 최고 성능 모델: {best['Model']} ({best['Tuning']}) | F1={best['F1']:.4f} | AUC={best['AUC']:.4f}")


# ================================
# Main
# ================================
def run_all():
    print("🚀 머신러닝 분류 파이프라인 시작 \n")

    # 1. 데이터 로드 및 분할
    df_all, X_train, X_test, y_train, y_test, feature_cols = data_processing()

    # 클래스 균형 확인
    class_0_train = (y_train == 0).sum()
    class_1_train = (y_train == 1).sum()
    class_0_test = (y_test == 0).sum()
    class_1_test = (y_test == 1).sum()

    print(f"\n📊 클래스 분포 확인")
    print(f"   Train: Class 0={class_0_train}개, Class 1={class_1_train}개")
    print(f"   Test:  Class 0={class_0_test}개, Class 1={class_1_test}개")
    print(f"   균형도: {min(class_0_train, class_1_train) / max(class_0_train, class_1_train):.3f} (Train)")

    # 2. 특성 선택 (Train 데이터만 사용 - Test 오염 방지)
    selected_features = feature_selection(X_train, y_train, final_k=50)

    # Test 데이터에서 동일한 특성만 선택 (정보 유출 없음)
    X_train_selected = X_train[selected_features]
    X_test_selected = X_test[selected_features]

    all_results = []
    all_cv_results = []
    tuning_methods = [TuningMethod.DEFAULT, TuningMethod.GRID_SEARCH, TuningMethod.OPTUNA]

    for tuning_method in tuning_methods:
        print(f"\n{'=' * 60}")
        print(f"🎯 {tuning_method.upper()} 실행")
        print(f"{'=' * 60}")

        # 3. 모델 학습 (각 방법별)
        if tuning_method == TuningMethod.DEFAULT:
            models = get_all_models()

        elif tuning_method == TuningMethod.GRID_SEARCH:
            models = grid_search_tuning(X_train_selected, y_train)

        elif tuning_method == TuningMethod.OPTUNA:
            models = optuna_tuning(X_train_selected, y_train, n_trials=30)

        # 4. 교차검증 평가 (Train 데이터로만, 모든 방법 동일)
        print(f"📊 {tuning_method.upper()} 교차검증 평가 (5-Fold CV, Train 데이터)")
        cv_results = evaluate_models_with_cv(models, X_train_selected, y_train, cv_folds=5)

        # CV 결과를 전체 결과에 추가
        for model_name, cv_scores in cv_results.items():
            cv_result = {
                'Tuning': tuning_method,
                'Model': model_name,
                **cv_scores
            }
            all_cv_results.append(cv_result)

        # 6. 최종 테스트 평가 (Test 데이터, 한 번만)
        print(f"\n📊 {tuning_method.upper()} 최종 테스트 평가:")
        for name, model in models.items():
            try:
                preds = model.predict(X_test_selected)

                if hasattr(model, "predict_proba"):
                    proba = model.predict_proba(X_test_selected)[:, 1]
                elif hasattr(model, "decision_function"):
                    proba = model.decision_function(X_test_selected)
                else:
                    proba = None

                metrics = compute_metrics(y_test, preds, proba)

                result = {
                    'Tuning': tuning_method,
                    'Model': name,
                    'y_pred': preds,
                    'y_proba': proba,
                    **metrics
                }
                all_results.append(result)

                # 균형 데이터이므로 Accuracy를 주 메트릭으로 출력
                print(
                    f"   {name:20s} | Accuracy: {metrics['Accuracy']:.4f} | F1: {metrics['F1']:.4f} | AUC: {metrics['AUC']:.4f}")

            except Exception as e:
                print(f"   ❌ {name} 평가 실패: {str(e)}")

    # 7. 결과 정리 및 저장
    print(f"\n{'=' * 60}")
    print("📈 최종 결과 정리")
    print(f"{'=' * 60}")

    results_df = pd.DataFrame(all_results)
    cv_results_df = pd.DataFrame(all_cv_results)

    # 균형 데이터이므로 Accuracy 기준으로 순위 출력
    print(f"\n🏅 교차검증 성능 순위 (CV Accuracy 기준)")
    cv_display_df = cv_results_df[['Tuning', 'Model', 'CV_Accuracy_Mean', 'CV_F1_Mean', 'CV_AUC_Mean']].sort_values(
        'CV_Accuracy_Mean', ascending=False)
    print(cv_display_df.to_string(index=False, float_format='%.4f'))

    print(f"\n🏅 최종 테스트 성능 순위 (Test Accuracy 기준)")
    display_df = results_df[['Tuning', 'Model', 'Accuracy', 'F1', 'AUC']].sort_values('Accuracy', ascending=False)
    print(display_df.to_string(index=False, float_format='%.4f'))

    # 8. 결과 저장 및 시각화
    save_results_by_tuning(results_df, y_test)

    # CV 결과도 별도 저장
    cv_results_df.to_excel('./result/cross_validation_results.xlsx', index=False)
    print("📊 교차검증 결과 저장: ./result/cross_validation_results.xlsx")

    # 최종 추천 모델 (Test Accuracy 기준)
    best_model = results_df.sort_values('Accuracy', ascending=False).iloc[0]
    print(f"\n🏆 최종 추천 모델: {best_model['Model']} ({best_model['Tuning']})")
    print(f"   Test Accuracy: {best_model['Accuracy']:.4f} ⭐")
    print(f"   Test F1: {best_model['F1']:.4f} | Test AUC: {best_model['AUC']:.4f}")

    print(f"\n✅ 모든 작업 완료!")


if __name__ == "__main__":
    run_all()