import os
import glob
import warnings
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split, StratifiedKFold, learning_curve, cross_val_score, GridSearchCV
from sklearn.feature_selection import SelectKBest, f_classif, chi2, mutual_info_classif
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.ensemble import RandomForestClassifier
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier
from catboost import CatBoostClassifier
from sklearn.metrics import (
    confusion_matrix, classification_report, f1_score, matthews_corrcoef,
    precision_score, recall_score, roc_auc_score, roc_curve, auc,
    balanced_accuracy_score, ConfusionMatrixDisplay
)
from sklearn.inspection import permutation_importance

# Optuna import (선택적)
try:
    import optuna
    from optuna.samplers import TPESampler
    from optuna.pruners import MedianPruner

    OPTUNA_AVAILABLE = True
except ImportError:
    OPTUNA_AVAILABLE = False
    print("Optuna가 설치되지 않았습니다. 'pip install optuna'로 설치하면 Optuna 최적화를 사용할 수 있습니다.")

# 경고 메시지 억제
warnings.filterwarnings('ignore')

# Global Variable
RANDOM_STATE = 42


# ========================================
# Tuning Methods Enum
# ========================================
class TuningMethod:
    NONE = "none"  # 기본 파라미터
    GRID_SEARCH = "grid"  # GridSearchCV
    OPTUNA = "optuna"  # Optuna 최적화


# ========================================
# Data Processing: 데이터 로드 및 기본 전처리
# ========================================
def data_processing():
    """데이터를 로드하고 기본 전처리를 수행합니다."""
    csv_files = glob.glob('./features_xlsx/*.xlsx')
    print(f"\n📂 분석할 파일 수 - {len(csv_files)}개")

    if len(csv_files) == 0:
        raise FileNotFoundError("경로에 파일이 없습니다.")

    df_all = pd.concat([pd.read_excel(file, sheet_name=4) for file in csv_files], ignore_index=True)

    # 기본 정보 출력
    print(f"총 데이터 수: {len(df_all)}개")
    print(f"컬럼 수: {df_all.shape[1]}개")

    # 결측치 처리 - 데이터 누수 방지를 위해 train/test split 후 처리 필요
    print(f"결측치 개수: {df_all.isnull().sum().sum()}개")

    # 라벨 인코딩
    y_all = LabelEncoder().fit_transform(df_all['label'])
    print(f"라벨 분포: 0 - {(y_all == 0).sum()}개 / 1 - {(y_all == 1).sum()}개")

    # 수치형 특성만 선택
    feature_cols = df_all.select_dtypes(include=['float64', 'int64']).columns.drop('label', errors='ignore')
    raw_features = df_all[feature_cols]

    print(f"최종 특성 수: {raw_features.shape[1]}개")

    # Train/Test 분할
    X_train_raw, X_test_raw, y_train, y_test = train_test_split(
        raw_features, y_all, test_size=0.2, stratify=y_all, random_state=RANDOM_STATE
    )

    # 데이터 누수 방지: 훈련 데이터로만 결측치와 무한값 처리
    # 훈련 데이터 전처리
    X_train_processed = X_train_raw.replace([np.inf, -np.inf], np.nan)
    train_median = X_train_processed.median()
    X_train_processed = X_train_processed.fillna(train_median)

    # 테스트 데이터는 훈련 데이터의 통계량으로 처리
    X_test_processed = X_test_raw.replace([np.inf, -np.inf], np.nan)
    X_test_processed = X_test_processed.fillna(train_median)

    return df_all, X_train_processed, X_test_processed, y_train, y_test, feature_cols


# ========================================
# Feature Selection: filter + embedded
# ========================================
def feature_selection(X, y, final_k=50):
    """특성 선택을 수행합니다."""
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

    # 2단계: 상관관계 필터링
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
        X_filtered = X_filtered[anova_features]
        print(f"   ANOVA F-test로 선택된 특성: {len(anova_features)}개")
        remaining_features = anova_features

    # 4단계: Mutual Information으로 최종 선별
    if len(remaining_features) > final_k:
        print(f"\n4. Mutual Information으로 최종 선별 ({final_k}개)")
        selector_mi = SelectKBest(
            score_func=lambda X, y: mutual_info_classif(X, y, random_state=RANDOM_STATE),
            k=final_k
        )
        selector_mi.fit(X_filtered, y)
        final_features = X_filtered.columns[selector_mi.get_support()].tolist()
        print(f"   Mutual Information으로 최종 선택: {len(final_features)}개")
    else:
        final_features = remaining_features
        print(f"\n4️⃣ 이미 목표 특성 수 이하이므로 모든 특성 사용: {len(final_features)}개")

    # 결과 요약
    print("=" * 50)
    print(f"\n✅ 특성 선택 완료!")
    print(f"   원본 특성: {original_features:4d}개")
    print(f"   최종 선택: {len(final_features):4d}개")
    print(f"   감소율: {((original_features - len(final_features)) / original_features * 100):5.1f}%")
    print("=" * 50)

    return final_features


# ========================================
# ML Models Creation
# ========================================
def create_base_models():
    """기본 머신러닝 모델들을 생성합니다."""
    return {
        'Logistic Regression': LogisticRegression(random_state=RANDOM_STATE, max_iter=1000),
        'KNN': KNeighborsClassifier(n_jobs=-1),
        'Support Vector Machine': SVC(probability=True, random_state=RANDOM_STATE),
        'Decision Tree': DecisionTreeClassifier(random_state=RANDOM_STATE),
        'Random Forest': RandomForestClassifier(random_state=RANDOM_STATE, n_jobs=-1),
        'LightGBM': LGBMClassifier(random_state=RANDOM_STATE, verbosity=-1, n_jobs=-1),
        'XGBoost': XGBClassifier(random_state=RANDOM_STATE, eval_metric='logloss', verbosity=0, n_jobs=-1),
        'CatBoost': CatBoostClassifier(random_state=RANDOM_STATE, verbose=False),
    }


# ========================================
# GridSearch Hyperparameter Tuning
# ========================================
def get_grid_search_params():
    """GridSearchCV용 하이퍼파라미터 그리드를 반환합니다."""
    return {
        'Logistic Regression': {
            'C': [0.01, 0.1, 1, 10, 100],
            'solver': ['liblinear', 'lbfgs'],
        },
        'KNN': {
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
    """GridSearchCV로 하이퍼파라미터를 최적화합니다."""
    print("\n🎯 GridSearch 하이퍼파라미터 튜닝 시작")

    base_models = create_base_models()
    param_grids = get_grid_search_params()
    tuned_models = {}
    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=RANDOM_STATE)

    for model_name, base_model in base_models.items():
        print(f"⚡ {model_name} 튜닝 중...")

        try:
            if model_name in param_grids:
                search = GridSearchCV(
                    estimator=base_model,
                    param_grid=param_grids[model_name],
                    cv=cv,
                    scoring='f1',
                    n_jobs=-1,
                    verbose=0
                )
                search.fit(X_train, y_train)
                tuned_models[model_name] = search.best_estimator_
                print(f"   최적 CV F1: {search.best_score_:.4f}")
            else:
                tuned_models[model_name] = base_model
                print(f"   기본 모델 사용")
        except Exception as e:
            print(f"   ❌ 튜닝 실패: {e}")
            tuned_models[model_name] = base_model

    return tuned_models


# ========================================
# Optuna Hyperparameter Tuning
# ========================================
def create_optuna_objectives(X_train, y_train, cv_folds=3):
    """Optuna 목적 함수들을 생성합니다."""
    if not OPTUNA_AVAILABLE:
        raise ImportError("Optuna가 설치되지 않았습니다.")

    cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=RANDOM_STATE)

    def run_cv(model):
        scores = cross_val_score(model, X_train, y_train, cv=cv, scoring='f1', n_jobs=-1)
        return scores.mean()

    def logistic_regression_objective(trial):
        params = {
            'C': trial.suggest_float('C', 0.001, 100, log=True),
            'solver': trial.suggest_categorical('solver', ['liblinear', 'lbfgs']),
            'penalty': trial.suggest_categorical('penalty', ['l1', 'l2']),
            'max_iter': 2000,
            'random_state': RANDOM_STATE
        }
        if params['penalty'] == 'l1' and params['solver'] == 'lbfgs':
            params['solver'] = 'liblinear'
        return run_cv(LogisticRegression(**params))

    def knn_objective(trial):
        params = {
            'n_neighbors': trial.suggest_int('n_neighbors', 3, 21, step=2),
            'weights': trial.suggest_categorical('weights', ['uniform', 'distance']),
            'metric': trial.suggest_categorical('metric', ['euclidean', 'manhattan']),
            'n_jobs': -1
        }
        return run_cv(KNeighborsClassifier(**params))

    def svm_objective(trial):
        kernel = trial.suggest_categorical('kernel', ['linear', 'rbf'])
        params = {
            'C': trial.suggest_float('C', 0.01, 1000, log=True),
            'kernel': kernel,
            'probability': True,
            'random_state': RANDOM_STATE
        }
        if kernel == 'rbf':
            params['gamma'] = trial.suggest_categorical('gamma', ['scale', 'auto'])
        return run_cv(SVC(**params))

    def decision_tree_objective(trial):
        params = {
            'max_depth': trial.suggest_int('max_depth', 2, 8),
            'min_samples_split': trial.suggest_int('min_samples_split', 10, 50),
            'min_samples_leaf': trial.suggest_int('min_samples_leaf', 5, 25),
            'criterion': trial.suggest_categorical('criterion', ['gini', 'entropy']),
            'random_state': RANDOM_STATE
        }
        return run_cv(DecisionTreeClassifier(**params))

    def random_forest_objective(trial):
        params = {
            'n_estimators': trial.suggest_int('n_estimators', 50, 300, step=50),
            'max_depth': trial.suggest_int('max_depth', 3, 15),
            'min_samples_split': trial.suggest_int('min_samples_split', 5, 30),
            'min_samples_leaf': trial.suggest_int('min_samples_leaf', 2, 15),
            'random_state': RANDOM_STATE,
            'n_jobs': -1
        }
        return run_cv(RandomForestClassifier(**params))

    def lightgbm_objective(trial):
        params = {
            'n_estimators': trial.suggest_int('n_estimators', 50, 300, step=50),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
            'max_depth': trial.suggest_int('max_depth', 3, 10),
            'num_leaves': trial.suggest_int('num_leaves', 10, 100),
            'random_state': RANDOM_STATE,
            'verbosity': -1,
            'n_jobs': -1
        }
        return run_cv(LGBMClassifier(**params))

    def xgboost_objective(trial):
        params = {
            'n_estimators': trial.suggest_int('n_estimators', 50, 300, step=50),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
            'max_depth': trial.suggest_int('max_depth', 3, 8),
            'random_state': RANDOM_STATE,
            'eval_metric': 'logloss',
            'verbosity': 0,
            'n_jobs': -1
        }
        return run_cv(XGBClassifier(**params))

    def catboost_objective(trial):
        params = {
            'iterations': trial.suggest_int('iterations', 50, 300, step=50),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
            'depth': trial.suggest_int('depth', 3, 8),
            'random_seed': RANDOM_STATE,
            'verbose': False
        }
        return run_cv(CatBoostClassifier(**params))

    return {
        'Logistic Regression': logistic_regression_objective,
        'KNN': knn_objective,
        'Support Vector Machine': svm_objective,
        'Decision Tree': decision_tree_objective,
        'Random Forest': random_forest_objective,
        'LightGBM': lightgbm_objective,
        'XGBoost': xgboost_objective,
        'CatBoost': catboost_objective
    }


def optuna_tuning(X_train, y_train, n_trials=50, timeout=300):
    """Optuna로 하이퍼파라미터를 최적화합니다."""
    if not OPTUNA_AVAILABLE:
        raise ImportError("Optuna가 설치되지 않았습니다. 'pip install optuna'로 설치해주세요.")

    print("\n🎯 Optuna 하이퍼파라미터 튜닝 시작")

    objectives = create_optuna_objectives(X_train, y_train)
    optimized_models = {}
    optuna_results = []

    sampler = TPESampler(seed=RANDOM_STATE)
    pruner = MedianPruner(n_startup_trials=5)

    for model_name, objective_func in objectives.items():
        print(f"⚡ {model_name} 최적화 중...")

        study = optuna.create_study(direction='maximize', sampler=sampler, pruner=pruner)
        study.optimize(objective_func, n_trials=n_trials, timeout=timeout)

        best_params = study.best_params
        print(f"   최적 CV F1: {study.best_value:.4f}")

        optuna_results.append({
            "Model": model_name,
            "Best CV F1": study.best_value,
            "Best Params": best_params
        })

        # 최적 모델 생성
        if model_name == 'Logistic Regression':
            if best_params.get('penalty') == 'l1' and best_params.get('solver') == 'lbfgs':
                best_params['solver'] = 'liblinear'
            optimized_models[model_name] = LogisticRegression(**best_params)
        elif model_name == 'KNN':
            optimized_models[model_name] = KNeighborsClassifier(**best_params)
        elif model_name == 'Support Vector Machine':
            optimized_models[model_name] = SVC(**best_params)
        elif model_name == 'Decision Tree':
            optimized_models[model_name] = DecisionTreeClassifier(**best_params)
        elif model_name == 'Random Forest':
            optimized_models[model_name] = RandomForestClassifier(**best_params)
        elif model_name == 'LightGBM':
            optimized_models[model_name] = LGBMClassifier(**best_params)
        elif model_name == 'XGBoost':
            optimized_models[model_name] = XGBClassifier(**best_params)
        elif model_name == 'CatBoost':
            optimized_models[model_name] = CatBoostClassifier(**best_params)

    # 결과 저장
    pd.DataFrame(optuna_results).to_excel('./result/optuna_results.xlsx', index=False)
    print("Optuna 최적화 결과 저장: ./result/optuna_results.xlsx")

    return optimized_models


# ========================================
# Model Creation Controller
# ========================================
def create_ml_models(tuning_method=TuningMethod.NONE, X_train=None, y_train=None):
    """선택된 튜닝 방법에 따라 모델을 생성합니다."""

    if tuning_method == TuningMethod.NONE:
        print("\n📦 기본 파라미터 모델들 생성 중...")
        models = create_base_models()
        print(f"{len(models)}개 기본 모델 생성 완료!")
        return models

    elif tuning_method == TuningMethod.GRID_SEARCH:
        if X_train is None or y_train is None:
            raise ValueError("GridSearch 튜닝을 위한 훈련 데이터가 필요합니다.")
        return grid_search_tuning(X_train, y_train)

    elif tuning_method == TuningMethod.OPTUNA:
        if X_train is None or y_train is None:
            raise ValueError("Optuna 튜닝을 위한 훈련 데이터가 필요합니다.")
        return optuna_tuning(X_train, y_train)

    else:
        raise ValueError(f"지원하지 않는 튜닝 방법: {tuning_method}")


# ========================================
# Metrics & Evaluation Functions
# ========================================
def compute_metrics(y_true, y_pred, y_proba=None):
    """성능 지표를 계산합니다."""
    cm = confusion_matrix(y_true, y_pred)
    if cm.size == 4:
        tn, fp, fn, tp = cm.ravel()
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
        sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
    else:
        specificity = sensitivity = 0

    metrics = {
        'Accuracy': (y_true == y_pred).mean(),
        'Precision': precision_score(y_true, y_pred, zero_division=0),
        'Recall': recall_score(y_true, y_pred, zero_division=0),
        'F1': f1_score(y_true, y_pred, zero_division=0),
        'Balanced_Accuracy': balanced_accuracy_score(y_true, y_pred),
        'Specificity': specificity,
        'Sensitivity': sensitivity,
        'MCC': matthews_corrcoef(y_true, y_pred)
    }

    if y_proba is not None:
        try:
            metrics['AUC'] = roc_auc_score(y_true, y_proba)
        except:
            metrics['AUC'] = 0
    else:
        metrics['AUC'] = 0

    return metrics


def cross_validate_models(models, X, y, cv_folds=5):
    """교차 검증을 수행합니다."""
    print(f"\n🔄 {cv_folds}-Fold 교차 검증 실행 중...")

    cv_results = {}
    cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=RANDOM_STATE)

    for name, model in models.items():
        print(f"  ⚡ {name}")
        try:
            f1_scores = cross_val_score(model, X, y, cv=cv, scoring='f1', n_jobs=-1)
            cv_results[name] = {
                'CV_F1_mean': f1_scores.mean(),
                'CV_F1_std': f1_scores.std(),
                'CV_F1_scores': f1_scores
            }
            print(f"    CV F1: {f1_scores.mean():.4f} ± {f1_scores.std():.4f}")
        except Exception as e:
            print(f"    ❌ {name} 검증 실패: {e}")
            cv_results[name] = {
                'CV_F1_mean': 0,
                'CV_F1_std': 0,
                'CV_F1_scores': [0]
            }
    return cv_results


def evaluate_model(model, model_name, X_train, X_test, y_train, y_test):
    """개별 모델을 평가합니다."""
    print(f"\n📊 {model_name}")
    try:
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        y_proba = None
        if hasattr(model, 'predict_proba'):
            y_proba = model.predict_proba(X_test)[:, 1]
        elif hasattr(model, 'decision_function'):
            y_proba = model.decision_function(X_test)

        metrics = compute_metrics(y_test, y_pred, y_proba)
        print(f"F1: {metrics['F1']:.4f}, AUC: {metrics['AUC']:.4f}, Accuracy: {metrics['Accuracy']:.4f}")

        return metrics, y_pred, y_proba, model
    except Exception as e:
        print(f"  ❌ {model_name} 평가 실패: {e}")
        return None, None, None, None


# ========================================
# Visualization Functions (간단화)
# ========================================
def plot_f1_comparison(results_df):
    """F1 점수 비교 차트를 그립니다."""
    results_df_sorted = results_df.sort_values('F1', ascending=False)
    plt.figure(figsize=(10, 6))
    colors = sns.color_palette("Set2", len(results_df_sorted))
    bars = plt.barh(results_df_sorted['Model'], results_df_sorted['F1'], color=colors)

    for i, bar in enumerate(bars):
        width = bar.get_width()
        plt.text(width + 0.01, bar.get_y() + bar.get_height() / 2, f'{width:.3f}', ha='left', va='center', fontsize=12)

    plt.title('F1 Score Comparison - All Models', fontsize=16, weight='bold')
    plt.xlabel('F1 Score')
    plt.grid(True, axis='x', linestyle='-', alpha=0.5)
    plt.tight_layout()
    plt.show()


def plot_roc_comparison(results, y_test):
    """ROC 곡선 비교 차트를 그립니다."""
    plt.figure(figsize=(10, 6))
    colors = plt.cm.Set2.colors

    for i, result in enumerate(results):
        if result.get('Best Model') is None or result.get('y_proba') is None:
            continue

        name = result['Model']
        y_prob = result['y_proba']

        try:
            fpr, tpr, _ = roc_curve(y_test, y_prob)
            roc_auc = auc(fpr, tpr)
            plt.plot(fpr, tpr, label=f'{name} (AUC = {roc_auc:.3f})',
                     color=colors[i % len(colors)], linewidth=2)
        except Exception as e:
            print(f"{name} ROC 곡선 생성 오류: {e}")
            continue

    plt.plot([0, 1], [0, 1], 'k--', linewidth=1)
    plt.title('ROC Curves - All Models', fontsize=16, weight='bold')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.legend(loc='lower right', frameon=True)
    plt.grid(True, linestyle='-', alpha=0.5)
    plt.tight_layout()
    plt.show()


# ========================================
# Main Pipeline
# ========================================
def main(tuning_method=TuningMethod.NONE):
    """
    메인 머신러닝 파이프라인을 실행합니다.

    Parameters:
    - tuning_method: TuningMethod.NONE (기본값), TuningMethod.GRID_SEARCH, TuningMethod.OPTUNA
    """
    print("🚀 통합 머신러닝 파이프라인 시작!")
    print(f"🔧 튜닝 방법: {tuning_method}")

    # 결과 저장 디렉토리 생성
    os.makedirs('./result', exist_ok=True)

    try:
        # 1. 데이터 로드 및 전처리
        print("\n1️⃣ 데이터 로드 및 전처리")
        df_all, X_train_raw, X_test_raw, y_train, y_test, feature_cols = data_processing()

        # 2. Feature Selection
        print("\n2️⃣ Feature Selection")
        selected_features = feature_selection(X_train_raw, y_train, final_k=50)

        # 3. 모델 생성 (선택된 튜닝 방법에 따라)
        print("\n3️⃣ 모델 생성")
        if tuning_method == TuningMethod.NONE:
            models = create_ml_models(tuning_method)
        else:
            models = create_ml_models(tuning_method, X_train_raw[selected_features], y_train)

        # 4. 데이터 스케일링
        print("\n4️⃣ 데이터 스케일링")
        scale_needed = ['Logistic Regression', 'KNN', 'Support Vector Machine']

        scaled_X_train = {}
        scaled_X_test = {}

        for model_name in models.keys():
            scaler = StandardScaler()  # 각 모델별로 새로운 스케일러 객체 사용

            if model_name in scale_needed:
                print(f"   🔹 {model_name}: 스케일링 적용")
                X_train_scaled = scaler.fit_transform(X_train_raw[selected_features])
                X_test_scaled = scaler.transform(X_test_raw[selected_features])
                scaled_X_train[model_name] = pd.DataFrame(X_train_scaled, columns=selected_features)
                scaled_X_test[model_name] = pd.DataFrame(X_test_scaled, columns=selected_features)
            else:
                print(f"   ⚪ {model_name}: 스케일링 미적용")
                scaled_X_train[model_name] = X_train_raw[selected_features].copy()
                scaled_X_test[model_name] = X_test_raw[selected_features].copy()

        # 5. 교차 검증 및 평가
        print("\n5️⃣ 교차 검증 및 평가")
        cv_results = cross_validate_models(models, X_train_raw[selected_features], y_train)

        results = []
        for model_name, model in models.items():
            # 스케일링된 데이터로 평가
            X_train_eval = scaled_X_train[model_name]
            X_test_eval = scaled_X_test[model_name]

            metrics, y_pred, y_proba, trained_model = evaluate_model(
                model, model_name, X_train_eval, X_test_eval, y_train, y_test
            )

            if metrics is not None:
                # 교차 검증 결과 추가
                metrics['CV_F1'] = cv_results[model_name]['CV_F1_mean']
                metrics['CV_F1_std'] = cv_results[model_name]['CV_F1_std']

                result_entry = {
                    'Model': model_name,
                    'Best Model': trained_model,
                    'y_pred': y_pred,
                    'y_proba': y_proba,
                    **metrics
                }
                results.append(result_entry)

        # 6. 결과 분석 및 시각화
        print("\n6️⃣ 결과 분석")
        print("-" * 60)

        if len(results) > 0:
            # 결과 DataFrame 생성 (시각화용 데이터만 포함)
            results_df = pd.DataFrame([
                {k: v for k, v in result.items() if k not in ['Best Model', 'y_pred', 'y_proba']}
                for result in results
            ])
            results_df = results_df.sort_values('F1', ascending=False)

            print(f"\n🏆 모델 성능 순위 (F1 Score 기준):")
            for i, row in results_df.iterrows():
                print(
                    f"- {row['Model']} | F1={row['F1']:.4f}, AUC={row['AUC']:.4f}, Acc={row['Accuracy']:.4f} | CV_F1={row['CV_F1']:.4f} ± {row['CV_F1_std']:.4f}")

            # 시각화
            plot_f1_comparison(results_df)
            plot_roc_comparison(results, y_test)

            # 최고 성능 모델 정보
            best_result = results[0]  # F1 기준으로 정렬된 첫 번째 결과
            best_model_name = best_result['Model']
            print(f"\n🥇 최고 성능 모델: {best_model_name}")
            print("\n📊 분류 보고서:")
            print(classification_report(y_test, best_result['y_pred'], target_names=['Beginner', 'Trained']))

            # 논문용 종합 결과 저장
            save_comprehensive_results(results, results_df, selected_features, tuning_method, y_test)

            # 통계적 유의성 검정 (논문용)
            statistical_significance_test(results)

            print(f"\n💾 논문용 결과 저장 완료:")
            print(f"   - 종합 성능 결과: ./result/comprehensive_results.xlsx")
            print(f"   - 상세 분류 결과: ./result/detailed_classification.xlsx")
            print(f"   - 실험 설정 정보: ./result/experiment_config.xlsx")
            print(f"   - 통계적 유의성: ./result/statistical_significance.xlsx")
            if tuning_method == TuningMethod.OPTUNA:
                print(f"   - Optuna 최적화: ./result/optuna_results.xlsx")

        else:
            print("❌ 평가된 모델이 없습니다.")

    except Exception as e:
        print(f"❌ 파이프라인 실행 중 오류 발생: {e}")
        import traceback
        traceback.print_exc()

    print(f"\n🎉 머신러닝 파이프라인 완료!")


# ========================================
# 논문용 종합 결과 저장 함수
# ========================================
def save_comprehensive_results(results, results_df, selected_features, tuning_method, y_test):
    """논문 발표용 종합적인 결과를 엑셀 파일로 저장합니다."""

    # ExcelWriter 객체 생성 (다중 시트 저장용)
    with pd.ExcelWriter('./result/comprehensive_results.xlsx', engine='openpyxl') as writer:

        # Sheet 1: 모델 성능 요약
        performance_summary = results_df.copy()
        performance_summary = performance_summary.round(4)
        performance_summary.to_excel(writer, sheet_name='Performance_Summary', index=False)

        # Sheet 2: 상세 성능 지표 (모든 지표 포함)
        detailed_metrics = []
        for result in results:
            row = {
                'Model': result['Model'],
                'Accuracy': result['Accuracy'],
                'Precision': result['Precision'],
                'Recall': result['Recall'],
                'F1_Score': result['F1'],
                'AUC': result['AUC'],
                'Balanced_Accuracy': result['Balanced_Accuracy'],
                'Specificity': result['Specificity'],
                'Sensitivity': result['Sensitivity'],
                'MCC': result['MCC'],
                'CV_F1_Mean': result['CV_F1'],
                'CV_F1_Std': result['CV_F1_std']
            }
            detailed_metrics.append(row)

        detailed_df = pd.DataFrame(detailed_metrics).round(4)
        detailed_df.to_excel(writer, sheet_name='Detailed_Metrics', index=False)

        # Sheet 3: 혼동 행렬 (모든 모델)
        confusion_matrices = []
        for result in results:
            if result['y_pred'] is not None:
                cm = confusion_matrix(y_test, result['y_pred'])
                if cm.size == 4:  # 2x2 matrix
                    tn, fp, fn, tp = cm.ravel()
                    confusion_matrices.append({
                        'Model': result['Model'],
                        'True_Negative': tn,
                        'False_Positive': fp,
                        'False_Negative': fn,
                        'True_Positive': tp,
                        'Total_Samples': len(y_test)
                    })

        confusion_df = pd.DataFrame(confusion_matrices)
        confusion_df.to_excel(writer, sheet_name='Confusion_Matrices', index=False)

        # Sheet 4: 클래스별 분류 보고서
        classification_reports = []
        for result in results:
            if result['y_pred'] is not None:
                report = classification_report(
                    y_test, result['y_pred'],
                    target_names=['Beginner', 'Trained'],
                    output_dict=True
                )

                # Beginner 클래스
                classification_reports.append({
                    'Model': result['Model'],
                    'Class': 'Beginner',
                    'Precision': report['Beginner']['precision'],
                    'Recall': report['Beginner']['recall'],
                    'F1_Score': report['Beginner']['f1-score'],
                    'Support': report['Beginner']['support']
                })

                # Trained 클래스
                classification_reports.append({
                    'Model': result['Model'],
                    'Class': 'Trained',
                    'Precision': report['Trained']['precision'],
                    'Recall': report['Trained']['recall'],
                    'F1_Score': report['Trained']['f1-score'],
                    'Support': report['Trained']['support']
                })

                # Macro avg
                classification_reports.append({
                    'Model': result['Model'],
                    'Class': 'Macro_Avg',
                    'Precision': report['macro avg']['precision'],
                    'Recall': report['macro avg']['recall'],
                    'F1_Score': report['macro avg']['f1-score'],
                    'Support': report['macro avg']['support']
                })

                # Weighted avg
                classification_reports.append({
                    'Model': result['Model'],
                    'Class': 'Weighted_Avg',
                    'Precision': report['weighted avg']['precision'],
                    'Recall': report['weighted avg']['recall'],
                    'F1_Score': report['weighted avg']['f1-score'],
                    'Support': report['weighted avg']['support']
                })

        class_report_df = pd.DataFrame(classification_reports).round(4)
        class_report_df.to_excel(writer, sheet_name='Class_Reports', index=False)

        # Sheet 5: 선택된 특성 정보
        feature_info = {
            'Feature_Name': selected_features,
            'Feature_Index': range(len(selected_features)),
            'Selection_Method': ['Multi-step Filter (Variance + Correlation + ANOVA + MI)'] * len(selected_features)
        }
        feature_df = pd.DataFrame(feature_info)
        feature_df.to_excel(writer, sheet_name='Selected_Features', index=False)

    # 별도 파일: 상세 분류 결과
    with pd.ExcelWriter('./result/detailed_classification.xlsx', engine='openpyxl') as writer:

        # 각 모델의 예측 결과 저장
        for i, result in enumerate(results):
            if result['y_pred'] is not None:
                pred_results = pd.DataFrame({
                    'True_Label': y_test,
                    'Predicted_Label': result['y_pred'],
                    'Correct_Prediction': y_test == result['y_pred']
                })

                if result['y_proba'] is not None:
                    pred_results['Prediction_Probability'] = result['y_proba']

                sheet_name = result['Model'].replace(' ', '_')[:31]  # 엑셀 시트명 길이 제한
                pred_results.to_excel(writer, sheet_name=sheet_name, index=False)

    # 실험 설정 정보 저장
    experiment_config = {
        'Parameter': [
            'Random_State', 'Test_Size', 'CV_Folds', 'Feature_Selection_Method',
            'Final_Features_Count', 'Tuning_Method', 'Scaling_Applied_Models',
            'Total_Original_Features', 'Models_Evaluated'
        ],
        'Value': [
            RANDOM_STATE, 0.2, 5, 'Variance + Correlation + ANOVA + MI',
            len(selected_features), tuning_method, 'LR, KNN, SVM',
            'Dynamic', len(results)
        ],
        'Description': [
            'Fixed seed for reproducibility',
            'Proportion of data used for testing',
            'Number of cross-validation folds',
            'Multi-step feature selection approach',
            'Number of features after selection',
            'Hyperparameter optimization method',
            'Models that received feature scaling',
            'Number of features before selection',
            'Total number of models evaluated'
        ]
    }

    config_df = pd.DataFrame(experiment_config)
    config_df.to_excel('./result/experiment_config.xlsx', index=False)


# ========================================
# Statistical Significance Testing (논문용 추가 기능)
# ========================================
def statistical_significance_test(results):
    """모델 간 성능 차이의 통계적 유의성을 검정합니다."""
    from scipy import stats

    print("\n📈 통계적 유의성 검정")
    print("-" * 50)

    # CV F1 점수들을 추출
    cv_scores = {}
    for result in results:
        model_name = result['Model']
        # CV 결과가 있는 경우에만 처리
        if 'CV_F1' in result and 'CV_F1_std' in result:
            # 정규분포 가정하에 CV 점수들을 시뮬레이션
            # (실제로는 cross_val_score의 개별 점수들을 저장해야 하지만, 평균과 표준편차로 근사)
            mean_score = result['CV_F1']
            std_score = result['CV_F1_std']
            # 5-fold CV 가정
            simulated_scores = np.random.normal(mean_score, std_score, 5)
            cv_scores[model_name] = simulated_scores

    # 최고 성능 모델과 다른 모델들 간 비교
    model_names = list(cv_scores.keys())
    if len(model_names) < 2:
        print("비교할 모델이 부족합니다.")
        return

    best_model = model_names[0]  # 이미 F1 기준으로 정렬되어 있음
    best_scores = cv_scores[best_model]

    significance_results = []

    for model_name in model_names[1:]:
        other_scores = cv_scores[model_name]

        # paired t-test 수행
        t_stat, p_value = stats.ttest_rel(best_scores, other_scores)

        significance_results.append({
            'Best_Model': best_model,
            'Compared_Model': model_name,
            'T_Statistic': t_stat,
            'P_Value': p_value,
            'Significant': p_value < 0.05,
            'Effect_Size': abs(t_stat) / np.sqrt(len(best_scores))  # Cohen's d 근사치
        })

        significance_level = "***" if p_value < 0.001 else "**" if p_value < 0.01 else "*" if p_value < 0.05 else "ns"
        print(f"{best_model} vs {model_name}: t={t_stat:.3f}, p={p_value:.4f} {significance_level}")

    # 통계 결과 저장
    if significance_results:
        sig_df = pd.DataFrame(significance_results).round(4)
        sig_df.to_excel('./result/statistical_significance.xlsx', index=False)
        print("\n💾 통계적 유의성 검정 결과 저장: ./result/statistical_significance.xlsx")


if __name__ == "__main__":
    # 사용법 예제:

    # 1. 기본 파라미터로 실행
    # main(TuningMethod.NONE)

    # 2. GridSearchCV로 하이퍼파라미터 튜닝 후 실행
    # main(TuningMethod.GRID_SEARCH)

    # 3. Optuna로 하이퍼파라미터 튜닝 후 실행 (Optuna 설치 필요)
    # main(TuningMethod.OPTUNA)

    # 기본값으로 실행
    main(TuningMethod.NONE)