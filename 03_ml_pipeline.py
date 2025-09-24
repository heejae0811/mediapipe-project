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

# Optuna import
try:
    import optuna
    from optuna.samplers import TPESampler
    from optuna.pruners import MedianPruner
    OPTUNA_AVAILABLE = True
except ImportError:
    OPTUNA_AVAILABLE = False
    print("Optuna 미설치: 'pip install optuna' 필요")

warnings.filterwarnings('ignore')
RANDOM_STATE = 42

# ================================
# Enum
# ================================
class TuningMethod:
    NONE = "none"
    GRID_SEARCH = "grid"
    OPTUNA = "optuna"

# ================================
# Data Processing
# ================================
def data_processing():
    csv_files = glob.glob('./features_xlsx/*.xlsx')
    if len(csv_files) == 0:
        raise FileNotFoundError("경로에 파일이 없습니다.")

    df_all = pd.concat([pd.read_excel(file, sheet_name=4) for file in csv_files], ignore_index=True)
    y_all = LabelEncoder().fit_transform(df_all['label'])
    feature_cols = df_all.select_dtypes(include=['float64', 'int64']).columns.drop('label', errors='ignore')
    raw_features = df_all[feature_cols]

    X_train_raw, X_test_raw, y_train, y_test = train_test_split(
        raw_features, y_all, test_size=0.2, stratify=y_all, random_state=RANDOM_STATE
    )

    train_median = X_train_raw.replace([np.inf, -np.inf], np.nan).median()
    X_train_processed = X_train_raw.fillna(train_median)
    X_test_processed = X_test_raw.fillna(train_median)

    return df_all, X_train_processed, X_test_processed, y_train, y_test, feature_cols

# ================================
# Feature Selection
# ================================
def feature_selection(X, y, final_k=50):
    variances = X.var()
    low_var_features = variances[variances <= 0.001].index.tolist()
    X_filtered = X.drop(columns=low_var_features)

    corr_matrix = X_filtered.corr().abs()
    upper_triangle = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    high_corr = [c for c in upper_triangle.columns if any(upper_triangle[c] > 0.9)]
    X_filtered = X_filtered.drop(columns=high_corr)

    if len(X_filtered.columns) > final_k * 2:
        selector = SelectKBest(score_func=f_classif, k=final_k * 2)
        selector.fit(X_filtered, y)
        X_filtered = X_filtered[X_filtered.columns[selector.get_support()]]

    if len(X_filtered.columns) > final_k:
        selector = SelectKBest(
            score_func=lambda X_, y_: mutual_info_classif(X_, y_, random_state=RANDOM_STATE),
            k=final_k
        )
        selector.fit(X_filtered, y)
        X_filtered = X_filtered[X_filtered.columns[selector.get_support()]]

    return list(X_filtered.columns)

# ================================
# Models (Pipeline 적용)
# ================================
def make_pipeline_with_scaler(model, needs_scaling=True):
    scaler = StandardScaler() if needs_scaling else "passthrough"
    return Pipeline([("scaler", scaler), ("clf", model)])

def get_all_models():
    scaling_models = {
        'LR': LogisticRegression(random_state=RANDOM_STATE, max_iter=2000),
        'KNN': KNeighborsClassifier(n_jobs=-1),
        'SVM': SVC(probability=True, random_state=RANDOM_STATE)
    }
    non_scaling_models = {
        'DT': DecisionTreeClassifier(random_state=RANDOM_STATE),
        'RF': RandomForestClassifier(random_state=RANDOM_STATE, n_jobs=-1),
        'LightGBM': LGBMClassifier(random_state=RANDOM_STATE, n_jobs=-1, verbosity=-1),
        'XGBoost': XGBClassifier(random_state=RANDOM_STATE, eval_metric="logloss", n_jobs=-1, verbosity=0),
        'CatBoost': CatBoostClassifier(random_state=RANDOM_STATE, verbose=False)
    }

    models = {}
    for name, m in scaling_models.items():
        models[name] = make_pipeline_with_scaler(m, needs_scaling=True)
    for name, m in non_scaling_models.items():
        models[name] = make_pipeline_with_scaler(m, needs_scaling=False)
    return models

# ================================
# GridSearch Parameters
# ================================
def get_grid_params():
    return {
        'LR': {'clf__C': [0.01, 0.1, 1, 10, 100], 'clf__solver': ['liblinear', 'lbfgs']},
        'KNN': {'clf__n_neighbors': [3, 5, 7, 11, 15], 'clf__weights': ['uniform', 'distance']},
        'SVM': {'clf__C': [0.1, 1, 10], 'clf__kernel': ['linear', 'rbf']},
        'DT': {'clf__max_depth': [3, 5, 10, None], 'clf__min_samples_split': [2, 5, 10]},
        'RF': {'clf__n_estimators': [50, 100, 200], 'clf__max_depth': [5, 10, None]},
        'LightGBM': {'clf__n_estimators': [50, 100, 200], 'clf__num_leaves': [15, 31, 63]},
        'XGBoost': {'clf__n_estimators': [50, 100, 200], 'clf__max_depth': [3, 5, 7]},
        'CatBoost': {'clf__depth': [4, 6, 8], 'clf__iterations': [100, 200]}
    }

def grid_search_tuning(X_train, y_train):
    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=RANDOM_STATE)
    tuned = {}
    params = get_grid_params()
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
# Optuna Objectives
# ================================
def create_optuna_objectives(X_train, y_train):
    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=RANDOM_STATE)
    def run_cv(model):
        return cross_val_score(model, X_train, y_train, cv=cv, scoring='f1', n_jobs=-1).mean()

    return {
        'LR': lambda trial: run_cv(make_pipeline_with_scaler(
            LogisticRegression(max_iter=2000, random_state=RANDOM_STATE), True
        ).set_params(**{
            'clf__C': trial.suggest_float('clf__C', 0.001, 100, log=True),
            'clf__solver': trial.suggest_categorical('clf__solver', ['liblinear', 'lbfgs'])
        })),
        'KNN': lambda trial: run_cv(make_pipeline_with_scaler(
            KNeighborsClassifier(n_jobs=-1), True
        ).set_params(**{
            'clf__n_neighbors': trial.suggest_int('clf__n_neighbors', 3, 20),
            'clf__weights': trial.suggest_categorical('clf__weights', ['uniform', 'distance'])
        })),
        'SVM': lambda trial: run_cv(make_pipeline_with_scaler(
            SVC(probability=True, random_state=RANDOM_STATE), True
        ).set_params(**{
            'clf__C': trial.suggest_float('clf__C', 0.01, 100, log=True),
            'clf__kernel': trial.suggest_categorical('clf__kernel', ['linear', 'rbf'])
        })),
        'DT': lambda trial: run_cv(make_pipeline_with_scaler(
            DecisionTreeClassifier(random_state=RANDOM_STATE), False
        ).set_params(**{
            'clf__max_depth': trial.suggest_categorical('clf__max_depth', [3, 5, 10, None]),
            'clf__min_samples_split': trial.suggest_int('clf__min_samples_split', 2, 10)
        })),
        'RF': lambda trial: run_cv(make_pipeline_with_scaler(
            RandomForestClassifier(random_state=RANDOM_STATE, n_jobs=-1), False
        ).set_params(**{
            'clf__n_estimators': trial.suggest_int('clf__n_estimators', 50, 300),
            'clf__max_depth': trial.suggest_categorical('clf__max_depth', [5, 10, None])
        })),
        'LightGBM': lambda trial: run_cv(make_pipeline_with_scaler(
            LGBMClassifier(random_state=RANDOM_STATE, n_jobs=-1, verbosity=-1), False
        ).set_params(**{
            'clf__n_estimators': trial.suggest_int('clf__n_estimators', 50, 300),
            'clf__num_leaves': trial.suggest_int('clf__num_leaves', 15, 63)
        })),
        'XGBoost': lambda trial: run_cv(make_pipeline_with_scaler(
            XGBClassifier(random_state=RANDOM_STATE, eval_metric="logloss", n_jobs=-1, verbosity=0), False
        ).set_params(**{
            'clf__n_estimators': trial.suggest_int('clf__n_estimators', 50, 300),
            'clf__max_depth': trial.suggest_int('clf__max_depth', 3, 7)
        })),
        'CatBoost': lambda trial: run_cv(make_pipeline_with_scaler(
            CatBoostClassifier(random_state=RANDOM_STATE, verbose=False), False
        ).set_params(**{
            'clf__depth': trial.suggest_int('clf__depth', 4, 10),
            'clf__iterations': trial.suggest_int('clf__iterations', 100, 300)
        }))
    }

def optuna_tuning(X_train, y_train, n_trials=30):
    if not OPTUNA_AVAILABLE:
        raise ImportError("Optuna가 설치되지 않았습니다.")
    objectives = create_optuna_objectives(X_train, y_train)
    optimized_models = {}
    results = []
    for name, objective in objectives.items():
        study = optuna.create_study(direction="maximize", sampler=TPESampler(seed=RANDOM_STATE), pruner=MedianPruner())
        study.optimize(objective, n_trials=n_trials)
        best_params = study.best_params
        base_models = get_all_models()
        optimized_models[name] = base_models[name].set_params(**best_params)
        results.append({'Model': name, 'Best Params': best_params, 'Best CV F1': study.best_value})
    pd.DataFrame(results).to_excel('./result/optuna_results.xlsx', index=False)
    return optimized_models

# ================================
# Evaluation
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
# Visualization
# ================================
def plot_f1_comparison_per_tuning(results_df, tuning_method):
    subset = results_df[results_df['Tuning'] == tuning_method]
    subset_sorted = subset.sort_values('F1', ascending=False)
    plt.figure(figsize=(8, 5))
    bars = plt.barh(subset_sorted['Model'], subset_sorted['F1'],
                    color=sns.color_palette("Set2", len(subset_sorted)))
    for bar, f1 in zip(bars, subset_sorted['F1']):
        plt.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height()/2, f"{f1:.3f}", va="center")
    plt.title(f"F1 Scores ({tuning_method.upper()})", fontsize=14)
    plt.xlabel("F1 Score")
    plt.tight_layout()

def plot_roc_comparison_per_tuning(results, y_test, tuning_method):
    plt.figure(figsize=(7, 6))
    subset = [r for r in results if r['Tuning'] == tuning_method]
    for r in subset:
        if r['y_proba'] is not None:
            fpr, tpr, _ = roc_curve(y_test, r['y_proba'])
            plt.plot(fpr, tpr, label=f"{r['Model']} (AUC={r['AUC']:.3f})")
    plt.plot([0, 1], [0, 1], 'k--', linewidth=1)
    plt.title(f"ROC Curves ({tuning_method.upper()})", fontsize=14)
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.legend(loc="lower right")
    plt.tight_layout()

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

    for tuning_method in [TuningMethod.NONE, TuningMethod.GRID_SEARCH, TuningMethod.OPTUNA]:
        print(f"\n===== {tuning_method.upper()} 실행 =====")
        if tuning_method == TuningMethod.NONE:
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
