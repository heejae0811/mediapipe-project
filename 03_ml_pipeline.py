import os
import glob
import warnings
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from sklearn.preprocessing import LabelEncoder, StandardScaler, FunctionTransformer
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
    precision_score, recall_score, roc_auc_score, roc_curve, auc,
    balanced_accuracy_score
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
    scaler = StandardScaler() if needs_scaling else FunctionTransformer(lambda x: x, validate=False)
    return Pipeline([("scaler", scaler), ("clf", model)])


def get_all_models():
    scaling_models = {
        'Logistic Regression': LogisticRegression(random_state=RANDOM_STATE, max_iter=1000),
        'KNN': KNeighborsClassifier(n_jobs=-1),
        'Support Vector Machine': SVC(probability=True, random_state=RANDOM_STATE)
    }
    non_scaling_models = {
        'Decision Tree': DecisionTreeClassifier(random_state=RANDOM_STATE),
        'Random Forest': RandomForestClassifier(random_state=RANDOM_STATE, n_jobs=-1),
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
# GridSearch
# ================================
def grid_search_tuning(X_train, y_train):
    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=RANDOM_STATE)
    tuned = {}

    params = {
        'Logistic Regression': {'clf__C': [0.01, 0.1, 1, 10], 'clf__solver': ['liblinear', 'lbfgs']},
        'KNN': {'clf__n_neighbors': [3, 5, 7], 'clf__weights': ['uniform', 'distance']},
        'Support Vector Machine': {'clf__C': [0.1, 1, 10], 'clf__kernel': ['linear', 'rbf']}
    }

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
def create_optuna_objectives(X_train, y_train):
    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=RANDOM_STATE)

    def run_cv(model):
        return cross_val_score(model, X_train, y_train, cv=cv, scoring='f1', n_jobs=-1).mean()

    def logistic_regression_objective(trial):
        params = {
            'clf__C': trial.suggest_float('clf__C', 0.001, 100, log=True),
            'clf__solver': trial.suggest_categorical('clf__solver', ['liblinear', 'lbfgs'])
        }
        pipe = make_pipeline_with_scaler(LogisticRegression(max_iter=2000, random_state=RANDOM_STATE), True)
        return run_cv(pipe.set_params(**params))

    def knn_objective(trial):
        params = {
            'clf__n_neighbors': trial.suggest_int('clf__n_neighbors', 3, 15),
            'clf__weights': trial.suggest_categorical('clf__weights', ['uniform', 'distance'])
        }
        pipe = make_pipeline_with_scaler(KNeighborsClassifier(n_jobs=-1), True)
        return run_cv(pipe.set_params(**params))

    def svm_objective(trial):
        params = {
            'clf__C': trial.suggest_float('clf__C', 0.01, 100, log=True),
            'clf__kernel': trial.suggest_categorical('clf__kernel', ['linear', 'rbf'])
        }
        pipe = make_pipeline_with_scaler(SVC(probability=True, random_state=RANDOM_STATE), True)
        return run_cv(pipe.set_params(**params))

    return {
        'Logistic Regression': logistic_regression_objective,
        'KNN': knn_objective,
        'Support Vector Machine': svm_objective,
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
        best_score = study.best_value
        results.append({'Model': name, 'Best CV F1': best_score, 'Best Params': best_params})

        base_models = get_all_models()
        optimized_models[name] = base_models[name].set_params(**best_params)

    pd.DataFrame(results).to_excel('./result/optuna_results.xlsx', index=False)
    print("💾 Optuna 최적화 결과 저장: ./result/optuna_results.xlsx")

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
        'MCC': matthews_corrcoef(y_true, y_pred)
    }
    metrics['AUC'] = roc_auc_score(y_true, y_proba) if y_proba is not None else 0
    return metrics


# ================================
# Visualization
# ================================
def plot_f1_comparison(results_df):
    results_df_sorted = results_df.sort_values('F1', ascending=False)
    plt.figure(figsize=(8, 6))
    bars = plt.barh(results_df_sorted['Model'], results_df_sorted['F1'],
                    color=sns.color_palette("Set2", len(results_df_sorted)))
    for bar, f1 in zip(bars, results_df_sorted['F1']):
        plt.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height() / 2,
                 f"{f1:.3f}", va="center")
    plt.title("F1 Score Comparison")
    plt.show()


def plot_roc_comparison(results, y_test):
    plt.figure(figsize=(8, 6))
    for r in results:
        if r['y_proba'] is not None:
            fpr, tpr, _ = roc_curve(y_test, r['y_proba'])
            plt.plot(fpr, tpr, label=f"{r['Model']} (AUC={r['AUC']:.3f})")
    plt.plot([0, 1], [0, 1], 'k--')
    plt.legend()
    plt.title("ROC Curves")
    plt.show()


# ================================
# Save Results
# ================================
def save_results(results, selected_features, tuning_method, y_test):
    os.makedirs('./result', exist_ok=True)
    results_df = pd.DataFrame(results).sort_values('F1', ascending=False)

    with pd.ExcelWriter('./result/comprehensive_results.xlsx') as writer:
        results_df.to_excel(writer, sheet_name="Summary", index=False)

        # Confusion Matrices
        cm_list = []
        for r in results:
            cm = confusion_matrix(y_test, r['y_pred'])
            if cm.size == 4:
                tn, fp, fn, tp = cm.ravel()
                cm_list.append({'Model': r['Model'], 'TN': tn, 'FP': fp, 'FN': fn, 'TP': tp})
        pd.DataFrame(cm_list).to_excel(writer, sheet_name="Confusion_Matrix", index=False)

        # Classification Reports
        reports = []
        for r in results:
            report = classification_report(y_test, r['y_pred'], target_names=['Beginner', 'Trained'], output_dict=True)
            for cls in report.keys():
                if cls in ['Beginner', 'Trained', 'macro avg', 'weighted avg']:
                    reports.append({
                        'Model': r['Model'],
                        'Class': cls,
                        'Precision': report[cls]['precision'],
                        'Recall': report[cls]['recall'],
                        'F1': report[cls]['f1-score'],
                        'Support': report[cls]['support']
                    })
        pd.DataFrame(reports).to_excel(writer, sheet_name="Classification_Report", index=False)

        # Feature Info
        pd.DataFrame({'Feature': selected_features}).to_excel(writer, sheet_name="Features", index=False)


# ================================
# Statistical Significance Test
# ================================
def statistical_significance_test(results):
    from scipy import stats
    print("\n📈 통계적 유의성 검정")
    best_model = sorted(results, key=lambda r: r['F1'], reverse=True)[0]['Model']
    best_score = sorted(results, key=lambda r: r['F1'], reverse=True)[0]['F1']
    for r in results:
        if r['Model'] == best_model: continue
        t, p = stats.ttest_rel([best_score], [r['F1']])
        print(f"{best_model} vs {r['Model']}: p={p:.4f}")


# ================================
# Main
# ================================
def main(tuning_method=TuningMethod.NONE):
    df_all, X_train_raw, X_test_raw, y_train, y_test, feature_cols = data_processing()
    selected = feature_selection(X_train_raw, y_train, final_k=50)

    models = None
    if tuning_method == TuningMethod.NONE:
        models = get_all_models()
    elif tuning_method == TuningMethod.GRID_SEARCH:
        models = grid_search_tuning(X_train_raw[selected], y_train)
    elif tuning_method == TuningMethod.OPTUNA:
        models = optuna_tuning(X_train_raw[selected], y_train)
    else:
        raise ValueError("Unknown tuning method")

    results = []
    for name, model in models.items():
        model.fit(X_train_raw[selected], y_train)
        preds = model.predict(X_test_raw[selected])
        proba = model.predict_proba(X_test_raw[selected])[:, 1] if hasattr(model, "predict_proba") else None
        m = compute_metrics(y_test, preds, proba)
        results.append({'Model': name, 'y_pred': preds, 'y_proba': proba, **m})

    results_df = pd.DataFrame(results).sort_values('F1', ascending=False)
    print(results_df[['Model', 'F1', 'AUC', 'Accuracy']])

    plot_f1_comparison(results_df)
    plot_roc_comparison(results, y_test)
    save_results(results, selected, tuning_method, y_test)
    statistical_significance_test(results)


if __name__ == "__main__":
    main(TuningMethod.NONE)  # NONE / GRID_SEARCH / OPTUNA 선택
