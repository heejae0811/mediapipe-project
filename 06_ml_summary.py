import glob
import optuna
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.ensemble import RandomForestClassifier
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier
from sklearn.metrics import (
    confusion_matrix, classification_report, f1_score, matthews_corrcoef,
    precision_score, recall_score, roc_auc_score, roc_curve, auc,
    balanced_accuracy_score, ConfusionMatrixDisplay
)
from sklearn.model_selection import train_test_split, StratifiedKFold, learning_curve
from sklearn.feature_selection import RFECV
from sklearn.inspection import permutation_importance
from sklearn.preprocessing import LabelEncoder, StandardScaler

# Global Variable
RANDOM_STATE = 42

# ========================================
# Data Processing
# ========================================
def data_processing():
    csv_files = glob.glob('./features_xlsx_1/*.xlsx')
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

# ========================================
# Feature Selection: Correlation
# ========================================
def find_highly_correlation(X, threshold=0.9):
    corr_matrix = X.corr().abs()
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    correlated_pairs = [
        (col1, col2, upper.loc[col1, col2])
        for col1 in upper.columns
        for col2 in upper.index
        if upper.loc[col1, col2] > threshold
    ]

    return correlated_pairs

# ========================================
# Feature Selection: drop low importance
# ========================================
def drop_low_importance(X, y, correlated_pairs):
    rf = RandomForestClassifier(
        n_estimators=200,
        random_state=RANDOM_STATE,
        n_jobs=-1
    )
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

    return list(to_drop)

# ========================================
# Feature Selection: RFECV
# ========================================
def run_rfecv(X, y):
    estimator = LGBMClassifier(
        n_estimators=50,
        learning_rate=0.1,
        max_depth=5,
        random_state=RANDOM_STATE,
        n_jobs=-1,
        verbosity=-1
    )
    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=RANDOM_STATE)
    rfecv = RFECV(estimator=estimator, step=3, cv=cv, scoring='roc_auc', n_jobs=-1)
    rfecv.fit(X, y)

    selected_features = X.columns[rfecv.support_]
    print(f"\n🎯LGBMClassifier RFECV로 선택된 변수 {len(selected_features)}개:")
    print(selected_features)

    plt.figure(figsize=(10, 6))
    plt.plot(range(1, len(rfecv.cv_results_['mean_test_score']) + 1), rfecv.cv_results_['mean_test_score'], marker='o')
    plt.xlabel("Number of Selected Features")
    plt.ylabel("Cross-Validation Score (AUC)")
    plt.title("RFECV with XGBoost")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    return X.columns[rfecv.support_]

# ========================================
# Metrics
# ========================================
def compute_metrics(y_true, y_pred, y_proba):
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()

    return {
        'Accuracy': (tn + tp) / (tn + fp + fn + tp),
        'Precision': precision_score(y_true, y_pred),
        'Recall': recall_score(y_true, y_pred),
        'F1': f1_score(y_true, y_pred),
        'Balanced_Accuracy': balanced_accuracy_score(y_true, y_pred),
        'Specificity': tn / (tn + fp),
        'Sensitivity': tp / (tp + fn),
        'MCC': matthews_corrcoef(y_true, y_pred),
        'AUC': roc_auc_score(y_true, y_proba) if y_proba is not None else None
    }

# ========================================
# Optuna Objective
# ========================================
def objective(trial, model_name, X, y, selected_features):
    if not isinstance(X, pd.DataFrame):
        X = pd.DataFrame(X, columns=selected_features)

    if model_name == 'Logistic Regression':
        params = {"C": trial.suggest_float("C", 1e-3, 10, log=True)}
        model = LogisticRegression(max_iter=500, random_state=RANDOM_STATE, penalty="l2", solver="liblinear", **params)

    elif model_name == 'Decision Tree':
        params = {"max_depth": trial.suggest_int("max_depth", 2, 6)}
        model = DecisionTreeClassifier(random_state=RANDOM_STATE, **params)

    elif model_name == 'Random Forest':
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 50, 100),
            "max_depth": trial.suggest_int("max_depth", 2, 6)
        }
        model = RandomForestClassifier(random_state=RANDOM_STATE, n_jobs=-1, **params)

    elif model_name == 'XGBoost':
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 50, 100),
            "max_depth": trial.suggest_int("max_depth", 2, 6),
            "learning_rate": trial.suggest_float("learning_rate", 0.05, 0.2)
        }
        model = XGBClassifier(random_state=RANDOM_STATE, eval_metric="logloss", n_jobs=-1, **params)

    elif model_name == 'LightGBM':
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 50, 100),
            "max_depth": trial.suggest_int("max_depth", -1, 6),
            "learning_rate": trial.suggest_float("learning_rate", 0.05, 0.2)
        }
        model = LGBMClassifier(random_state=RANDOM_STATE, n_jobs=-1, **params)

    elif model_name == 'Support Vector Machine':
        params = {"C": trial.suggest_float("C", 0.1, 5, log=True)}
        model = SVC(probability=True, random_state=RANDOM_STATE, kernel="rbf", gamma="scale", **params)

    else:
        raise ValueError(f"지원하지 않는 모델: {model_name}")

    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=RANDOM_STATE)
    scores = []

    for train_idx, val_idx in cv.split(X, y):
        X_tr, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_tr, y_val = y[train_idx], y[val_idx]
        try:
            model.fit(X_tr, y_tr)
            preds = model.predict(X_val)
            scores.append(f1_score(y_val, preds))
        except ValueError:
            return float('nan')

    return np.mean(scores)

# ========================================
# Visualization
# ========================================
def plot_confusion_matrix(y_true, y_pred, model_name):
    ConfusionMatrixDisplay.from_predictions(
        y_true, y_pred, display_labels=['Beginner', 'Trained'], cmap='Blues'
    )

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
    colors = ['red', 'orange', 'green', 'blue', 'purple' 'brown']
    plt.figure(figsize=(10, 6))

    for result, color in zip(results, colors):
        name, model = result['Model'], result['Best Model']

        if model is None:
            continue

        if hasattr(model, "predict_proba"):
            y_prob = model.predict_proba(X_test)[:, 1]
        elif hasattr(model, "decision_function"):
            y_prob = model.decision_function(X_test)
        else:
            continue

        fpr, tpr, _ = roc_curve(y_test, y_prob)
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, label=f'{name} (AUC = {roc_auc:.2f})', color=color)

    plt.plot([0, 1], [0, 1], linestyle='--', color='gray')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve (All Models)')
    plt.legend(loc='lower right')
    plt.grid(True, axis='both', linestyle='-', linewidth=0.4, color='gray', alpha=0.4)
    plt.tight_layout()
    plt.show()

def plot_learning_curve(estimator, X, y, model_name):
    sizes, train_scores, val_scores = learning_curve(estimator, X, y, cv=3, scoring='f1')
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

def plot_decision_tree(model, feature_names):
    if hasattr(model, 'tree_'):
        plt.figure(figsize=(30, 10))
        plot_tree(model, feature_names=feature_names, class_names=['Beginner', 'Trained'], filled=True, rounded=True, fontsize=10)
        plt.title(f"Decision Tree Structure")
        plt.tight_layout()
        plt.show()

    print(f"Max Depth of {model_name}: {model.get_depth()}")


# ========================================
# Run Model
# ========================================
def run_model(model_name, X_train, X_test, y_train, y_test, selected_features, n_trials=5):
    sampler = optuna.samplers.RandomSampler()
    study = optuna.create_study(direction="maximize", sampler=sampler)
    study.optimize(lambda trial: objective(trial, model_name, X_train, y_train, selected_features), n_trials=n_trials)
    best_params = study.best_params
    print(f"\n========== {model_name} (Light) ==========")
    print("Best Params:", best_params)
    print("Best CV F1:", study.best_value)

    # 모델별 best_params 적용
    if model_name == 'Logistic Regression':
        model = LogisticRegression(max_iter=500, random_state=RANDOM_STATE, penalty="l2", solver="liblinear", **best_params)

    elif model_name == 'Decision Tree':
        model = DecisionTreeClassifier(random_state=RANDOM_STATE, **best_params)

    elif model_name == 'Random Forest':
        model = RandomForestClassifier(random_state=RANDOM_STATE, n_jobs=-1, **best_params)

    elif model_name == 'XGBoost':
        model = XGBClassifier(random_state=RANDOM_STATE, eval_metric="logloss", n_jobs=-1, **best_params)

    elif model_name == 'LightGBM':
        model = LGBMClassifier(random_state=RANDOM_STATE, n_jobs=-1, **best_params)

    elif model_name == 'Support Vector Machine':
        model = SVC(probability=True, random_state=RANDOM_STATE, kernel="rbf", gamma="scale", **best_params)

    else:
        raise ValueError(f"지원하지 않는 모델: {model_name}")

    X_train_df = pd.DataFrame(X_train, columns=selected_features)
    X_test_df = pd.DataFrame(X_test, columns=selected_features)
    model.fit(X_train_df, y_train)
    y_pred = model.predict(X_test_df)
    y_proba = model.predict_proba(X_test_df)[:, 1]
    print(classification_report(y_test, y_pred))

    # 시각화
    plot_confusion_matrix(y_test, y_pred, model_name)
    if y_proba is not None:
        plot_roc_curve(y_test, y_proba, model_name)

    plot_learning_curve(model, X_train_df, y_train, model_name)
    plot_feature_importance(model, X_train_df, y_train, selected_features, model_name)

    if model_name == 'Decision Tree':
        plot_decision_tree(model, selected_features)

    return {
        'Model': model_name,
        'Best Params': str(best_params),
        'Best Model': model,
        **compute_metrics(y_test, y_pred, y_proba)
    }

# ========================================
# Main
# ========================================
if __name__ == '__main__':
    df_all, X_train_raw, X_test_raw, y_train, y_test, feature_cols = data_processing()

    correlated_pairs = find_highly_correlation(X_train_raw, threshold=0.85)
    to_drop = drop_low_importance(X_train_raw, y_train, correlated_pairs)
    all_corr_vars = set([col for pair in correlated_pairs for col in pair[:2]])
    remaining_corr_vars = all_corr_vars - set(to_drop)
    uncorrelated_vars = set(X_train_raw.columns) - all_corr_vars
    final_candidate_vars = list(remaining_corr_vars | uncorrelated_vars)
    selected_features = run_rfecv(X_train_raw[final_candidate_vars], y_train)

    scaler = StandardScaler()
    scaler.fit(X_train_raw[selected_features])
    X_train = pd.DataFrame(scaler.transform(X_train_raw[selected_features]), columns=selected_features)
    X_test = pd.DataFrame(scaler.transform(X_test_raw[selected_features]), columns=selected_features)

    model_names = [
        'Logistic Regression',
        'Decision Tree',
        'Random Forest',
        'XGBoost',
        'LightGBM',
        'Support Vector Machine'
    ]

    results = []
    for model_name in model_names:
        metrics = run_model(model_name, X_train, X_test, y_train, y_test, selected_features, n_trials=5)
        results.append(metrics)

    # ROC 통합
    plot_combined_roc_curves(results, X_test, y_test)

    # 결과 저장
    results_df = pd.DataFrame(results)
    results_df.sort_values(by='F1', ascending=False, inplace=True)
    results_df.to_excel('./result/model_comparison_optuna.xlsx', index=False)

    print("\n✅ 결과 저장 완료: ./result/model_comparison_optuna.xlsx")
