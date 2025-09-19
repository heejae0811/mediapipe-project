import os
import glob
import warnings
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib as mpl
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
from sklearn.model_selection import train_test_split, StratifiedKFold, learning_curve, cross_val_score
from sklearn.feature_selection import SelectKBest, f_classif, chi2, mutual_info_classif
from sklearn.inspection import permutation_importance
from sklearn.preprocessing import LabelEncoder, StandardScaler

# 경고 메시지 억제
warnings.filterwarnings('ignore')

# Global Variable
RANDOM_STATE = 42
# mpl.rcParams['figure.dpi'] = 300  # 고해상도
# mpl.rcParams['savefig.dpi'] = 300
# mpl.rcParams['font.size'] = 14
# mpl.rcParams['axes.titlesize'] = 16
# mpl.rcParams['axes.labelsize'] = 14
# mpl.rcParams['legend.fontsize'] = 12
# mpl.rcParams['lines.linewidth'] = 2.5
# mpl.rcParams['grid.linestyle'] = '--'
# mpl.rcParams['grid.alpha'] = 0.6
# sns.set_palette("Set2")

# ========================================
# Data Processing: 데이터 로드 및 기본 전처리
# ========================================
def data_processing():
    csv_files = glob.glob('./features_xlsx_2/*.xlsx')
    print(f"\n📂 분석할 파일 수 - {len(csv_files)}개")

    if len(csv_files) == 0:
        raise FileNotFoundError("경로에 파일이 없습니다.")

    df_all = pd.concat([pd.read_excel(file, sheet_name=0) for file in csv_files], ignore_index=True)

    # 기본 정보 출력
    print(f"총 데이터 수: {len(df_all)}개")
    print(f"컬럼 수: {df_all.shape[1]}개")

    # 결측치 처리
    print(f"결측치 개수: {df_all.isnull().sum().sum()}개")
    if df_all.isnull().sum().sum() > 0:
        df_all = df_all.fillna(df_all.median(numeric_only=True))

    # 라벨 인코딩
    y_all = LabelEncoder().fit_transform(df_all['label'])
    print(f"라벨 분포: 0 - {(y_all == 0).sum()}개 / 1 - {(y_all == 1).sum()}개")

    # 수치형 특성만 선택
    feature_cols = df_all.select_dtypes(include=['float64', 'int64']).columns.drop('label', errors='ignore')
    raw_features = df_all[feature_cols]

    # 무한값 처리
    raw_features = raw_features.replace([np.inf, -np.inf], np.nan)
    raw_features = raw_features.fillna(raw_features.median())

    print(f"최종 특성 수: {raw_features.shape[1]}개")

    X_train_raw, X_test_raw, y_train, y_test = train_test_split(
        raw_features, y_all, test_size=0.2, stratify=y_all, random_state=RANDOM_STATE
    )

    return df_all, X_train_raw, X_test_raw, y_train, y_test, feature_cols


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

    # 2단계: 상관관계 필터링 (높은 상관관계 제거)
    print("\n2. 상관관계 필터링 (Pearson Correlation)")
    corr_threshold = 0.90  # 높은 임계값으로 설정
    corr_matrix = X_filtered.corr().abs()
    upper_triangle = corr_matrix.where(
        np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
    )

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

        selector_mi = SelectKBest(score_func=mutual_info_classif, k=final_k)
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


# ========================================
# ML models training
# ========================================
def create_ml_models():
    print("\n📦 머신러닝 기본 파라미터 모델들 생성 중...")

    models = {
        'Logistic Regression': LogisticRegression(
            random_state=RANDOM_STATE,
            max_iter=1000
        ),
        'Decision Tree': DecisionTreeClassifier(
            random_state=RANDOM_STATE
        ),
        'Random Forest': RandomForestClassifier(
            random_state=RANDOM_STATE,
            n_jobs=-1
        ),
        'XGBoost': XGBClassifier(
            random_state=RANDOM_STATE,
            eval_metric='logloss',
            verbosity=0,
            n_jobs=-1
        ),
        'LightGBM': LGBMClassifier(
            random_state=RANDOM_STATE,
            verbosity=-1,
            n_jobs=-1
        ),
        'Support Vector Machine': SVC(
            probability=True,
            random_state=RANDOM_STATE
        )
    }

    print(f"{len(models)}개 기본 모델 생성 완료!")

    return models


# ========================================
# Metrics: 성능 지표 계산
# ========================================
def compute_metrics(y_true, y_pred, y_proba=None):
    cm = confusion_matrix(y_true, y_pred)
    if cm.size == 4:  # 2x2 confusion matrix
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


# ========================================
# Cross Validation: 교차 검증
# ========================================
def cross_validate_models(models, X, y, cv_folds=5):
    print(f"\n🔄 {cv_folds}-Fold 교차 검증 실행 중...")

    cv_results = {}
    cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=RANDOM_STATE)

    for name, model in models.items():
        print(f"  ⚡ {name}")

        try:
            # F1 Score 기준 교차 검증
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


# ========================================
# Model Evaluation
# ========================================
def evaluate_model(model, model_name, X_train, X_test, y_train, y_test):
    print(f"\n📊 {model_name}")

    try:
        # 모델 학습
        model.fit(X_train, y_train)

        # 예측
        y_pred = model.predict(X_test)

        # 확률 예측
        y_proba = None
        if hasattr(model, 'predict_proba'):
            y_proba = model.predict_proba(X_test)[:, 1]
        elif hasattr(model, 'decision_function'):
            y_proba = model.decision_function(X_test)

        # 성능 지표 계산
        metrics = compute_metrics(y_test, y_pred, y_proba)

        print(f"  ✅ F1: {metrics['F1']:.4f}, AUC: {metrics['AUC']:.4f}, Accuracy: {metrics['Accuracy']:.4f}")

        return metrics, y_pred, y_proba, model

    except Exception as e:
        print(f"  ❌ {model_name} 평가 실패: {e}")
        return None, None, None, None


# ========================================
# Visualization Functions
# ========================================

def plot_confusion_matrix(y_true, y_pred, model_name):
    fig, ax = plt.subplots(figsize=(10, 6))
    ConfusionMatrixDisplay.from_predictions(
        y_true, y_pred,
        display_labels=['Beginner', 'Trained'],
        cmap=plt.cm.Blues, values_format='d',
        colorbar=False,
        ax=ax
    )
    ax.set_title(f'Confusion Matrix - {model_name}', fontsize=16, weight='bold')
    ax.tick_params(axis='both', labelsize=12)
    fig.tight_layout()
    plt.show()


def plot_roc_curve(y_true, y_proba, model_name):
    fpr, tpr, _ = roc_curve(y_true, y_proba)
    roc_auc = auc(fpr, tpr)

    plt.figure(figsize=(10, 6))
    plt.plot(fpr, tpr, label=f'{model_name} (AUC = {roc_auc:.3f})', linewidth=2)
    plt.plot([0, 1], [0, 1], 'k--', linewidth=1)
    plt.title(f'ROC Curve - {model_name}', fontsize=16, weight='bold')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.legend(loc='lower right', frameon=True)
    plt.grid(True, linestyle='-', alpha=0.5)
    plt.tight_layout()
    plt.show()


def plot_combined_roc_curves(results, X_test, y_test):
    plt.figure(figsize=(10, 6))
    colors = plt.cm.tab10.colors

    for i, result in enumerate(results):
        if result['Best Model'] is None:
            continue
        model = result['Best Model']
        name = result['Model']

        try:
            if hasattr(model, "predict_proba"):
                y_prob = model.predict_proba(X_test)[:, 1]
            elif hasattr(model, "decision_function"):
                y_prob = model.decision_function(X_test)
            else:
                continue
            fpr, tpr, _ = roc_curve(y_test, y_prob)
            roc_auc = auc(fpr, tpr)
            plt.plot(fpr, tpr, label=f'{name} (AUC = {roc_auc:.3f})', color=colors[i % len(colors)], linewidth=2)
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


def plot_learning_curve(estimator, X, y, model_name):
    try:
        train_sizes, train_scores, val_scores = learning_curve(
            estimator, X, y, cv=3, scoring='f1', n_jobs=-1, train_sizes=np.linspace(0.1, 1.0, 10)
        )

        train_mean = train_scores.mean(axis=1)
        train_std = train_scores.std(axis=1)
        val_mean = val_scores.mean(axis=1)
        val_std = val_scores.std(axis=1)

        plt.figure(figsize=(10, 6))
        plt.plot(train_sizes, train_mean, 'o-', label='Training Score')
        plt.plot(train_sizes, val_mean, 's-', label='Validation Score')
        plt.fill_between(train_sizes, train_mean - train_std, train_mean + train_std, alpha=0.2)
        plt.fill_between(train_sizes, val_mean - val_std, val_mean + val_std, alpha=0.2)
        plt.title(f'Learning Curve - {model_name}', fontsize=16, weight='bold')
        plt.xlabel('Training Set Size')
        plt.ylabel('F1 Score')
        plt.legend(loc='lower right', frameon=True)
        plt.grid(True, linestyle='-', alpha=0.5)
        plt.tight_layout()
        plt.show()

    except Exception as e:
        print(f"Learning curve 생성 오류: {e}")


def plot_feature_importance(model, X, y, feature_names, model_name):
    colors = sns.color_palette("tab10")

    if hasattr(model, 'feature_importances_'):
        imp = model.feature_importances_
        sorted_idx = np.argsort(imp)[::-1]

        plt.figure(figsize=(15, 7))
        top_features = min(15, len(feature_names))
        sns.barplot(x=imp[sorted_idx][:top_features], y=np.array(feature_names)[sorted_idx][:top_features], palette=colors)
        plt.title(f'Feature Importance - {model_name}', fontsize=16, weight='bold')
        plt.xlabel('Importance Score')
        plt.ylabel('Features')
        plt.grid(True, axis='x', linestyle='-', alpha=0.5)
        plt.tight_layout()
        plt.show()

    try:
        result = permutation_importance(model, X, y, n_repeats=5, random_state=RANDOM_STATE, n_jobs=-1)
        idx = result.importances_mean.argsort()[::-1]

        plt.figure(figsize=(15, 7))
        top_features = min(15, len(feature_names))
        sns.barplot(x=result.importances_mean[idx][:top_features], y=np.array(feature_names)[idx][:top_features], palette=colors)
        plt.title(f'Permutation Importance - {model_name}', fontsize=16, weight='bold')
        plt.xlabel('Importance Score')
        plt.ylabel('Features')
        plt.grid(True, axis='x', linestyle='-', alpha=0.5)
        plt.tight_layout()
        plt.show()

    except Exception as e:
        print(f"Permutation importance 계산 오류: {e}")


def plot_decision_tree(model, feature_names):
    if hasattr(model, 'tree_'):
        plt.figure(figsize=(15, 7))
        plot_tree(model, feature_names=feature_names, class_names=['Beginner', 'Trained'], filled=True, rounded=True, fontsize=10, max_depth=3)
        plt.title("Decision Tree Structure (Max Depth 3)", fontsize=16, weight='bold')
        plt.tight_layout()
        plt.show()

        print(f"실제 Decision Tree 깊이: {model.get_depth()}")


def plot_model_comparison(results_df):
    # F1 점수가 높은 모델이 위로 오도록 내림차순 정렬
    results_df_sorted = results_df.sort_values('F1', ascending=False)

    plt.figure(figsize=(10, 6))
    colors = sns.color_palette("tab10", len(results_df_sorted))

    bars = plt.barh(results_df_sorted['Model'], results_df_sorted['F1'], color=colors)

    for i, bar in enumerate(bars):
        width = bar.get_width()
        plt.text(width + 0.01,
                 bar.get_y() + bar.get_height() / 2,
                 f'{width:.3f}', ha='left', va='center', fontsize=12)

    plt.title('F1 Score - All Models', fontsize=16, weight='bold')
    plt.xlabel('F1 Score')
    plt.grid(True, axis='x', linestyle='-', alpha=0.5)
    plt.tight_layout()
    plt.show()


# ========================================
# Main Pipeline
# ========================================
def main():
    print("🚀 머신러닝 파이프라인 시작!")

    # 결과 저장 디렉토리 생성
    os.makedirs('./result', exist_ok=True)

    try:
        # 1. 데이터 로드 및 전처리
        print("\n1️⃣ 데이터 로드 및 전처리")
        df_all, X_train_raw, X_test_raw, y_train, y_test, feature_cols = data_processing()

        # 2. Feature Selection
        print("\n2️⃣ Feature Selection: filter + embedded")
        selected_features = feature_selection(X_train_raw, y_train, final_k=50)

        # 3. 데이터 스케일링
        print("\n3️⃣ 데이터 스케일링")
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train_raw[selected_features])
        X_test = scaler.transform(X_test_raw[selected_features])

        # DataFrame 형태로 변환
        X_train = pd.DataFrame(X_train, columns=selected_features)
        X_test = pd.DataFrame(X_test, columns=selected_features)

        # 4. 기본 파라미터 모델 생성
        print("\n4️⃣ 기본 파라미터 모델 생성")
        models = create_ml_models()

        # 5. 교차 검증
        print("\n5️⃣ 교차 검증 수행")
        cv_results = cross_validate_models(models, X_train, y_train)

        # 6. 테스트 세트 평가
        print("\n6️⃣ 테스트 세트 평가")
        results = []

        for model_name, model in models.items():
            metrics, y_pred, y_proba, trained_model = evaluate_model(
                model, model_name, X_train, X_test, y_train, y_test
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

                # Confusion Matrix
                plot_confusion_matrix(y_test, y_pred, model_name)

                # ROC Curve
                if y_proba is not None:
                    plot_roc_curve(y_test, y_proba, model_name)

                # Learning Curve
                plot_learning_curve(trained_model, X_train, y_train, model_name)

                # Feature Importance / Permutation Importance
                plot_feature_importance(trained_model, X_train, y_train, selected_features, model_name)

                # Decision Tree 모델인 경우만
                if model_name == 'Decision Tree':
                    plot_decision_tree(trained_model, selected_features)

        # 7. 결과 분석 및 시각화
        print("\n7️⃣ 머신러닝 결과 분석")
        print("-" * 60)

        if len(results) > 0:
            # 결과 DataFrame 생성 및 정렬
            results_df = pd.DataFrame([{k: v for k, v in result.items()
                                        if k not in ['Best Model', 'y_pred', 'y_proba']}
                                       for result in results])
            results_df = results_df.sort_values('F1', ascending=False)

            # 성능 순위 출력
            print(f"\n🏆 모델 성능 순위 (F1 Score 기준):")
            for i, row in results_df.iterrows():
                rank = len(results_df) - results_df.index.get_loc(i)
                print(f"{rank}. {row['Model']}")
                print(f"   테스트 F1: {row['F1']:.4f} | AUC: {row['AUC']:.4f} | Accuracy: {row['Accuracy']:.4f}")
                print(f"   교차검증 F1: {row['CV_F1']:.4f} ± {row['CV_F1_std']:.4f}")
                print("-" * 60)

            # 시각화
            plot_model_comparison(results_df)
            plot_combined_roc_curves(results, X_test, y_test)

            # 최고 성능 모델에 대한 상세 분석
            best_result = results[results_df.index[0]]
            best_model = best_result['Best Model']
            best_model_name = best_result['Model']

            print(f"\n🥇 최고 성능 모델: {best_model_name}")

            # 분류 보고서
            print("\n📊 분류 보고서:")
            print(classification_report(y_test, best_result['y_pred'], target_names=['Beginner', 'Trained']))

            # 8. 결과 저장 (엑셀 파일)
            print("\n8️⃣ 결과 저장")

            # 모델 성능 결과 저장
            results_df.to_excel('./result/ml_results.xlsx', index=False)
            print(f"모델 성능 결과 저장: ./result/ml_results.xlsx")

            # 선택된 특성 저장 (Filter + Embedded 방법 정보 포함)
            feature_selection_details = {
                'Selected_Features': selected_features,
                'Feature_Index': range(len(selected_features)),
                'Selection_Method': ['Variance + Correlation Filter + RandomForest Embedded'] * len(selected_features),
                'Filter_Method': ['Variance Threshold + Pearson Correlation'] * len(selected_features),
                'Embedded_Method': ['Random Forest Feature Importance'] * len(selected_features)
            }
            feature_df = pd.DataFrame(feature_selection_details)
            feature_df.to_excel('./result/filter_embedded_selected_features.xlsx', index=False)
            print(f"선택된 특성 저장: ./result/filter_embedded_selected_features.xlsx")

            # 상세 성능 지표 저장
            detailed_results = []
            for result in results:
                detailed_row = {
                    'Model': result['Model'],
                    'Test_F1': result['F1'],
                    'Test_AUC': result['AUC'],
                    'Test_Accuracy': result['Accuracy'],
                    'Test_Precision': result['Precision'],
                    'Test_Recall': result['Recall'],
                    'Test_Balanced_Accuracy': result['Balanced_Accuracy'],
                    'Test_Specificity': result['Specificity'],
                    'Test_Sensitivity': result['Sensitivity'],
                    'Test_MCC': result['MCC'],
                    'CV_F1_Mean': result['CV_F1'],
                    'CV_F1_Std': result['CV_F1_std']
                }
                detailed_results.append(detailed_row)

            detailed_df = pd.DataFrame(detailed_results)
            detailed_df.to_excel('./result/detailed_performance_metrics.xlsx', index=False)
            print(f"상세 성능 지표 저장: ./result/detailed_performance_metrics.xlsx")

            # 최적 Filter 조합 방법론 요약 저장
            methodology_summary = {
                'Step': [
                    '1_Variance_Filtering',
                    '2_Correlation_Filtering',
                    '3_ANOVA_F_test_Primary',
                    '4_Mutual_Information_Final'
                ],
                'Method': [
                    'Variance Threshold',
                    'Pearson Correlation',
                    'ANOVA F-test',
                    'Mutual Information'
                ],
                'Threshold_Parameter': [
                    '0.001',
                    '0.95',
                    f'Top {50 * 2} features',
                    f'Top {50} features'
                ],
                'Purpose': [
                    'Remove meaningless low-variance features',
                    'Remove highly correlated redundant features',
                    'Statistical significance-based primary selection',
                    'Non-linear relationship-based final selection'
                ],
                'Paper_Reference': [
                    'Table 3 - Variance based filtering',
                    'Table 3 - Sr.No. 19 Pearson correlation',
                    'Table 3 - Sr.No. 21 ANOVA F-value',
                    'Table 3 - Sr.No. 11 Mutual Information'
                ]
            }
            methodology_df = pd.DataFrame(methodology_summary)
            methodology_df.to_excel('./result/ml_summary.xlsx', index=False)
            print(f"최적 방법론 요약 저장: ./result/ml_summary.xlsx")

        else:
            print("❌ 평가된 모델이 없습니다.")

    except Exception as e:
        print(f"❌ 파이프라인 실행 중 오류 발생: {e}")
        import traceback
        traceback.print_exc()

    print(f"\n🎉 머신러닝 파이프라인 완료!")


if __name__ == "__main__":
    main()