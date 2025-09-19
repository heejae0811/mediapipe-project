import os
import glob
import warnings
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
from sklearn.model_selection import train_test_split, StratifiedKFold, learning_curve, cross_val_score
from sklearn.feature_selection import RFECV, SelectKBest, f_classif
from sklearn.inspection import permutation_importance
from sklearn.preprocessing import LabelEncoder, StandardScaler

# 경고 메시지 억제
warnings.filterwarnings('ignore')

# Global Variable
RANDOM_STATE = 42
plt.rcParams['figure.figsize'] = (10, 6)
plt.rcParams['font.size'] = 12


# ========================================
# Data Processing
# ========================================
def data_processing():
    """데이터 로드 및 기본 전처리"""
    csv_files = glob.glob('./features_xlsx_1/*.xlsx')
    print(f"\n📂 분석할 파일 수 - {len(csv_files)}개")

    if len(csv_files) == 0:
        raise FileNotFoundError("./features_xlsx_1/ 경로에 xlsx 파일이 없습니다.")

    df_all = pd.concat([pd.read_excel(file, sheet_name=4) for file in csv_files], ignore_index=True)

    # 기본 정보 출력
    print(f"총 데이터 수: {len(df_all)}")
    print(f"컬럼 수: {df_all.shape[1]}")

    # 결측치 처리
    print(f"결측치 개수: {df_all.isnull().sum().sum()}")
    if df_all.isnull().sum().sum() > 0:
        df_all = df_all.fillna(df_all.median(numeric_only=True))

    # 라벨 인코딩
    y_all = LabelEncoder().fit_transform(df_all['label'])
    print(f"라벨 분포 - 0: {(y_all == 0).sum()}개 / 1: {(y_all == 1).sum()}개")

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
# Feature Selection: Simple Correlation
# ========================================
def remove_high_correlation_features(X, threshold=0.9):
    """높은 상관관계 특성 제거 (간단한 방법)"""
    print(f"\n🔍 상관계수 {threshold} 이상인 특성 제거 중...")

    corr_matrix = X.corr().abs()
    upper_triangle = corr_matrix.where(
        np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
    )

    # 높은 상관관계를 가진 특성 찾기
    to_drop = [column for column in upper_triangle.columns
               if any(upper_triangle[column] > threshold)]

    print(f"제거할 특성: {len(to_drop)}개")
    if len(to_drop) > 0:
        print(f"제거된 특성들: {to_drop[:10]}{'...' if len(to_drop) > 10 else ''}")

    return [col for col in X.columns if col not in to_drop]


# ========================================
# Feature Selection: Statistical
# ========================================
def select_best_features(X, y, k=50):
    """통계적 방법으로 최고 특성 선택"""
    if X.shape[1] <= k:
        print(f"특성 수({X.shape[1]})가 이미 적으므로 특성 선택 생략")
        return X.columns.tolist()

    print(f"\n📊 {X.shape[1]}개 특성에서 상위 {k}개 선택 중...")

    selector = SelectKBest(score_func=f_classif, k=k)
    selector.fit(X, y)

    selected_features = X.columns[selector.get_support()].tolist()
    print(f"✅ {len(selected_features)}개 특성 선택 완료")

    return selected_features


# ========================================
# Alternative RFECV (Optional)
# ========================================
def simple_rfecv(X, y, max_features=30):
    """간단한 RFECV (원한다면 사용)"""
    print(f"\n🔍 RFECV 실행 중... (입력 특성 수: {X.shape[1]})")

    if X.shape[1] > max_features:
        # 먼저 통계적 방법으로 줄이기
        selected_features = select_best_features(X, y, k=max_features)
        X = X[selected_features]
        print(f"사전 선택으로 {max_features}개 특성으로 제한")

    # 기본 파라미터로 RFECV 실행
    estimator = RandomForestClassifier(random_state=RANDOM_STATE, n_jobs=-1)
    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=RANDOM_STATE)
    rfecv = RFECV(
        estimator=estimator,
        step=3,
        cv=cv,
        scoring='f1',
        n_jobs=-1,
        min_features_to_select=5
    )

    try:
        rfecv.fit(X, y)
        selected_features = X.columns[rfecv.support_].tolist()
        print(f"🎯 RFECV로 선택된 변수 {len(selected_features)}개")

        # RFECV 결과 시각화
        plt.figure(figsize=(12, 6))
        plt.plot(range(1, len(rfecv.cv_results_['mean_test_score']) + 1),
                 rfecv.cv_results_['mean_test_score'], marker='o', linewidth=2)
        plt.axvline(x=rfecv.n_features_, color='red', linestyle='--',
                    label=f'Optimal: {rfecv.n_features_} features')
        plt.xlabel("Number of Selected Features")
        plt.ylabel("Cross-Validation F1 Score")
        plt.title("RFECV Feature Selection")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()

        return selected_features

    except Exception as e:
        print(f"RFECV 실행 중 오류: {e}")
        # 오류 발생 시 통계적 방법으로 대체
        return select_best_features(X, y, k=20)


# ========================================
# Create Default Models
# ========================================
def create_default_models():
    """기본 파라미터로 모든 모델 생성"""

    models = {
        # 로지스틱 회귀 (기본값)
        'Logistic Regression': LogisticRegression(
            random_state=RANDOM_STATE,
            max_iter=1000
        ),

        # 결정 트리 (기본값)
        'Decision Tree': DecisionTreeClassifier(
            random_state=RANDOM_STATE
        ),

        # 랜덤 포레스트 (기본값)
        'Random Forest': RandomForestClassifier(
            random_state=RANDOM_STATE,
            n_jobs=-1
        ),

        # XGBoost (기본값)
        'XGBoost': XGBClassifier(
            random_state=RANDOM_STATE,
            eval_metric='logloss',
            verbosity=0,
            n_jobs=-1
        ),

        # LightGBM (기본값)
        'LightGBM': LGBMClassifier(
            random_state=RANDOM_STATE,
            verbosity=-1,
            n_jobs=-1
        ),

        # SVM (기본값)
        'Support Vector Machine': SVC(
            probability=True,
            random_state=RANDOM_STATE
        )
    }

    print(f"📦 {len(models)}개 기본 모델 생성 완료")
    return models


# ========================================
# Metrics
# ========================================
def compute_metrics(y_true, y_pred, y_proba=None):
    """성능 지표 계산"""
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
# Cross Validation
# ========================================
def cross_validate_models(models, X, y, cv_folds=5):
    """교차 검증으로 모델들 성능 비교"""

    print(f"\n🔄 {cv_folds}-Fold 교차 검증 실행 중...")

    cv_results = {}
    cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=RANDOM_STATE)

    for name, model in models.items():
        print(f"  ⚡ {name} 검증 중...")

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
    """개별 모델 평가"""

    print(f"\n📊 {model_name} 평가 중...")

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
    """Confusion Matrix 시각화"""
    plt.figure(figsize=(8, 6))
    ConfusionMatrixDisplay.from_predictions(
        y_true, y_pred, display_labels=['Beginner', 'Trained'],
        cmap='Blues', values_format='d'
    )
    plt.title(f'Confusion Matrix - {model_name}')
    plt.tight_layout()
    plt.show()


def plot_roc_curve(y_true, y_proba, model_name):
    """ROC Curve 시각화"""
    fpr, tpr, _ = roc_curve(y_true, y_proba)
    roc_auc = auc(fpr, tpr)

    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, linewidth=2, label=f'AUC = {roc_auc:.3f}')
    plt.plot([0, 1], [0, 1], 'k--', linewidth=1)
    plt.title(f'ROC Curve - {model_name}')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.legend(loc='lower right')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


def plot_combined_roc_curves(results, X_test, y_test):
    """모든 모델의 ROC Curve 통합 시각화"""
    plt.figure(figsize=(12, 8))
    colors = ['red', 'orange', 'green', 'blue', 'purple', 'brown']

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
            color = colors[i % len(colors)]
            plt.plot(fpr, tpr, label=f'{name} (AUC = {roc_auc:.3f})',
                     color=color, linewidth=2)
        except Exception as e:
            print(f"{name} ROC 곡선 생성 오류: {e}")
            continue

    plt.plot([0, 1], [0, 1], linestyle='--', color='gray', linewidth=1)
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curves - All Models (기본 파라미터)')
    plt.legend(loc='lower right')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


def plot_model_comparison(results_df):
    """모델 성능 비교 차트"""
    # F1 Score 기준 정렬
    results_df_sorted = results_df.sort_values('F1', ascending=True)

    plt.figure(figsize=(12, 8))

    # 바 차트
    bars = plt.barh(results_df_sorted['Model'], results_df_sorted['F1'])

    # 색상 적용
    colors = plt.cm.viridis(np.linspace(0, 1, len(bars)))
    for bar, color in zip(bars, colors):
        bar.set_color(color)

    plt.xlabel('F1 Score')
    plt.title('모델 성능 비교 (기본 파라미터)')
    plt.grid(True, alpha=0.3)

    # 값 표시
    for i, bar in enumerate(bars):
        width = bar.get_width()
        plt.text(width + 0.005, bar.get_y() + bar.get_height() / 2,
                 f'{width:.3f}', ha='left', va='center')

    plt.tight_layout()
    plt.show()


def plot_learning_curve(estimator, X, y, model_name):
    """Learning Curve 시각화"""
    try:
        train_sizes, train_scores, val_scores = learning_curve(
            estimator, X, y, cv=3, scoring='f1', n_jobs=-1,
            train_sizes=np.linspace(0.1, 1.0, 10)
        )

        train_mean = train_scores.mean(axis=1)
        train_std = train_scores.std(axis=1)
        val_mean = val_scores.mean(axis=1)
        val_std = val_scores.std(axis=1)

        plt.figure(figsize=(10, 6))
        plt.plot(train_sizes, train_mean, 'o-', label='Training Score', linewidth=2)
        plt.plot(train_sizes, val_mean, 'o-', label='Validation Score', linewidth=2)
        plt.fill_between(train_sizes, train_mean - train_std, train_mean + train_std, alpha=0.2)
        plt.fill_between(train_sizes, val_mean - val_std, val_mean + val_std, alpha=0.2)
        plt.title(f'Learning Curve - {model_name}')
        plt.xlabel('Training Set Size')
        plt.ylabel('F1 Score')
        plt.legend(loc='lower right')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()
    except Exception as e:
        print(f"Learning curve 생성 오류: {e}")


def plot_feature_importance(model, X, y, feature_names, model_name):
    """Feature Importance 시각화"""
    # 내재적 중요도 (있는 경우)
    if hasattr(model, 'feature_importances_'):
        imp = model.feature_importances_
        sorted_idx = np.argsort(imp)[::-1]

        plt.figure(figsize=(12, 8))
        top_features = min(15, len(feature_names))
        sns.barplot(x=imp[sorted_idx][:top_features],
                    y=np.array(feature_names)[sorted_idx][:top_features])
        plt.title(f'Feature Importance - {model_name}')
        plt.xlabel('Importance Score')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()

    # Permutation Importance
    try:
        result = permutation_importance(model, X, y, n_repeats=3,
                                        random_state=RANDOM_STATE, n_jobs=-1)
        idx = result.importances_mean.argsort()[::-1]

        plt.figure(figsize=(12, 8))
        top_features = min(15, len(feature_names))
        sns.barplot(x=result.importances_mean[idx][:top_features],
                    y=np.array(feature_names)[idx][:top_features])
        plt.title(f'Permutation Importance - {model_name}')
        plt.xlabel('Importance Score')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()
    except Exception as e:
        print(f"Permutation importance 계산 오류: {e}")


def plot_decision_tree(model, feature_names):
    """Decision Tree 구조 시각화"""
    if hasattr(model, 'tree_'):
        plt.figure(figsize=(30, 10))
        plot_tree(model, feature_names=feature_names,
                  class_names=['Beginner', 'Trained'],
                  filled=True, rounded=True, fontsize=8, max_depth=3)
        plt.title("Decision Tree Structure (Max Depth 3)")
        plt.tight_layout()
        plt.show()
        print(f"실제 Decision Tree 깊이: {model.get_depth()}")


# ========================================
# Main Pipeline
# ========================================
def main():
    """기본 파라미터 머신러닝 파이프라인"""

    print("🚀 기본 파라미터 머신러닝 파이프라인 시작")
    print("=" * 60)
    print("🎯 특징: 하이퍼파라미터 튜닝 없이 기본값만 사용")
    print("⚡ 장점: 빠르고 간단하며 대부분 경우에 충분한 성능")
    print("=" * 60)

    # 결과 저장 디렉토리 생성
    os.makedirs('./result', exist_ok=True)

    try:
        # 1. 데이터 로드 및 전처리
        print("\n1️⃣ 데이터 로드 및 전처리")
        df_all, X_train_raw, X_test_raw, y_train, y_test, feature_cols = data_processing()

        # 2. 특성 선택 (간단한 방법들)
        print("\n2️⃣ 특성 선택")

        # 높은 상관관계 특성 제거
        remaining_features = remove_high_correlation_features(X_train_raw, threshold=0.9)
        X_train_filtered = X_train_raw[remaining_features]
        X_test_filtered = X_test_raw[remaining_features]

        # 통계적 방법으로 최고 특성 선택 (옵션 1)
        selected_features = select_best_features(X_train_filtered, y_train, k=50)

        # 또는 RFECV 사용 (옵션 2 - 주석 해제하면 사용 가능)
        # selected_features = simple_rfecv(X_train_filtered, y_train, max_features=50)

        print(f"최종 선택된 특성: {len(selected_features)}개")

        # 3. 데이터 스케일링
        print("\n3️⃣ 데이터 스케일링")
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train_filtered[selected_features])
        X_test = scaler.transform(X_test_filtered[selected_features])

        # DataFrame 형태로 변환 (컬럼명 유지)
        X_train = pd.DataFrame(X_train, columns=selected_features)
        X_test = pd.DataFrame(X_test, columns=selected_features)

        # 4. 기본 파라미터 모델 생성
        print("\n4️⃣ 기본 파라미터 모델 생성")
        models = create_default_models()

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

                results.append({
                    'Model': model_name,
                    'Best Model': trained_model,
                    'y_pred': y_pred,
                    'y_proba': y_proba,
                    **metrics
                })

        # 7. 결과 분석 및 시각화
        print("\n7️⃣ 결과 분석 및 시각화")

        if len(results) > 0:
            # 결과 DataFrame 생성 및 정렬
            results_df = pd.DataFrame([{k: v for k, v in result.items()
                                        if k not in ['Best Model', 'y_pred', 'y_proba']}
                                       for result in results])
            results_df = results_df.sort_values('F1', ascending=False)

            # 성능 순위 출력
            print(f"\n🏆 모델 성능 순위 (F1 Score 기준):")
            print("=" * 80)
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
            print(classification_report(y_test, best_result['y_pred'],
                                        target_names=['Beginner', 'Trained']))

            # 시각화
            plot_confusion_matrix(y_test, best_result['y_pred'], best_model_name)

            if best_result['y_proba'] is not None:
                plot_roc_curve(y_test, best_result['y_proba'], best_model_name)

            plot_learning_curve(best_model, X_train, y_train, best_model_name)
            plot_feature_importance(best_model, X_train, y_train, selected_features, best_model_name)

            if best_model_name == 'Decision Tree':
                plot_decision_tree(best_model, selected_features)

            # 결과 저장
            results_df.to_excel('./result/basic_ml_results.xlsx', index=False)
            print(f"\n💾 결과 저장: ./result/basic_ml_results.xlsx")

            # 선택된 특성 저장
            feature_df = pd.DataFrame({'Selected_Features': selected_features})
            feature_df.to_excel('./result/selected_features.xlsx', index=False)
            print(f"💾 선택된 특성 저장: ./result/selected_features.xlsx")

        else:
            print("❌ 평가된 모델이 없습니다.")

    except Exception as e:
        print(f"❌ 파이프라인 실행 중 오류 발생: {e}")
        import traceback
        traceback.print_exc()

    print(f"\n🎉 기본 파라미터 머신러닝 파이프라인