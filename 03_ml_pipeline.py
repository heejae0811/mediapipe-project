"""
📚 단계별 머신러닝 파이프라인
1단계: 데이터 불러오기
2단계: Train/Test 분리 (8:2)
3단계: Data Scaling (표준화)
4단계: Feature Selection (RF 기반 Top-K)
5단계: 8개 ML 모델 학습 (기본 하이퍼파라미터)
6단계: ML Evaluation (성능 평가 지표 계산)
7단계: 엑셀 저장 및 시각화
8단계: XAI (LIME + SHAP)
"""

import os
import glob
import warnings
import random
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import (confusion_matrix, precision_score, recall_score, f1_score, matthews_corrcoef, roc_auc_score, balanced_accuracy_score, roc_curve, auc)
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier
from catboost import CatBoostClassifier
from lime.lime_tabular import LimeTabularExplainer
import joblib


# =====================================================
# 전역 설정
# =====================================================
RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)
random.seed(RANDOM_STATE)

RESULT_DIR = "./result"
os.makedirs(RESULT_DIR, exist_ok=True)

warnings.filterwarnings("ignore")


# =====================================================
# 1단계: 데이터 불러오기
# =====================================================
def load_data():
    print("\n[1단계] 데이터 불러오기")

    files = glob.glob("./features_xlsx/*.xlsx")
    print(f"찾은 파일 수: {len(files)}")

    if len(files) == 0:
        raise FileNotFoundError("❌ features_xlsx 폴더에 엑셀 파일이 없습니다.")

    # 여러 개의 엑셀 파일을 하나로 합치기
    df_list = [pd.read_excel(f, sheet_name=0) for f in files]
    df = pd.concat(df_list, ignore_index=True)

    # label 인코딩 (문자 → 0/1)
    le = LabelEncoder()
    y = le.fit_transform(df["label"])
    class_names = list(le.classes_)
    print(f"클래스 분포: {dict(zip(class_names, np.bincount(y)))}")

    # 숫자형 feature만 사용 (label은 제외)
    feature_cols = df.select_dtypes(include=["float64", "int64"]).columns
    feature_cols = feature_cols.drop("label", errors="ignore")
    X = df[feature_cols]

    print(f"Feature 개수: {len(feature_cols)}")
    print(f"결측치 개수: {X.isnull().sum().sum()}")

    return X, y, list(feature_cols), class_names


# =====================================================
# 2단계: Train/Test 분리
# =====================================================
def split_data(X, y, test_size=0.2):
    print("\n[2단계] Train/Test 분리 (8:2)")

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        stratify=y,
        random_state=RANDOM_STATE
    )

    print(f"Train 샘플 수: {len(X_train)}")
    print(f"Test 샘플 수:  {len(X_test)}")
    print(f"Train 클래스 분포: {np.bincount(y_train)}")
    print(f"Test  클래스 분포: {np.bincount(y_test)}")

    return X_train, X_test, y_train, y_test


# =====================================================
# 3단계: Data Scaling
# =====================================================
def scale_data(X_train, X_test):
    print("\n[3단계] Data Scaling")

    # 결측치는 각 컬럼의 중앙값으로 채우기
    X_train_filled = X_train.fillna(X_train.median())
    X_test_filled = X_test.fillna(X_train.median())  # Train 기준으로 채우기

    scaler = StandardScaler()
    scaler.fit(X_train_filled)

    X_train_scaled = pd.DataFrame(
        scaler.transform(X_train_filled),
        columns=X_train.columns,
        index=X_train.index
    )

    X_test_scaled = pd.DataFrame(
        scaler.transform(X_test_filled),
        columns=X_test.columns,
        index=X_test.index
    )

    print("Scaling 완료 (평균 0, 표준편차 1 기준)")

    return X_train_scaled, X_test_scaled, scaler


# =====================================================
# 4단계: Feature Selection (RF 기반 Top-K)
# =====================================================
def rf_importance_elbow(X_train, y_train, plot_path=None):
    """
    1) RF로 feature importance 계산
    2) 중요도 내림차순 정렬
    3) 중요도 차이(derivative) 계산
    4) 가장 큰 변화량(drop)이 있는 지점 → elbow point = 최적 K
    """

    print("\n[4단계] Feature Selection")

    rf = RandomForestClassifier(n_estimators=600, random_state=42, n_jobs=-1)
    rf.fit(X_train, y_train)

    importances = rf.feature_importances_
    idx_sorted = np.argsort(importances)[::-1]

    sorted_imp = importances[idx_sorted]
    sorted_feat = X_train.columns[idx_sorted]

    # 기울기(변화량) 계산
    diffs = np.diff(sorted_imp)

    # 가장 크게 떨어진 지점 = elbow
    elbow_k = np.argmin(diffs) + 1
    elbow_k = max(3, elbow_k)  # 최소 3개 이상 보장

    selected_features = list(sorted_feat[:elbow_k])

    # Plot 저장
    if plot_path:
        plt.figure(figsize=(7, 5))
        plt.plot(sorted_imp, marker="o")
        plt.axvline(elbow_k, color="red", linestyle="--", label=f"Elbow K={elbow_k}")
        plt.title("Random Forest Feature Importance Curve")
        plt.xlabel("Feature Rank")
        plt.ylabel("Importance")
        plt.legend()
        plt.tight_layout()
        plt.savefig(plot_path, dpi=300)
        plt.close()

    return selected_features, sorted_feat, sorted_imp, elbow_k


# =====================================================
# 5단계: 8개 ML 모델 정의 (기본 하이퍼파라미터)
# =====================================================
def get_models():
    print("\n[5단계] ML 모델 생성")

    models = {
        "Logistic Regression": LogisticRegression(
            max_iter=1000,
            random_state=RANDOM_STATE
        ),
        "KNN": KNeighborsClassifier(),
        "SVM": SVC(
            probability=True,
            random_state=RANDOM_STATE
        ),
        "Decision Tree": DecisionTreeClassifier(
            random_state=RANDOM_STATE
        ),
        "Random Forest": RandomForestClassifier(
            random_state=RANDOM_STATE,
            n_estimators=200,
            n_jobs=-1
        ),
        "LightGBM": LGBMClassifier(
            random_state=RANDOM_STATE,
            n_jobs=-1
        ),
        "XGBoost": XGBClassifier(
            random_state=RANDOM_STATE,
            eval_metric="logloss",
            n_jobs=-1,
            use_label_encoder=False
        ),
        "CatBoost": CatBoostClassifier(
            random_state=RANDOM_STATE,
            verbose=False
        ),
    }

    print(f"모델 개수: {len(models)}개")
    return models


# =====================================================
# 6단계: 모델 학습 + 평가
# =====================================================
def evaluate_models(models, X_train, y_train, X_test, y_test):
    print("\n[6단계] 모델 학습 및 평가")

    results_list = []
    y_proba_dict = {}
    model_objects = {}

    for name, model in models.items():
        print(f"\n⚡ Training: {name}")

        # 1) 모델 학습
        model.fit(X_train, y_train)

        # 2) 예측 (라벨, 확률)
        y_pred = model.predict(X_test)
        # 이진분류라고 가정하고, 양성 클래스(1)의 확률만 사용
        y_proba = model.predict_proba(X_test)[:, 1]

        # 3) 성능 지표 계산
        cm = confusion_matrix(y_test, y_pred)
        tn, fp, fn, tp = cm.ravel()

        accuracy = (y_pred == y_test).mean()
        precision = precision_score(y_test, y_pred, zero_division=0)
        recall = recall_score(y_test, y_pred, zero_division=0)  # sensitivity
        f1 = f1_score(y_test, y_pred, zero_division=0)
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
        sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
        bal_acc = balanced_accuracy_score(y_test, y_pred)
        mcc = matthews_corrcoef(y_test, y_pred)
        auc_score = roc_auc_score(y_test, y_proba)

        result = {
            "Model": name,
            "Accuracy": accuracy,
            "Precision": precision,
            "Recall": recall,
            "F1": f1,
            "Specificity": specificity,
            "Sensitivity": sensitivity,
            "Balanced_Accuracy": bal_acc,
            "MCC": mcc,
            "AUC": auc_score
        }
        results_list.append(result)

        y_proba_dict[name] = y_proba
        model_objects[name] = model

        print(f"   - Accuracy: {accuracy:.3f}, F1: {f1:.3f}, MCC: {mcc:.3f}, AUC: {auc_score:.3f}")

    # MCC 기준으로 Best Model 선정
    df_results = pd.DataFrame(results_list)
    best_idx = df_results["MCC"].idxmax()
    best_model_name = df_results.loc[best_idx, "Model"]
    best_model = model_objects[best_model_name]

    print(f"\n✅ Best Model (MCC 기준): {best_model_name}")

    return results_list, y_proba_dict, best_model_name, best_model


# =====================================================
# 7단계: 엑셀 저장 및 시각화
# =====================================================
def save_results_and_plots(results_list, y_test, y_proba_dict, best_model_name, best_model, X_test_fs, selected_features, class_names):
    print("\n[7단계] 엑셀 저장 및 시각화")

    df_results = pd.DataFrame(results_list)
    df_results_sorted = df_results.sort_values("MCC", ascending=False)

    # 1) 엑셀 저장
    excel_path = os.path.join(RESULT_DIR, "final_results.xlsx")
    df_results_sorted.to_excel(excel_path, index=False)
    print(f"성능 지표 엑셀 저장 완료: {excel_path}")

    # 2) Confusion Matrix (Best Model)
    y_pred_best = best_model.predict(X_test_fs)
    cm = confusion_matrix(y_test, y_pred_best)

    plt.figure(figsize=(7, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=class_names, yticklabels=class_names)
    plt.title(f"Confusion Matrix - {best_model_name}")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.tight_layout()
    cm_path = os.path.join(RESULT_DIR, "confusion_matrix_best.png")
    plt.savefig(cm_path, dpi=300)
    plt.close()
    print(f"Confusion Matrix 저장: {cm_path}")

    # 3) ROC Curve (모든 모델 비교)
    plt.figure(figsize=(7, 6))
    for name, y_proba in y_proba_dict.items():
        fpr, tpr, _ = roc_curve(y_test, y_proba)
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, lw=2, label=f"{name} (AUC={roc_auc:.3f})")

    plt.plot([0, 1], [0, 1], "k--", label="Random")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve Comparison (All Models)")
    plt.legend(loc="lower right", fontsize=8)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    roc_all_path = os.path.join(RESULT_DIR, "roc_all_models.png")
    plt.savefig(roc_all_path, dpi=300)
    plt.close()
    print(f"ROC Curve 저장: {roc_all_path}")

    # 4) Feature Importance (지원 안되면 Permutation Importance 사용)
    if hasattr(best_model, "feature_importances_"):
        # Tree 모델 Feature Importance
        importances = best_model.feature_importances_
        df_fi = pd.DataFrame({
            "Feature": selected_features,
            "Importance": importances
        })
    else:
        # Permutation Importance로 대체
        from sklearn.inspection import permutation_importance

        perm = permutation_importance(
            best_model,
            X_test_fs,
            y_test,
            scoring="matthews_corrcoef",
            n_repeats=10,
            random_state=42
        )
        df_fi = pd.DataFrame({
            "Feature": selected_features,
            "Importance": perm.importances_mean
        })


# =====================================================
# 8단계: XAI (LIME + SHAP)
# =====================================================
def run_xai(best_model, X_train_fs, X_test_fs, selected_features, class_names):
    """
    Best Model을 대상으로 LIME, SHAP 실행.
    - LIME: 개별 샘플에 대한 국소(local) 설명
    - SHAP: 전체적인(global) feature 중요도 설명 (Tree 기반 모델에서)
    """
    print("\n[8단계] XAI (LIME + SHAP) 실행")

    # ---------- LIME ----------
    print("LIME 실행 중...")
    try:
        explainer = LimeTabularExplainer(
            training_data=np.array(X_train_fs),
            feature_names=selected_features,
            class_names=class_names,
            mode="classification"
        )

        # 첫 번째 테스트 샘플 선택
        sample = X_test_fs.iloc[0].values

        def predict_fn(x):
            return best_model.predict_proba(x)

        exp = explainer.explain_instance(sample, predict_fn)
        lime_path = os.path.join(RESULT_DIR, "lime_explanation_best.html")
        exp.save_to_file(lime_path)
        print(f"LIME 결과 저장: {lime_path}")
    except Exception as e:
        print(f"❌ LIME 실행 실패: {e}")


# =====================================================
# 9단계: 베스트 모델 알고리즘 저장
# =====================================================
def save_final_artifacts(best_model, scaler, selected_features):
    print("\n[저장 단계] 모델 / 스케일러 / 피처 저장")

    model_path = "./result/best_model.pkl"
    scaler_path = "./result/scaler.pkl"
    features_path = "./result/selected_features.pkl"

    joblib.dump(best_model, model_path)
    joblib.dump(scaler, scaler_path)
    joblib.dump(selected_features, features_path)

    print(f"✅ 모델 저장 완료: {model_path}")
    print(f"✅ 스케일러 저장 완료: {scaler_path}")
    print(f"✅ 피처 저장 완료: {features_path}")


# =====================================================
# MAIN: 전체 파이프라인 실행
# =====================================================
def main():
    print("\n============================================")
    print("🚀 머신러닝 파이프라인 시작")
    print("============================================")

    # 1단계: 데이터 불러오기
    X, y, feature_names, class_names = load_data()

    # 2단계: Train/Test 분리
    X_train, X_test, y_train, y_test = split_data(X, y, test_size=0.2)

    # 3단계: Scaling
    X_train_scaled, X_test_scaled, scaler = scale_data(X_train, X_test)

    # 4단계: Feature Selection (RF Elbow 적용)
    selected_features, sorted_feat, sorted_imp, K = rf_importance_elbow(
        X_train_scaled, y_train, "./result/rf_importance_curve.png"
    )
    print(f"Selected Feature({K}개): {selected_features}")

    X_train_fs = X_train_scaled[selected_features]
    X_test_fs = X_test_scaled[selected_features]

    # 5단계: 모델 생성
    models = get_models()

    # 6단계: 모델 학습 + 평가
    results_list, y_proba_dict, best_model_name, best_model = evaluate_models(
        models, X_train_fs, y_train, X_test_fs, y_test
    )

    # 7단계: 시각화 및 엑셀 저장
    save_results_and_plots(
        results_list, y_test, y_proba_dict,
        best_model_name, best_model,
        X_test_fs, selected_features, class_names
    )

    # 8단계: XAI (LIME)
    run_xai(best_model, X_train_fs, X_test_fs, selected_features, class_names)

    # 9단계: 서버 저장
    save_final_artifacts(best_model, scaler, selected_features)

    print("\n🎉 전체 작업 완료! result 폴더를 확인하세요.")


# =====================================================
# 스크립트 실행
# =====================================================
if __name__ == "__main__":
    main()
