import os
import glob
import warnings
import random
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from sklearn.feature_selection import RFECV
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    confusion_matrix, precision_score, recall_score, f1_score,
    matthews_corrcoef, roc_auc_score, balanced_accuracy_score,
    roc_curve, auc
)
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
def data_loading():
    print("\n[1단계] Data Loading")

    files = glob.glob("./features_xlsx/*.xlsx")
    print(f"찾은 파일 수: {len(files)}")

    if len(files) == 0:
        raise FileNotFoundError("❌ 폴더에 엑셀 파일이 없습니다.")

    df_list = [pd.read_excel(f) for f in files]
    df = pd.concat(df_list, ignore_index=True)

    if "label" not in df.columns:
        raise KeyError("❌ label 컬럼이 없습니다.")

    y = df["label"].astype(int).values
    class_names = ["Intermediate", "Advanced"]
    label_counts = np.bincount(y)

    # 수치형 feature 자동 선택
    feature_cols = df.select_dtypes(include=["float64", "int64"]).columns
    feature_cols = feature_cols.drop("label", errors="ignore")

    X = df[feature_cols]

    # 결측치 확인
    total_missing = X.isnull().sum().sum()
    print(f"결측치 개수: {total_missing}")

    if total_missing > 0:
        print("⚠️ 경고: 결측치가 존재합니다. \n")

    print(f"Feature 개수: {len(feature_cols)}")
    print(f"사용 클래스: {class_names}")
    print(f"클래스 분포: 0 - {label_counts[0]}개, 1 - {label_counts[1]}개")

    return X, y, list(feature_cols), class_names


# =====================================================
# 2단계: Data Split (Train/Test)
# =====================================================
def data_split(X, y, test_size=0.2):
    print("\n[2단계] Data Split (Train/Test)")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=test_size,
        stratify=y,
        random_state=RANDOM_STATE
    )

    print(f"Train 샘플 수: {len(X_train)}")
    print(f"Test  샘플 수: {len(X_test)}")
    print(f"Train 클래스 분포: {np.bincount(y_train)}")
    print(f"Test  클래스 분포: {np.bincount(y_test)}")

    return X_train, X_test, y_train, y_test


# =====================================================
# 3단계: Feature Selection
# =====================================================
def feature_selection_rfecv_rf(X_train, y_train, min_features=5):
    print("\n[3단계] Feature Selection (RFECV - RandomForest 기반)")

    # Scaling (트리에 꼭 필요하진 않지만 전체 파이프라인 일관성 유지)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)

    base_estimator = RandomForestClassifier(
        n_estimators=300,
        random_state=RANDOM_STATE,
        n_jobs=-1
    )

    cv = StratifiedKFold(
        n_splits=5, shuffle=True, random_state=RANDOM_STATE
    )

    rfecv = RFECV(
        estimator=base_estimator,
        step=1,
        cv=cv,
        scoring="matthews_corrcoef",
        min_features_to_select=min_features,
        n_jobs=-1
    )

    rfecv.fit(X_train_scaled, y_train)

    selected_features = X_train.columns[rfecv.support_]
    print(f"선택된 Feature 수: {len(selected_features)}")
    print("Selected:", list(selected_features))

    return list(selected_features)


# =====================================================
# Feature Scaling
# =====================================================
def feature_scaling(X_train, X_test, selected_features, model_name):
    scaling_required = ["Logistic Regression", "KNN", "SVM"]

    if model_name not in scaling_required:
        print(f"⚠ 스케일링 생략: {model_name}")
        return X_train[selected_features], X_test[selected_features], None

    print(f"✔ 스케일링 적용: {model_name}")

    scaler = StandardScaler()
    scaler.fit(X_train[selected_features])

    X_train_scaled = pd.DataFrame(
        scaler.transform(X_train[selected_features]),
        columns=selected_features,
        index=X_train.index
    )

    X_test_scaled = pd.DataFrame(
        scaler.transform(X_test[selected_features]),
        columns=selected_features,
        index=X_test.index
    )

    return X_train_scaled, X_test_scaled, scaler


# =====================================================
# 4단계: ML 모델 정의
# =====================================================
def model_development():
    print("\n[4단계] ML 모델 생성")

    models = {
        # Linear models: 데이터가 선형적으로 분리될 때 효과적, 해석 가능성 높고, 속도 빠름
        "Logistic Regression": LogisticRegression(
            max_iter=1000,
            random_state=RANDOM_STATE
        ),
        # Non-Linear models (Distance-based / Kernel-based): 복잡한 결정경계를 학습 가능
        "KNN": KNeighborsClassifier(),
        "SVM": SVC(
            probability=True,  # 확률 기반 예측 → XAI 용도
            random_state=RANDOM_STATE
        ),
        # Tree models
        "Decision Tree": DecisionTreeClassifier(
            random_state=RANDOM_STATE
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=200,
            random_state=RANDOM_STATE,
            n_jobs=-1
        ),
        # Gradient boosting models: 작은 데이터에서도 강력한 성능, 복잡한 패턴 학습에 적합
        "LightGBM": LGBMClassifier(
            random_state=RANDOM_STATE,
            n_jobs=-1,
            verbose=-1
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
        )
    }

    print(f"모델 개수: {len(models)}개")
    return models


# =====================================================
# 5단계: 모델 학습 및 평가
# =====================================================
def model_evaluation(model, X_train, y_train, X_test, y_test, model_name):
    # 1) 모델 학습
    model.fit(X_train, y_train)

    # 2) 예측
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    # 3) 성능 지표 계산
    cm = confusion_matrix(y_test, y_pred)
    tn, fp, fn, tp = cm.ravel()

    accuracy = (y_pred == y_test).mean()
    precision = precision_score(y_test, y_pred, zero_division=0)
    recall = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
    bal_acc = balanced_accuracy_score(y_test, y_pred)
    mcc = matthews_corrcoef(y_test, y_pred)
    auc_score = roc_auc_score(y_test, y_proba)

    metrics = {
        "Model": model_name,
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

    print(f"- Accuracy: {accuracy:.3f}, F1: {f1:.3f}, MCC: {mcc:.3f}, AUC: {auc_score:.3f} \n")

    return model, y_pred, y_proba, mcc, metrics


# =====================================================
# 6단계: 엑셀 저장 및 시각화
# =====================================================
def save_results(results_list, y_test, y_proba_dict, best_model_name, best_model, X_test_fs, selected_features, class_names):
    print("\n[6단계] 엑셀 저장 및 시각화")

    df_results = pd.DataFrame(results_list)
    df_results_sorted = df_results.sort_values("MCC", ascending=False)

    # 1) 엑셀 저장
    excel_path = os.path.join(RESULT_DIR, "final_results.xlsx")
    df_results_sorted.to_excel(excel_path, index=False)
    print(f"성능 지표 엑셀 저장 완료: {excel_path}")

    # 2) Confusion Matrix (Best Model)
    y_pred_best = best_model.predict(X_test_fs)
    cm = confusion_matrix(y_test, y_pred_best)

    plt.figure(figsize=(8, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=class_names, yticklabels=class_names)
    plt.title(f"Confusion Matrix - {best_model_name}")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.tight_layout()
    cm_path = os.path.join(RESULT_DIR, "best_confusion_matrix.png")
    plt.savefig(cm_path, dpi=300)
    plt.close()
    print(f"Confusion Matrix 저장: {cm_path}")

    # 3) ROC Curve (모든 모델 비교)
    plt.figure(figsize=(8, 5))
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
    roc_all_path = os.path.join(RESULT_DIR, "all_roc_curve.png")
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
# 7단계: XAI (LIME)
# =====================================================
def xai_lime(best_model, X_train_fs, X_test_fs, selected_features, class_names):
    print("\n[7단계] XAI (LIME)")

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
        lime_path = os.path.join(RESULT_DIR, "best_lime_explanation.html")
        exp.save_to_file(lime_path)
        print(f"LIME 결과 저장: {lime_path}")

    except Exception as e:
        print(f"❌ LIME 실행 실패: {e}")


# =====================================================
# 8단계: 알고리즘 저장
# =====================================================
def save_algorithm(best_model, scaler, selected_features):
    print("\n[8단계] 알고리즘 저장")

    model_path = "./result/best_model.pkl"
    scaler_path = "./result/best_scaler.pkl"
    features_path = "./result/best_features.pkl"

    joblib.dump(best_model, model_path)
    joblib.dump(scaler, scaler_path)
    joblib.dump(selected_features, features_path)

    print(f"모델 저장 완료: {model_path}")
    print(f"스케일러 저장 완료: {scaler_path}")
    print(f"피처 저장 완료: {features_path}")


# =====================================================
# MAIN: 머신러닝 파이프라인
# =====================================================
def main():
    print("\n============================================")
    print("🚀 머신러닝 파이프라인 시작")
    print("============================================")

    # 1단계: 데이터 불러오기
    X, y, feature_names, class_names = data_loading()

    # 2단계: Data Split (Train/Test)
    X_train, X_test, y_train, y_test = data_split(X, y)

    # 3단계: Feature Selection
    selected_features = feature_selection_rfecv_rf(
        X_train, y_train
    )
    print(selected_features)

    # 4단계: ML 모델 정의
    models = model_development()

    best_model = None
    best_model_name = None
    best_mcc = -1
    best_scaler = None
    best_X_train_fs = None
    best_X_test_fs = None

    results_list = []
    y_proba_dict = {}

    # 5단계: 모델 학습 및 평가
    print("\n[5단계] 모델 학습 및 평가")

    for model_name, model in models.items():
        X_train_fs, X_test_fs, scaler = feature_scaling(
            X_train, X_test, selected_features, model_name
        )

        model, y_pred, y_proba, mcc, metrics = model_evaluation(
            model, X_train_fs, y_train, X_test_fs, y_test, model_name
        )

        # 결과 저장용
        results_list.append(metrics)
        y_proba_dict[model_name] = y_proba

        # Best Model 갱신
        if mcc > best_mcc:
            best_mcc = mcc
            best_model_name = model_name
            best_model = model
            best_scaler = scaler
            best_X_train_fs = X_train_fs
            best_X_test_fs = X_test_fs

    print(f"✅ Best Model (MCC 기준): {best_model_name}")

    # 6단계: 엑셀 저장 및 시각화
    save_results(
        results_list=results_list,
        y_test=y_test,
        y_proba_dict=y_proba_dict,
        best_model_name=best_model_name,
        best_model=best_model,
        X_test_fs=best_X_test_fs,
        selected_features=selected_features,
        class_names=class_names
    )

    # 7단계: XAI (LIME)
    xai_lime(best_model, best_X_train_fs, best_X_test_fs, selected_features, class_names)

    # 8단계: 알고리즘 저장
    save_algorithm(best_model, best_scaler, selected_features)

    print("\n🎉 전체 파이프라인 완료!")


# =====================================================
# MAIN 실행
# =====================================================
if __name__ == "__main__":
    main()
