"""
Iris PCA + KNN 학습 스크립트

데이터 누수 방지:
1) 먼저 Train/Test를 분리한다.
2) Pipeline 안의 StandardScaler와 PCA는 train 데이터에 대해서만 fit된다.
3) test 데이터는 학습된 scaler/PCA를 거쳐 transform만 수행된다.
"""
import json
import os
import pickle

import numpy as np
from sklearn import datasets
from sklearn.decomposition import PCA
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
)
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "model")
MODEL_PATH = os.path.join(MODEL_DIR, "iris_pca_knn_pipeline.pkl")
METRICS_PATH = os.path.join(MODEL_DIR, "evaluation.json")

RANDOM_STATE = 42
TEST_SIZE = 0.2
PCA_COMPONENTS = 2
N_NEIGHBORS = 3


def main():
    iris = datasets.load_iris()
    X = iris.data
    y = iris.target

    # 1. 데이터 분리: 품종별 비율을 유지하고, 테스트 데이터는 이후 학습에 사용하지 않는다.
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    # 2. 하나의 Pipeline으로 전처리와 모델을 묶어 저장한다.
    # fit()은 X_train에만 호출되므로 Scaler/PCA 모두 Train 데이터만 학습한다.
    pipeline = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            ("pca", PCA(n_components=PCA_COMPONENTS)),
            ("knn", KNeighborsClassifier(n_neighbors=N_NEIGHBORS)),
        ]
    )
    pipeline.fit(X_train, y_train)

    # 3. Test에는 fit/fit_transform을 호출하지 않고 predict만 수행한다.
    y_pred = pipeline.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(
        y_test,
        y_pred,
        target_names=iris.target_names,
        output_dict=True,
        zero_division=0,
    )
    matrix = confusion_matrix(y_test, y_pred).tolist()

    # PCA가 Train 데이터에서 설명하는 분산 비율
    explained_variance = (
        pipeline.named_steps["pca"].explained_variance_ratio_.tolist()
    )

    metrics = {
        "accuracy": round(float(accuracy), 4),
        "test_size": TEST_SIZE,
        "train_samples": int(len(X_train)),
        "test_samples": int(len(X_test)),
        "pca_components": PCA_COMPONENTS,
        "pca_explained_variance_ratio": [
            round(float(v), 4) for v in explained_variance
        ],
        "pca_cumulative_explained_variance": round(
            float(np.sum(explained_variance)), 4
        ),
        "n_neighbors": N_NEIGHBORS,
        "random_state": RANDOM_STATE,
        "confusion_matrix": matrix,
        "classification_report": report,
        "class_names": iris.target_names.tolist(),
    }

    os.makedirs(MODEL_DIR, exist_ok=True)
    with open(MODEL_PATH, "wb") as f:
        pickle.dump(pipeline, f)

    with open(METRICS_PATH, "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)

    print("=== Iris PCA + KNN 평가 결과 ===")
    print(f"Train samples: {len(X_train)}")
    print(f"Test samples : {len(X_test)}")
    print(f"PCA components: {PCA_COMPONENTS}")
    print(f"Explained variance ratio: {explained_variance}")
    print(f"Cumulative explained variance: {sum(explained_variance):.4f}")
    print(f"Test accuracy: {accuracy:.4f}")
    print("\nClassification report:")
    print(classification_report(y_test, y_pred, target_names=iris.target_names))
    print("Confusion matrix:")
    print(np.array(matrix))
    print(f"\nPipeline saved: {MODEL_PATH}")
    print(f"Metrics saved : {METRICS_PATH}")


if __name__ == "__main__":
    main()
