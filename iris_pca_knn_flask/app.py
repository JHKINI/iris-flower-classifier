import json
import os
import pickle

import numpy as np
from flask import Flask, abort, render_template, request, send_from_directory

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "model", "iris_pca_knn_pipeline.pkl")
METRICS_PATH = os.path.join(BASE_DIR, "model", "evaluation.json")
IMAGE_DIR = os.path.join(BASE_DIR, "img")

app = Flask(__name__)

# Pipeline 전체를 불러온다.
# 사용자의 원본 4개 측정값은 Pipeline 내부에서
# 학습 당시의 StandardScaler -> PCA -> KNN 순서로 자동 처리된다.
with open(MODEL_PATH, "rb") as f:
    model_pipeline = pickle.load(f)

with open(METRICS_PATH, "r", encoding="utf-8") as f:
    evaluation = json.load(f)

IRIS_INFO = {
    0: {"name": "Setosa", "korean": "세토사", "image": "setosa.png"},
    1: {"name": "Versicolor", "korean": "버시컬러", "image": "versicolor.png"},
    2: {"name": "Virginica", "korean": "버지니카", "image": "virginica.png"},
}

FEATURES = [
    ("sepal_length", "꽃받침 길이 (cm)"),
    ("sepal_width", "꽃받침 너비 (cm)"),
    ("petal_length", "꽃잎 길이 (cm)"),
    ("petal_width", "꽃잎 너비 (cm)"),
]


@app.route("/img/<path:filename>")
def flower_image(filename):
    image_path = os.path.join(IMAGE_DIR, filename)
    if not os.path.isfile(image_path):
        abort(404)
    return send_from_directory(IMAGE_DIR, filename)


@app.route("/", methods=["GET", "POST"])
def index():
    prediction = None
    error = None
    input_values = {key: "" for key, _ in FEATURES}

    if request.method == "POST":
        try:
            values = []
            for key, _ in FEATURES:
                raw_value = request.form.get(key, "").strip()
                input_values[key] = raw_value
                value = float(raw_value)
                if not np.isfinite(value) or value <= 0:
                    raise ValueError("측정값은 0보다 큰 숫자로 입력해 주세요.")
                values.append(value)

            # 원본 4개 컬럼을 전달하면 저장된 Pipeline이
            # scaler.transform -> pca.transform -> knn.predict를 수행한다.
            input_array = np.asarray([values], dtype=float)
            predicted_class = int(model_pipeline.predict(input_array)[0])
            prediction = IRIS_INFO[predicted_class]

        except (ValueError, TypeError):
            error = "네 가지 측정값을 모두 0보다 큰 숫자로 입력해 주세요."
        except Exception:
            app.logger.exception("Prediction failed")
            error = "예측 중 오류가 발생했습니다. 서버 로그를 확인해 주세요."

    return render_template(
        "index.html",
        prediction=prediction,
        error=error,
        input_values=input_values,
        features=FEATURES,
        evaluation=evaluation,
    )


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=True)
