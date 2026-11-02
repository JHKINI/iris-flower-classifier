from flask import Flask, render_template, request, send_from_directory, abort
import pickle
import numpy as np
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
IMAGE_DIR = os.path.join(BASE_DIR, "img")
MODEL_PATH = os.path.join(BASE_DIR, "model", "iris_model.pkl")

app = Flask(__name__)

with open(MODEL_PATH, "rb") as f:
    model = pickle.load(f)

iris_info = {
    0: {"name": "setosa", "image": "setosa.png", "korean": "세토사"},
    1: {"name": "versicolor", "image": "versicolor.png", "korean": "버시컬러"},
    2: {"name": "virginica", "image": "virginica.png", "korean": "버지니카"},
}

@app.route("/img/<path:filename>")
def flower_image(filename):
    image_path = os.path.join(IMAGE_DIR, filename)
    if not os.path.isfile(image_path):
        abort(404)
    return send_from_directory(IMAGE_DIR, filename)

@app.route("/", methods=["GET", "POST"])
def index():
    result = None
    error = None
    if request.method == "POST":
        try:
            values = [
                float(request.form["sepal_length"]),
                float(request.form["sepal_width"]),
                float(request.form["petal_length"]),
                float(request.form["petal_width"]),
            ]
            pred = int(model.predict(np.array([values]))[0])
            result = iris_info[pred]
        except (ValueError, KeyError):
            error = "네 가지 측정값을 숫자로 입력해 주세요."
        except Exception as e:
            error = f"예측 중 오류가 발생했습니다: {e}"
    return render_template("index.html", result=result, error=error)

if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=True)
