# Iris Flower Classifier (KNN)

![붓꽃 품종 예측 웹 페이지 실행 결과](result.png)

꽃받침과 꽃잎의 길이·너비를 입력하면 KNN(K-Nearest Neighbors) 알고리즘으로 붓꽃 품종을 예측하고, 예측된 품종의 이미지를 웹 페이지에 표시하는 Flask 프로젝트입니다.

## 주요 기능
- 꽃받침 길이·너비와 꽃잎 길이·너비 입력
- 학습된 KNN 모델을 이용한 붓꽃 품종 예측
- 예측 결과에 따라 해당 품종 이미지 표시
  - Setosa (세토사)
  - Versicolor (버시컬러)
  - Virginica (버지니카)

## 기술 스택

| 구분 | 기술 |
|---|---|
| Language | Python |
| Web Framework | Flask |
| Machine Learning | scikit-learn |
| Model | K-Nearest Neighbors (KNN) |
| Data | scikit-learn Iris Dataset |
| Numerical Processing | NumPy |
| Model Storage | Pickle |
| Frontend | HTML, CSS |

## 프로젝트 구조

```text
iris_knn_flask/
├── app.py                  # Flask 서버 및 품종 예측 처리
├── train_model.py          # Iris 데이터 학습 및 모델 저장
├── model/
│   └── iris_model.pkl      # 학습된 KNN 모델
├── img/
│   ├── setosa.png
│   ├── versicolor.png
│   └── virginica.png
├── templates/
│   └── index.html          # 입력 폼 및 예측 결과 화면
├── result.png              # 실행 결과 스크린샷
└── README.md
```

## 실행 방법

### 1. 저장소 복제

```bash
git clone https://github.com/깃허브아이디/iris-flower-classifier.git
cd iris-flower-classifier
```

### 2. 라이브러리 설치

```bash
pip install flask numpy scikit-learn
```

### 3. Flask 서버 실행

```bash
python app.py
```

### 4. 웹 페이지 접속

브라우저에서 `http://127.0.0.1:5000`으로 접속합니다.

## 모델 학습

모델을 다시 학습하려면 프로젝트 루트 폴더에서 실행합니다.

```bash
python train_model.py
```

학습된 모델은 `model/iris_model.pkl`에 저장됩니다.

## 예측 과정

1. 사용자가 붓꽃의 네 가지 측정값을 입력합니다.
2. Flask가 입력값을 학습된 KNN 모델에 전달합니다.
3. 모델이 Setosa, Versicolor, Virginica 중 하나를 예측합니다.
4. 예측 결과에 해당하는 품종명과 이미지 파일을 연결합니다.
5. 결과 페이지에 예측 품종과 해당 붓꽃 이미지가 표시됩니다.

## 이미지 제공 방식

붓꽃 이미지는 `img/` 폴더에 저장되어 있습니다. Flask의 `/img/<파일명>` 경로와 `send_from_directory()`를 통해 브라우저에 이미지를 제공합니다.

## 참고
- scikit-learn에서 제공하는 Iris Dataset을 사용합니다.
- KNN 모델의 이웃 수는 `n_neighbors=3`으로 설정되어 있습니다.
- 학습 데이터와 테스트 데이터는 8:2 비율로 분리합니다.
- README 상단의 실행 화면을 표시하려면 결과 스크린샷 파일을 프로젝트 최상위 폴더에 `result.png`라는 이름으로 넣어야 합니다.
