import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score

# 데이터 생성을 위한 함수
def make_wave(n_samples=100):
    rnd = np.random.RandomState(42)
    x = rnd.uniform(-3, 3, size=(n_samples,))
    y_no_noise = np.sin((4 * x) + x)
    y = (y_no_noise + rnd.normal(size=n_samples)) / 2
    return x.reshape(-1, 1), y

# 데이터 생성 및 분할
x, y = make_wave(100)

x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=0)

# 테스트 할 k 값 리스트트
k_list = [1, 3, 5, 7, 10]
scores = []


plt.figure(figsize=(15, 10))

# 예측용 선형 공간간
x_line = np.linspace(-3, 3, 1000).reshape(-1, 1)

# KNeighborsRegressor 모델을 사용하여 k 값에 따른 회귀 모델을 학습하고 예측 및 시각화
for i, k in enumerate(k_list, 1):
    knn = KNeighborsRegressor(n_neighbors=k)
    knn.fit(x_train, y_train)
    y_pred = knn.predict(x_test)
    r2 = r2_score(y_test, y_pred)
    scores.append(r2)

    
    plt.subplot(2, 3, i)
    y_line = knn.predict(x_line)
    plt.plot(x_line, y_line, label=f"k={k}")
    plt.scatter(x_train, y_train, alpha=0.6, label="train")
    plt.scatter(x_test, y_test, alpha=0.6, label="test")
    plt.title(f"k={k} | R²: {r2:.3f}") #성능 출력
    plt.legend()

plt.tight_layout()
plt.show()