import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.linear_model import Ridge, Lasso, LinearRegression


# Load the dataset
df = pd.read_csv('python/BostonHousing.csv')  # 경로 수정

# Split features and target
X = df.drop('medv', axis=1)
y = df['medv']

# Add polynomial features
poly = PolynomialFeatures(degree=3, include_bias=False)
X_poly = poly.fit_transform(X)

# Split into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X_poly, y, random_state=30)

# Standardize the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Define alpha values
alpha_values = [0.001, 0.01, 0.1, 1, 10, 100]

# Store results
results = []

# 일반 선형 회귀 (alpha = 0)
lr = LinearRegression()
lr.fit(X_train_scaled, y_train)
lr_train_score = lr.score(X_train_scaled, y_train)
lr_test_score = lr.score(X_test_scaled, y_test)

results.append({
    'alpha': 0,
    'ridge_train_score': lr_train_score,
    'ridge_test_score': lr_test_score,
    'lasso_train_score': lr_train_score,
    'lasso_test_score': lr_test_score
})

for alpha in alpha_values:
    # Ridge
    ridge = Ridge(alpha=alpha)
    ridge.fit(X_train_scaled, y_train)
    ridge_train_score = ridge.score(X_train_scaled, y_train)
    ridge_test_score = ridge.score(X_test_scaled, y_test)

    # Lasso
    lasso = Lasso(alpha=alpha, max_iter=1000000)
    lasso.fit(X_train_scaled, y_train)
    lasso_train_score = lasso.score(X_train_scaled, y_train)
    lasso_test_score = lasso.score(X_test_scaled, y_test)

    # 결과 저장
    results.append({
        'alpha': alpha,
        'ridge_train_score': ridge_train_score,
        'ridge_test_score': ridge_test_score,
        'lasso_train_score': lasso_train_score,
        'lasso_test_score': lasso_test_score
    })

# Convert to DataFrame
results_df = pd.DataFrame(results)

# Display the DataFrame in the console
print("\n📊 Ridge vs Lasso vs Linear Regression Accuracy by Alpha")
print(results_df)
# ---------------------------------------------
# 가장 성능이 좋았던 Lasso(alpha=0.01)로 다시 학습
best_lasso = Lasso(alpha=0.01, max_iter=1000000)
best_lasso.fit(X_train_scaled, y_train)

# 다항 특성 이름 가져오기
feature_names = poly.get_feature_names_out(input_features=X.columns)

# 계수 추출 및 절댓값 기준 정렬
coef_series = pd.Series(np.abs(best_lasso.coef_), index=feature_names)
top_features = coef_series.sort_values(ascending=False).head(3)

print("\n🔍 Lasso(alpha=0.01)에서 영향력 큰 상위 3개 특성:")
print(top_features)