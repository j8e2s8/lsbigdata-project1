import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.stats import norm, uniform
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.linear_model import Lasso, Ridge, LinearRegression
from sklearn.model_selection import cross_val_score, KFold
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
from sklearn.metrics import make_scorer, mean_squared_error
import statsmodels.api as sm
import os



berry_train = pd.read_csv('./blueberry/train.csv')
berry_test = pd.read_csv('./blueberry/test.csv')
submission = pd.read_csv('./blueberry/sample_submission.csv')



## train
X=berry_train.drop(["yield", "id"], axis=1)
y=berry_train["yield"]
berry_test=berry_test.drop(["id"], axis=1)

# 표준화
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
test_X_scaled=scaler.transform(berry_test)

# 정규화된 데이터를 DataFrame으로 변환
X = pd.DataFrame(X_scaled, columns=X.columns)
test_X= pd.DataFrame(test_X_scaled, columns=berry_test.columns)

polynomial_transformer=PolynomialFeatures(3)

polynomial_features=polynomial_transformer.fit_transform(X.values)
features=polynomial_transformer.get_feature_names_out(X.columns)
X=pd.DataFrame(polynomial_features,columns=features)

polynomial_features=polynomial_transformer.fit_transform(test_X.values)
features=polynomial_transformer.get_feature_names_out(test_X.columns)
test_X=pd.DataFrame(polynomial_features,columns=features)

#######alpha
# 교차 검증 설정
kf = KFold(n_splits=20, shuffle=True, random_state=2024)

def rmse(model):
    score = np.sqrt(-cross_val_score(model, X, y, cv = kf,
                                     n_jobs = -1, scoring = "neg_mean_absolute_error").mean())
    return(score)

# 각 알파 값에 대한 교차 검증 점수 저장
<<<<<<< HEAD
alpha_values = np.arange(2.36, 2.43, 0.01)
=======
alpha_values = np.arange(2.1, 4.0, 0.1)
>>>>>>> f6225cb6866db1dbc484cf29f80d50d2e032083a
mean_scores = np.zeros(len(alpha_values))

k=0
for alpha in alpha_values:
    lasso = Lasso(alpha=alpha)
    mean_scores[k] = rmse(lasso)
    k += 1

# 결과를 DataFrame으로 저장
df = pd.DataFrame({
    'lambda': alpha_values,
    'validation_error': mean_scores
})

# 최적의 alpha 값 찾기
optimal_alpha = df['lambda'][np.argmin(df['validation_error'])]
<<<<<<< HEAD
print("Optimal lambda:", optimal_alpha, "\n해당 람다에서의 validation_error = ", df['validation_error'].min())
=======
print("Optimal lambda:", optimal_alpha)
>>>>>>> f6225cb6866db1dbc484cf29f80d50d2e032083a

# 결과 시각화
plt.plot(df['lambda'], df['validation_error'], label='Validation Error', color='red')
plt.xlabel('Lambda')
plt.ylabel('Mean Squared Error')
plt.legend()
plt.title('Lasso Regression Train vs Validation Error')
plt.show()


### model
<<<<<<< HEAD
model= Lasso(alpha = 2.5499999999999994)
# 2.9
# 2.4000000000000004   # squared, absolute 일 때
# 2.5499999999999994   # 다시 돌렸을 때 나온 람다값  <- 성능 어떤지 알아봐


# 모델 학습
model.fit(X, y)  # 자동으로 기울기, 절편 값을 구해줌

pred_y=model.predict(test_X) # test 셋에 대한 집값
pred_y

# SalePrice 바꿔치기
submission["yield"] = pred_y
submission

# csv 파일로 내보내기
submission.to_csv("./blueberry/std_allbig_lasso2.csv", index=False)




# ------------------------
# 일반 회귀직선도 해보기
model = LinearRegression()
model.fit(X, y)
pred_y=model.predict(test_X) # test 셋에 대한 집값
pred_y

# SalePrice 바꿔치기
submission["yield"] = pred_y
submission

# csv 파일로 내보내기
submission.to_csv("./blueberry/std_allbig_linearregression.csv", index=False)





# --------------------------------

# 릿지 회귀직선도 해보기

# 교차 검증 설정
kf = KFold(n_splits=10, shuffle=True, random_state=2024)

def rmse(model):
    score = np.sqrt(-cross_val_score(model, X, y, cv = kf,
                                     n_jobs = -1, scoring = "neg_mean_absolute_error").mean())
    return(score)

# 각 알파 값에 대한 교차 검증 점수 저장
alpha_values = np.arange(143, 145, 0.1)
mean_scores = np.zeros(len(alpha_values))

k=0
for alpha in alpha_values:
    ridge = Ridge(alpha=alpha)
    mean_scores[k] = rmse(ridge)
    k += 1

# 결과를 DataFrame으로 저장
df = pd.DataFrame({
    'lambda': alpha_values,
    'validation_error': mean_scores
})

# 최적의 alpha 값 찾기
optimal_alpha = df['lambda'][np.argmin(df['validation_error'])]
print("Optimal lambda:", optimal_alpha)

# 결과 시각화
plt.plot(df['lambda'], df['validation_error'], label='Validation Error', color='red')
plt.xlabel('Lambda')
plt.ylabel('Mean Squared Error')
plt.legend()
plt.title('Ridge Regression Train vs Validation Error')
plt.show()


### model
model= Ridge(alpha = 144.29999999999993)

=======
model= Lasso(alpha = 2.4000000000000004)
# 2.9
# 2.4000000000000004   # squared, absolute 일 때
>>>>>>> f6225cb6866db1dbc484cf29f80d50d2e032083a



# 모델 학습
model.fit(X, y)  # 자동으로 기울기, 절편 값을 구해줌

pred_y=model.predict(test_X) # test 셋에 대한 집값
pred_y

# SalePrice 바꿔치기
submission["yield"] = pred_y
submission

# csv 파일로 내보내기
<<<<<<< HEAD
submission.to_csv("./blueberry/std_allbig_ridge.csv", index=False)



# ---------------------------------------
# big ridge가 성능이 안 좋으니까 big 하지말고 표준화만 해보기
# 교차 검증 설정
kf = KFold(n_splits=10, shuffle=True, random_state=2024)

def rmse(model):
    score = np.sqrt(-cross_val_score(model, X, y, cv = kf,
                                     n_jobs = -1, scoring = "neg_mean_absolute_error").mean())
    return(score)

# 각 알파 값에 대한 교차 검증 점수 저장
alpha_values = np.arange(0, 1, 0.1)
mean_scores = np.zeros(len(alpha_values))

k=0
for alpha in alpha_values:
    ridge = Ridge(alpha=alpha)
    mean_scores[k] = rmse(ridge)
    k += 1

# 결과를 DataFrame으로 저장
df = pd.DataFrame({
    'lambda': alpha_values,
    'validation_error': mean_scores
})

# 최적의 alpha 값 찾기
optimal_alpha = df['lambda'][np.argmin(df['validation_error'])]
print("Optimal lambda:", optimal_alpha)


### model
model= Ridge(alpha = 0.30000000000000004)




# 모델 학습
model.fit(X, y)  # 자동으로 기울기, 절편 값을 구해줌

pred_y=model.predict(test_X) # test 셋에 대한 집값
pred_y

# SalePrice 바꿔치기
submission["yield"] = pred_y
submission

# csv 파일로 내보내기
submission.to_csv("./blueberry/std_all_ridge.csv", index=False)



=======
submission.to_csv("./blueberry/std_allbig_lasso.csv", index=False)
>>>>>>> f6225cb6866db1dbc484cf29f80d50d2e032083a
