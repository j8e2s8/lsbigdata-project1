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
alpha_values = np.arange(2.1, 4.0, 0.1)
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
print("Optimal lambda:", optimal_alpha)

# 결과 시각화
plt.plot(df['lambda'], df['validation_error'], label='Validation Error', color='red')
plt.xlabel('Lambda')
plt.ylabel('Mean Squared Error')
plt.legend()
plt.title('Lasso Regression Train vs Validation Error')
plt.show()


### model
model= Lasso(alpha = 2.4000000000000004)
# 2.9
# 2.4000000000000004   # squared, absolute 일 때



# 모델 학습
model.fit(X, y)  # 자동으로 기울기, 절편 값을 구해줌

pred_y=model.predict(test_X) # test 셋에 대한 집값
pred_y

# SalePrice 바꿔치기
submission["yield"] = pred_y
submission

# csv 파일로 내보내기
submission.to_csv("./blueberry/std_allbig_lasso.csv", index=False)