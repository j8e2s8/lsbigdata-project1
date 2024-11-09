import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.stats import norm, uniform
from sklearn.preprocessing import StandardScaler, MinMaxScaler, PolynomialFeatures
from sklearn.linear_model import Lasso, Ridge, LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import cross_val_score, KFold, train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.metrics import make_scorer, mean_squared_error
import statsmodels.api as sm

import os


train_df = pd.read_csv('./blueberry/train.csv')
train_df = train_df.drop(columns=('id'))
test_df = pd.read_csv('./blueberry/test.csv')
test_df = test_df.drop(columns= ('id'))
submission = pd.read_csv('./blueberry/sample_submission.csv')

train_x = train_df.drop(columns=('yield'))
train_y = train_df['yield']

test_x = test_df.copy()


# box-cox : 'fruitset', 'fruitmass', 'seeds'
def box_cox_transform(data):
    """
    Apply Box-Cox transformation to the data and find optimal lambda.
    
    Parameters:
    - data: The data to transform (must be positive).
    
    Returns:
    - Transformed data and lambda value used.
    """
    transformed_data, lambda_ = stats.boxcox(data + 1)  # Adding 1 to avoid zero values
    return transformed_data, lambda_


# train
# Box-Cox 변환을 적용할 열 목록
columns_to_transform = ['fruitset', 'fruitmass', 'seeds']


# Box-Cox 변환 및 데이터프레임 업데이트
lambda_dict = {}  # 최적의 lambda 값을 저장할 딕셔너리


for column in columns_to_transform:
    data = train_x[column]
    transformed_data, optimal_lambda = box_cox_transform(data)
    
    # 변환된 데이터를 데이터프레임에 추가
    train_x[column + '_boxcox'] = transformed_data
    
    # 최적의 lambda 값을 저장
    lambda_dict[column] = optimal_lambda

print(f"Optimal lambda values for Box-Cox transformation: {lambda_dict}")


# 표준화 및 정규화
standard_scaler = StandardScaler()
minmax_scaler = MinMaxScaler()

# 표준화 적용
standardized_data = standard_scaler.fit_transform(train_x[[col + '_boxcox' for col in columns_to_transform]])
# 표준화 및 정규화된 데이터프레임 추가
train_x[[col + '_boxcox_standardized' for col in columns_to_transform]] = standardized_data


# 
columns_to_transform = ['clonesize', 'honeybee', 'bumbles', 'andrena', 'osmia','MaxOfUpperTRange', 'MinOfUpperTRange', 'AverageOfUpperTRange','MaxOfLowerTRange', 'MinOfLowerTRange', 'AverageOfLowerTRange',  'RainingDays', 'AverageRainingDays'] 
standardized_data = standard_scaler.fit_transform(train_x[columns_to_transform])
train_x[[col + '_standardized' for col in columns_to_transform]] = standardized_data
train_x.columns


np.where(train_x.columns == 'fruitset_boxcox_standardized')
train_nbs_x = train_x.iloc[:,19:]
train_nbs_x.columns










# test
# Box-Cox 변환을 적용할 열 목록
columns_to_transform = ['fruitset', 'fruitmass', 'seeds']


# Box-Cox 변환 및 데이터프레임 업데이트
lambda_dict = {}  # 최적의 lambda 값을 저장할 딕셔너리


for column in columns_to_transform:
    data = test_x[column]
    transformed_data, optimal_lambda = box_cox_transform(data)
    
    # 변환된 데이터를 데이터프레임에 추가
    test_x[column + '_boxcox'] = transformed_data
    
    # 최적의 lambda 값을 저장
    lambda_dict[column] = optimal_lambda

print(f"Optimal lambda values for Box-Cox transformation: {lambda_dict}")


# 표준화 및 정규화
standard_scaler = StandardScaler()
minmax_scaler = MinMaxScaler()

# 표준화 적용
standardized_data = standard_scaler.fit_transform(test_x[[col + '_boxcox' for col in columns_to_transform]])
# 표준화 및 정규화된 데이터프레임 추가
test_x[[col + '_boxcox_standardized' for col in columns_to_transform]] = standardized_data


# 
columns_to_transform = ['clonesize', 'honeybee', 'bumbles', 'andrena', 'osmia','MaxOfUpperTRange', 'MinOfUpperTRange', 'AverageOfUpperTRange','MaxOfLowerTRange', 'MinOfLowerTRange', 'AverageOfLowerTRange',  'RainingDays', 'AverageRainingDays'] 
standardized_data = standard_scaler.fit_transform(test_x[columns_to_transform])
test_x[[col + '_standardized' for col in columns_to_transform]] = standardized_data
test_x.columns


np.where(test_x.columns == 'fruitset_boxcox_standardized')
test_nbs_x = test_x.iloc[:,19:]
test_nbs_x.columns




# 변수 늘리기
polynomial_transformer=PolynomialFeatures(3)

polynomial_features=polynomial_transformer.fit_transform(train_nbs_x.values)
features=polynomial_transformer.get_feature_names_out(train_nbs_x.columns)
train_bignbs_x=pd.DataFrame(polynomial_features,columns=features)

polynomial_features=polynomial_transformer.fit_transform(test_nbs_x.values)
features=polynomial_transformer.get_feature_names_out(test_nbs_x.columns)
test_bignbs_x=pd.DataFrame(polynomial_features,columns=features)



# -----------------------
# 라쏘 회귀직선
#######alpha
# 교차 검증 설정
kf = KFold(n_splits=10, shuffle=True, random_state=2024)

def rmse(model):
    score = np.sqrt(-cross_val_score(model, train_bignbs_x, train_y, cv = kf,
                                     n_jobs = -1, scoring = "neg_mean_absolute_error").mean())
    return(score)

# 각 알파 값에 대한 교차 검증 점수 저장
alpha_values = np.arange(2, 4, 0.1)
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
model= Lasso(alpha = 3.100000000000001)

# 모델 학습
model.fit(train_bignbs_x, train_y)  # 자동으로 기울기, 절편 값을 구해줌

pred_y=model.predict(test_bignbs_x) # test 셋에 대한 집값
pred_y

# SalePrice 바꿔치기
submission["yield"] = pred_y
submission

# csv 파일로 내보내기
submission.to_csv("./blueberry/partstd_allbig_lasso.csv", index=False)






