import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.stats import norm, uniform
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.linear_model import Lasso, Ridge, LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import cross_val_score, KFold
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
from sklearn.metrics import make_scorer, mean_squared_error
import statsmodels.api as sm

import os

# !pip install statsmodels


os.getcwd()
os.chdir('c://Users//USER//Documents//LS 빅데이터 스쿨//lsbigdata-project1')



train_df = pd.read_csv('./blueberry/train.csv')
train_df = train_df.drop(columns=('id'))
test_df = pd.read_csv('./blueberry/test.csv')
test_df = test_df.drop(columns= ('id'))
submission = pd.read_csv('./blueberry/sample_submission.csv')


# 데이터 파악 : 다 수치 변수임
train_df.info()
test_df.info()


# 결측값 없음
nan_train = train_df.isna().sum()
nan_train[nan_train>0]

nan_test = test_df.isna().sum()
nan_test[nan_test>0]


# 데이터 개수
len(train_df)
len(test_df)

train_x = train_df.drop(columns=('yield'))
train_y = train_df['yield']

test_x = test_df.copy()



# 다중공선성
def calculate_vif(df):
    vif_data = pd.DataFrame()
    vif_data["Variable"] = df.columns
    vif_data["VIF"] = [calculate_single_vif(df, i) for i in range(df.shape[1])]
    return vif_data

def calculate_single_vif(df, i):
    # 독립 변수 X와 종속 변수 y 설정
    X = df.drop(df.columns[i], axis=1)
    y = df.iloc[:, i]
    
    # 상수항 추가
    X = sm.add_constant(X)
    
    # 회귀 모델 적합
    model = sm.OLS(y, X).fit()
    
    # VIF 계산
    return 1 / (1 - model.rsquared)



# VIF 계산
vif_df = calculate_vif(train_x)
print(vif_df)

train_x2 = train_x[['clonesize', 'honeybee', 'bumbles', 'andrena', 'osmia', 'AverageOfUpperTRange', 'AverageOfLowerTRange', 'AverageRainingDays','fruitmass' ]]
train_df.columns


# 이상치 찾기
train_out = train_df[['clonesize', 'honeybee', 'bumbles', 'andrena', 'osmia', 'AverageOfUpperTRange', 'AverageRainingDays','fruitmass','yield' ]]

train_out = train_out[(train_out['honeybee'] <= 1)&(train_out['osmia'] <= 1) &(train_out['bumbles'] <= 1)&(train_out['andrena'] <= 1)]
train_out[train_out['bumbles']>1]

train_out_x = train_out.drop(columns =('yield'))
train_out_y = train_out['yield']

train_out_x.columns

# VIF 계산
vif_df = calculate_vif(train_out_x)
print(vif_df)


# 분포 확인
def hist(df, numeric_col):
	plt.clf()
	plt.rcPramas['font.family'] = 'Malgun Gothic'
	plt.rcPramas['axes.unicode_minus'] = False
	a = numeric_col + "의 분포"
	plt.title(a)
	sns.histplot(df[numeric_col], stat='density')
	plt.tight_layout()
	plt.show()

train_df.columns
<<<<<<< HEAD
len(train_df.columns)
hist(train_df, train_df.columns[1])

sns.histplot(data = train_df, x=train_df.columns[0], stat='density') # 'clonesize'
sns.histplot(data = train_df, x=train_df.columns[1], stat='density') # 'honeybee'
sns.histplot(data = train_df, x=train_df.columns[2], stat='density') # 'bumbles'
sns.histplot(data = train_df, x=train_df.columns[3], stat='density') # 'andrena'
sns.histplot(data = train_df, x=train_df.columns[4], stat='density') # 'osmia'
sns.histplot(data = train_df, x=train_df.columns[5], stat='density') # 'MaxOfUpperTRange'
sns.histplot(data = train_df, x=train_df.columns[6], stat='density') # 'MinOfUpperTRange'
sns.histplot(data = train_df, x=train_df.columns[7], stat='density') # 'AverageOfUpperTRange'
sns.histplot(data = train_df, x=train_df.columns[8], stat='density') # 'MaxOfLowerTRange'
sns.histplot(data = train_df, x=train_df.columns[9], stat='density') # 'MinOfLowerTRange'
sns.histplot(data = train_df, x=train_df.columns[10], stat='density') # 'AverageOfLowerTRange'
sns.histplot(data = train_df, x=train_df.columns[11], stat='density') # 'RainingDays'
sns.histplot(data = train_df, x=train_df.columns[12], stat='density') # 'AverageRainingDays'
sns.histplot(data = train_df, x=train_df.columns[13], stat='density') # 'fruitset'
sns.histplot(data = train_df, x=train_df.columns[14], stat='density') # 'fruitmass'
sns.histplot(data = train_df, x=train_df.columns[15], stat='density') # 'seeds'
sns.histplot(data = train_df, x=train_df.columns[16], stat='density') # 'yield'
=======
hist(train_df, train_df.columns[1])

sns.histplot(data = train_df, x=train_df.columns[1], stat='density') # 'clonesize'
sns.histplot(data = train_df, x=train_df.columns[2], stat='density') # 'honeybee'
sns.histplot(data = train_df, x=train_df.columns[3], stat='density') # 'bumbles'
sns.histplot(data = train_df, x=train_df.columns[4], stat='density') # 'andrena'
sns.histplot(data = train_df, x=train_df.columns[5], stat='density') # 'osmia'
sns.histplot(data = train_df, x=train_df.columns[6], stat='density') # 'MaxOfUpperTRange'
sns.histplot(data = train_df, x=train_df.columns[7], stat='density') # 'MinOfUpperTRange'
sns.histplot(data = train_df, x=train_df.columns[8], stat='density') # 'AverageOfUpperTRange'
sns.histplot(data = train_df, x=train_df.columns[9], stat='density') # 'MaxOfLowerTRange'
sns.histplot(data = train_df, x=train_df.columns[10], stat='density') # 'MinOfLowerTRange'
sns.histplot(data = train_df, x=train_df.columns[11], stat='density') # 'AverageOfLowerTRange'
sns.histplot(data = train_df, x=train_df.columns[12], stat='density') # 'RainingDays'
sns.histplot(data = train_df, x=train_df.columns[13], stat='density') # 'AverageRainingDays'
sns.histplot(data = train_df, x=train_df.columns[14], stat='density') # 'fruitset'
sns.histplot(data = train_df, x=train_df.columns[15], stat='density') # 'fruitmass'
sns.histplot(data = train_df, x=train_df.columns[16], stat='density') # 'seeds'
sns.histplot(data = train_df, x=train_df.columns[17], stat='density') # 'yield'
>>>>>>> f6225cb6866db1dbc484cf29f80d50d2e032083a

train_df['honeybee'].max()
sns.histplot(data = train_df, x='honeybee', stat='density')




# -----------------------------
# box cox
y, lambda_optimal = stats.boxcox(train_df['fruitset'])  # 최적의 람다 찾기



def box_cox_transform(data, lambda_=None):
    """
    Apply Box-Cox transformation to the data.
    
    Parameters:
    - data: The data to transform (must be positive).
    - lambda_: The lambda value for the Box-Cox transformation. If None, it will be estimated.
    
    Returns:
    - Transformed data and lambda value used.
    """
    if lambda_ is None:
        # Find the optimal lambda value
        transformed_data, lambda_ = stats.boxcox(data)
    else:
        # Apply Box-Cox transformation with a specific lambda value
        transformed_data = stats.boxcox(data, lmbda=lambda_)
    
    return transformed_data, lambda_



train_x.columns
column_to_transform = ['fruitset','fruitmass']
data = train_df[column_to_transform]
# Box-Cox 변환을 적용하고 최적의 lambda 값 찾기
transformed_data, optimal_lambda = box_cox_transform(data + 1)  # 0 또는 음수가 포함될 수 있으므로 1을 추가하여 안정성 확보

print(f"Optimal lambda for Box-Cox transformation: {optimal_lambda}")

# 결과 데이터프레임에 변환된 데이터 추가
train_df[column_to_transform + '_boxcox'] = transformed_data








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

# 예제 데이터프레임 (여기에 실제 데이터프레임을 사용해야 합니다)
df = pd.DataFrame({
    'feature1': [10, 20, 30, 40, 50, 60, 70, 80, 90, 100],
    'feature2': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
    'feature3': [5, 15, 25, 35, 45, 55, 65, 75, 85, 95]
})

# Box-Cox 변환을 적용할 열 목록
columns_to_transform = ['feature1', 'feature2', 'feature3']

# Box-Cox 변환 및 데이터프레임 업데이트
lambda_dict = {}  # 최적의 lambda 값을 저장할 딕셔너리

for column in columns_to_transform:
    data = df[column]
    transformed_data, optimal_lambda = box_cox_transform(data)
    
    # 변환된 데이터를 데이터프레임에 추가
    df[column + '_boxcox'] = transformed_data
    
    # 최적의 lambda 값을 저장
    lambda_dict[column] = optimal_lambda

print(f"Optimal lambda values for Box-Cox transformation: {lambda_dict}")

# 표준화 및 정규화
standard_scaler = StandardScaler()
minmax_scaler = MinMaxScaler()

# 표준화 적용
standardized_data = standard_scaler.fit_transform(df[[col + '_boxcox' for col in columns_to_transform]])

# 정규화 적용
normalized_data = minmax_scaler.fit_transform(df[[col + '_boxcox' for col in columns_to_transform]])

# 표준화 및 정규화된 데이터프레임 추가
df[[col + '_boxcox_standardized' for col in columns_to_transform]] = standardized_data
df[[col + '_boxcox_normalized' for col in columns_to_transform]] = normalized_data





# ---------------------
# 정규화


# MinMaxScaler 객체 생성 (기본 범위는 0~1)
scaler = MinMaxScaler()

# 데이터 정규화
mmscaled_train_x = scaler.fit_transform(train_x)  # x에 대해서만
mmscaled_test_x = scaler.transform(test_x)

# 정규화된 데이터프레임 생성
mmscaled_train_x = pd.DataFrame(mmscaled_train_x, columns=train_x.columns)
mmscaled_test_x = pd.DataFrame(mmscaled_test_x, columns = test_x.columns)






# ---------------------------

# 표준화
scaler = StandardScaler()
scaled_train_x = scaler.fit_transform(train_x)  # train_x변수들로만 표준화 수행 -> 데이터프레임 아님
scaled_test_x = scaler.transform(test_x) # test_x변수들로만 동일한 스케일러로 표준화 수행 -> 데이터프레임 아님
scaled_train_x.shape
train_df.shape

# 정규화된 데이터를 DataFrame으로 변환
scaled_train_x = pd.DataFrame(scaled_train_x, columns = train_x.columns)
scaled_test_x = pd.DataFrame(scaled_test_x, columns = test_x.columns)


# 이상치
# 이상치 판단 기준 설정: 일반적으로 표준적인 z-score 기준을 사용하여 이상치를 판단합니다. 보통 z-score가 특정 기준값(일반적으로 2 ~ 3)을 넘어가는 데이터를 이상치로 간주합니다.
def outliers_z_score(ys):
    threshold = 3

    mean_y = np.mean(ys)
    stdev_y = np.std(ys)
    z_scores = [(y - mean_y) / stdev_y for y in ys]
    return np.where(np.abs(z_scores) > threshold)[0].tolist()


train_x.columns
sns.histplot(data = scaled_train_x, x='clonesize', stat='density')
sns.histplot(data = train_x, x='clonesize', stat='density')
train_df.iloc[outliers_z_score(train_df['clonesize']),:]  # 'clonesize'에서의 이상치 

sns.histplot(data = scaled_train_x, x='honeybee', stat='density')
sns.histplot(data = train_x, x='honeybee', stat='density')
train_df.iloc[outliers_z_score(train_df['honeybee']),:]  # 'honeybee'에서의 이상치 <- 좁은 범위에서 많은 데이터가 이상치라서 제거하면 너무 많은 정보 손실이 될듯

sns.histplot(data = scaled_train_x, x='bumbles', stat='density')
sns.histplot(data = train_x, x='bumbles', stat='density')
train_df.iloc[outliers_z_score(train_df['bumbles']),:]  # 'bumbles'에서의 이상치 <- 좁은 범위에서 많은 데이터가 이상치라서 제거하면 너무 많은 정보 손실이 될듯


sns.histplot(data = scaled_train_x, x='andrena', stat='density')
sns.histplot(data = train_x, x='andrena', stat='density')
train_df.iloc[outliers_z_score(train_df['andrena']),:]  # 'andrena'에서의 이상치

sns.histplot(data = scaled_train_x, x='osmia', stat='density')
sns.histplot(data = train_x, x='osmia', stat='density')
train_df.iloc[outliers_z_score(train_df['osmia']),:]  # 'osmia'에서의 이상치  <- 좁은 범위에서 많은 데이터가 이상치라서 제거하면 너무 많은 정보 손실이 될듯


sns.histplot(data = scaled_train_x, x='MaxOfUpperTRange', stat='density')
sns.histplot(data = train_x, x='MaxOfUpperTRange', stat='density')
train_df.iloc[outliers_z_score(train_df['MaxOfUpperTRange']),:]  # 'MaxOfUpperTRange' 없음

sns.histplot(data = scaled_train_x, x='MinOfUpperTRange', stat='density')
sns.histplot(data = train_x, x='MinOfUpperTRange', stat='density')
train_df.iloc[outliers_z_score(train_df['MinOfUpperTRange']),:]  # 'MinOfUpperTRange' 없음


sns.histplot(data = scaled_train_x, x='AverageOfUpperTRange', stat='density')
sns.histplot(data = train_x, x='AverageOfUpperTRange', stat='density')
train_df.iloc[outliers_z_score(train_df['AverageOfUpperTRange']),:]  # 'AverageOfUpperTRange' 없음

sns.histplot(data = scaled_train_x, x='MaxOfLowerTRange', stat='density')
sns.histplot(data = train_x, x='MaxOfLowerTRange', stat='density')
train_df.iloc[outliers_z_score(train_df['MaxOfLowerTRange']),:]  # 'MaxOfLowerTRange' 없음

sns.histplot(data = scaled_train_x, x='MinOfLowerTRange', stat='density')
sns.histplot(data = train_x, x='MinOfLowerTRange', stat='density')
train_df.iloc[outliers_z_score(train_df['MinOfLowerTRange']),:]  # 'MinOfLowerTRange' 없음

sns.histplot(data = scaled_train_x, x='AverageOfLowerTRange', stat='density')
sns.histplot(data = train_x, x='AverageOfLowerTRange', stat='density')
train_df.iloc[outliers_z_score(train_df['AverageOfLowerTRange']),:]  # 'AverageOfLowerTRange' 없음

sns.histplot(data = scaled_train_x, x='RainingDays', stat='density')
sns.histplot(data = train_x, x='RainingDays', stat='density')
train_df.iloc[outliers_z_score(train_df['RainingDays']),:]  # 'RainingDays' 없음

sns.histplot(data = scaled_train_x, x='AverageRainingDays', stat='density')
sns.histplot(data = train_x, x='AverageRainingDays', stat='density')
train_df.iloc[outliers_z_score(train_df['AverageRainingDays']),:]  # 'AverageRainingDays' 없음

sns.histplot(data = scaled_train_x, x='fruitset', stat='density')
sns.histplot(data = train_x, x='fruitset', stat='density')
train_df.iloc[outliers_z_score(train_df['fruitset']),:]  # 'fruitset' 없음



sns.histplot(data= train_df, x = 'fruitset' , stat = 'density')
sns.histplot(data= train_df, x = 'fruitset_boxcox' , stat = 'density')


# -------------------------
# 총
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
columns_to_transform = train_x.columns


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




# test
# Box-Cox 변환을 적용할 열 목록
columns_to_transform = test_x.columns


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


train_bs_x = train_x.iloc[:,32:]
test_bs_x = test_x.iloc[:,32:]








# 정규화 적용 (아직 안함)
normalized_data = minmax_scaler.fit_transform(train_x[[col + '_boxcox' for col in columns_to_transform]])

df[[col + '_boxcox_normalized' for col in columns_to_transform]] = normalized_data



np.where(train_x.columns == 'clonesize_boxcox_standardized')
col = train_x.columns[33:]

sns.histplot(data=train_x, x=col[0] , stat='density')  # clonesize_boxcox_standardized : 이상치 있어보임  => 없음
sns.histplot(data=train_x, x=col[1] , stat='density')  # 'honeybee_boxcox_standardized' : 이상치 있어보임
sns.histplot(data=train_x, x=col[2] , stat='density')  # 'bumbles_boxcox_standardized' : 이상치 있어보임
sns.histplot(data=train_x, x=col[3] , stat='density')  # 'andrena_boxcox_standardized' : 이상치 있어보임
sns.histplot(data=train_x, x=col[4] , stat='density')  # 'osmia_boxcox_standardized' : 이상치 있어보임
sns.histplot(data=train_x, x=col[5] , stat='density')  # 'MaxOfUpperTRange_boxcox_standardized'
sns.histplot(data=train_x, x=col[6] , stat='density')  # 'MinOfUpperTRange_boxcox_standardized' : 이상치 있어보임
sns.histplot(data=train_x, x=col[7] , stat='density')  # 'AverageOfUpperTRange_boxcox_standardized'
sns.histplot(data=train_x, x=col[8] , stat='density')  # 'MaxOfLowerTRange_boxcox_standardized'
sns.histplot(data=train_x, x=col[9] , stat='density')  # 'MinOfLowerTRange_boxcox_standardized'
sns.histplot(data=train_x, x=col[10] , stat='density')  # 'AverageOfLowerTRange_boxcox_standardized'
sns.histplot(data=train_x, x=col[11] , stat='density')  # 'RainingDays_boxcox_standardized'
sns.histplot(data=train_x, x=col[12] , stat='density')  # 'AverageRainingDays_boxcox_standardized'
sns.histplot(data=train_x, x=col[13] , stat='density')  # 'fruitset_boxcox_standardized' : 이상치 있어보임.
sns.histplot(data=train_x, x=col[14] , stat='density')  # 'fruitmass_boxcox_standardized' : 이상치 있어보임.
sns.histplot(data=train_x, x=col[15] , stat='density')  # 'seeds_boxcox_standardized' : 이상치 있어보임.


train_bs_x = train_x.iloc[:,33:]
train_bs_y = train_y.copy()

def outliers_z_score(ys):
    threshold = 3

    mean_y = np.mean(ys)
    stdev_y = np.std(ys)
    z_scores = [(y - mean_y) / stdev_y for y in ys]
    return np.where(np.abs(z_scores) > threshold)[0].tolist()


outliers_z_score(train_x['clonesize_boxcox_standardized'])
outliers_z_score(train_x['honeybee_boxcox_standardized'])
train_x.iloc[outliers_z_score(train_x['honeybee_boxcox_standzardized']),:]



# 변수 늘리기
polynomial_transformer=PolynomialFeatures(3)

polynomial_features=polynomial_transformer.fit_transform(train_bs_x.values)
features=polynomial_transformer.get_feature_names_out(train_bs_x.columns)
train_bigbs_x=pd.DataFrame(polynomial_features,columns=features)

polynomial_features=polynomial_transformer.fit_transform(test_bs_x.values)
features=polynomial_transformer.get_feature_names_out(test_bs_x.columns)
test_bigbs_x=pd.DataFrame(polynomial_features,columns=features)




X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)




# fruitmass 0.35 미만 yield 5000 이상
train_out = train_out[(train_out['fruitmass']>=0.35) & (train_out['yield']<5000)]

train_out.columns

train_out_x = train_out.drop(columns =('yield'))
train_out_y = train_out['yield']


# 일반회귀 모든 변수
model = LinearRegression()
model.fit(train_x, train_y)
train_y_pred = model.predict(train_x)

print("train MAE : ", np.abs(train_y_pred - train_y).mean() )

test_y_pred = model.predict(test_x)

submission['yield'] = test_y_pred
submission.to_csv('./blueberry/all_linearregression.csv', index=False)


# 일반회귀 - 다중공선성 제거 변수
model = LinearRegression()

train_perf_x2 = train_x2[:10193]
train_perf_y = train_y[:10193]
valid_perf_x2 = train_x2[10193:]
valid_perf_y = train_y[10193:]
model.fit(train_perf_x2, train_perf_y)
train_pred = model.predict(train_perf_x2)
valid_pred = model.predict(valid_perf_x2)

print("train MAE : ", np.abs(train_pred - train_perf_y).mean() , "\n valid MAE : ", np.abs(valid_pred - valid_perf_y).mean())

model.fit(train_x2, train_y)
train_y_pred = model.predict(train_x)

test_y_pred = model.predict(test_x)

# submission['yield'] = test_y_pred
# submission.to_csv('./blueberry/all_linearregression.csv', index=False)



# 일반회귀 - 이상치 제거 + 다중공선성 제거 변수
model = LinearRegression()

train_perf_x2 = train_out_x[:10193]
train_perf_y = train_out_y[:10193]
valid_perf_x2 = train_out_x[10193:]
valid_perf_y = train_out_y[10193:]
model.fit(train_perf_x2, train_perf_y)
train_pred = model.predict(train_perf_x2)
valid_pred = model.predict(valid_perf_x2)

print("train MAE : ", np.abs(train_pred - train_perf_y).mean() , "\n valid MAE : ", np.abs(valid_pred - valid_perf_y).mean())

model.fit(train_out_x, train_out_y)

test_y_pred = model.predict(test_x)

submission['yield'] = test_y_pred
submission.to_csv('./blueberry/out_VIF2_linearregression.csv', index=False)



# 일반회귀 - 표준화  valid MSE : 376.5 나오는데 score : 362.49 나옴
cut_train_x, cut_valid_x, cut_train_y, cut_valid_y = train_test_split(scaled_train_x, train_y, test_size=0.3, random_state=20240828)

model = LinearRegression()
model.fit(cut_train_x, cut_train_y)
cut_train_pred = model.predict(cut_train_x)
cut_valid_pred = model.predict(cut_valid_x)

print("train MAE : ", np.abs(cut_train_pred - cut_train_y).mean(), "\n valid MAE : ", np.abs(cut_valid_pred - cut_valid_y).mean() )


model = LinearRegression()
model.fit(scaled_train_x, train_y)

test_y_pred = model.predict(scaled_test_x)

submission['yield'] = test_y_pred
submission.to_csv('./blueberry/scaled_all_linearregression.csv', index=False)



# ----------------------------------------------


# lasso 회귀분석 모든 변수
kf = KFold(n_splits=30, shuffle=True, random_state=2024)

def rmse(model):
    score = np.sqrt(-cross_val_score(model, train_x, train_y, cv = kf,
                                     n_jobs=-1, scoring= "neg_mean_absolute_error").mean())
    return(score)                   # n_jobs=-1 : 계산 가능한 공간이 4개가 있는데 각 공간마다 fold1이 valid일 때의 계산, fold2가 valid일 때의 계산, ... 을 맡겨서 계산을 더 빠르게 하는 것임


# 각 알파 값에 대한 교차 검증 점수 저장
alpha_values = np.arange(0, 0.1, 0.01)
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

df.head()


# 최적의 alpha 값 찾기
optimal_alpha = df['lambda'][np.argmin(df['validation_error'])]
print("Optimal lambda:", optimal_alpha)



model = Lasso(alpha=0)   # 람다가 0이면 일반 회귀직선과 다를 게 없음

len(train_x)

train_perf_x = train_x[:10193]
train_perf_y = train_y[:10193]
valid_perf_x = train_x[10193:]
valid_perf_y = train_y[10193:]
model.fit(train_perf_x, train_perf_y)
train_pred = model.predict(train_perf_x)
valid_pred = model.predict(valid_perf_x)

print("train MAE : ", np.abs(train_pred - train_perf_y).mean() , "\n valid MAE : ", np.abs(valid_pred - valid_perf_y).mean())

model.fit(train_x, train_y)
train_y_pred = model.predict(train_x)

test_y_pred = model.predict(test_x)

# submission['yield'] = test_y_pred
# submission.to_csv('./blueberry/all_linearregression.csv', index=False)




# lasso 회귀분석 - 다중공선성 제거 변수
kf = KFold(n_splits=30, shuffle=True, random_state=2024)

def rmse(model):
    score = np.sqrt(-cross_val_score(model, train_x2, train_y, cv = kf,
                                     n_jobs=-1, scoring= "neg_mean_absolute_error").mean())
    return(score)                   # n_jobs=-1 : 계산 가능한 공간이 4개가 있는데 각 공간마다 fold1이 valid일 때의 계산, fold2가 valid일 때의 계산, ... 을 맡겨서 계산을 더 빠르게 하는 것임


# 각 알파 값에 대한 교차 검증 점수 저장
alpha_values = np.arange(0, 0.1, 0.01)
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

df.head()


# 최적의 alpha 값 찾기
optimal_alpha = df['lambda'][np.argmin(df['validation_error'])]
print("Optimal lambda:", optimal_alpha)



model = Lasso(alpha=0)   # 람다가 0이면 일반 회귀직선과 다를 게 없음

len(train_x)

train_perf_x = train_x[:10193]
train_perf_y = train_y[:10193]
valid_perf_x = train_x[10193:]
valid_perf_y = train_y[10193:]
model.fit(train_perf_x, train_perf_y)
train_pred = model.predict(train_perf_x)
valid_pred = model.predict(valid_perf_x)

print("train MAE : ", np.abs(train_pred - train_perf_y).mean() , "\n valid MAE : ", np.abs(valid_pred - valid_perf_y).mean())

model.fit(train_x, train_y)
train_y_pred = model.predict(train_x)

test_y_pred = model.predict(test_x)

# submission['yield'] = test_y_pred
# submission.to_csv('./blueberry/all_linearregression.csv', index=False)




# lasso 회귀분석 - 이상치 제거 + 다중공선성 제거 변수
kf = KFold(n_splits=30, shuffle=True, random_state=2024)

def rmse(model):
    score = np.sqrt(-cross_val_score(model, train_out_x, train_out_y, cv = kf,
                                     n_jobs=-1, scoring= "neg_mean_absolute_error").mean())
    return(score)                   # n_jobs=-1 : 계산 가능한 공간이 4개가 있는데 각 공간마다 fold1이 valid일 때의 계산, fold2가 valid일 때의 계산, ... 을 맡겨서 계산을 더 빠르게 하는 것임


# 각 알파 값에 대한 교차 검증 점수 저장
alpha_values = np.arange(0, 10, 0.1)
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

df.head()


# 최적의 alpha 값 찾기
optimal_alpha = df['lambda'][np.argmin(df['validation_error'])]
print("Optimal lambda:", optimal_alpha)



model = Lasso(alpha=0.03)   # 람다가 0이면 일반 회귀직선과 다를 게 없음

len(train_x)

train_perf_x = train_out_x[:10193]
train_perf_y = train_out_y[:10193]
valid_perf_x = train_out_x[10193:]
valid_perf_y = train_out_y[10193:]
model.fit(train_perf_x, train_perf_y)
train_pred = model.predict(train_perf_x)
valid_pred = model.predict(valid_perf_x)

print("train MAE : ", np.abs(train_pred - train_perf_y).mean() , "\n valid MAE : ", np.abs(valid_pred - valid_perf_y).mean())

model.fit(train_out_x, train_out_y)

test_y_pred = model.predict(test_x)

submission['yield'] = test_y_pred
submission.to_csv('./blueberry/out_VIF_lasso.csv', index=False)




# lasso 회귀분석 - boxcox + 표준화 모든 변수
#train MAE :  385.70143475392314 
#valid MAE :  382.35399386988104
# test MAE : 378.39
kf = KFold(n_splits=10, shuffle=True, random_state=2024)

def rmse(model):
    score = np.sqrt(-cross_val_score(model, train_bigbs_x, train_bs_y, cv = kf,
                                     n_jobs=-1, scoring= "neg_mean_absolute_error").mean())
    return(score)                   # n_jobs=-1 : 계산 가능한 공간이 4개가 있는데 각 공간마다 fold1이 valid일 때의 계산, fold2가 valid일 때의 계산, ... 을 맡겨서 계산을 더 빠르게 하는 것임


# 각 알파 값에 대한 교차 검증 점수 저장
alpha_values = np.arange(3.0, 3.2, 0.01)  # 3.1 나옴
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

df.head()


# 최적의 alpha 값 찾기
optimal_alpha = df['lambda'][np.argmin(df['validation_error'])]
print("Optimal lambda:", optimal_alpha)

df[df['lambda']==3.1299999999999972]  # 19.095814 validation_error

model = Lasso(alpha=3.1299999999999972)   # 람다가 0이면 일반 회귀직선과 다를 게 없음

model.fit(train_bigbs_x, train_bs_y)

test_y_pred = model.predict(test_bigbs_x)

submission['yield'] = test_y_pred
submission.to_csv('./blueberry/box_std_allbig_lasso.csv', index=False)

lasso_pred = test_y_pred











# -------------------------------


# ridge 회귀분석 모든 변수
<<<<<<< HEAD
kf = KFold(n_splits=10, shuffle=True, random_state=2024)
=======
kf = KFold(n_splits=30, shuffle=True, random_state=2024)
>>>>>>> f6225cb6866db1dbc484cf29f80d50d2e032083a

def rmse(model):
    score = np.sqrt(-cross_val_score(model, train_x, train_y, cv = kf,
                                     n_jobs=-1, scoring= "neg_mean_absolute_error").mean())
    return(score)                   # n_jobs=-1 : 계산 가능한 공간이 4개가 있는데 각 공간마다 fold1이 valid일 때의 계산, fold2가 valid일 때의 계산, ... 을 맡겨서 계산을 더 빠르게 하는 것임


# 각 알파 값에 대한 교차 검증 점수 저장
<<<<<<< HEAD
alpha_values = np.arange(0.07, 0.09, 0.001)
=======
alpha_values = np.arange(0, 10, 0.1)
>>>>>>> f6225cb6866db1dbc484cf29f80d50d2e032083a
mean_scores = np.zeros(len(alpha_values))

k=0
for alpha in alpha_values:
<<<<<<< HEAD
    ridge = Ridge(alpha=alpha)
    mean_scores[k] = rmse(ridge)
=======
    lasso = Lasso(alpha=alpha)
    mean_scores[k] = rmse(lasso)
>>>>>>> f6225cb6866db1dbc484cf29f80d50d2e032083a
    k += 1


# 결과를 DataFrame으로 저장
df = pd.DataFrame({
    'lambda': alpha_values,
    'validation_error': mean_scores
})

df.head()


# 최적의 alpha 값 찾기
optimal_alpha = df['lambda'][np.argmin(df['validation_error'])]
print("Optimal lambda:", optimal_alpha)



<<<<<<< HEAD
model = Ridge(alpha=0.07800000000000001)  
=======
model = Ridge(alpha=0)  # 얘도 람다 0 나옴
>>>>>>> f6225cb6866db1dbc484cf29f80d50d2e032083a

train_perf_x = train_x[:10193]
train_perf_y = train_y[:10193]
valid_perf_x = train_x[10193:]
valid_perf_y = train_y[10193:]
model.fit(train_perf_x, train_perf_y)
train_pred = model.predict(train_perf_x)
valid_pred = model.predict(valid_perf_x)

print("train MAE : ", np.abs(train_pred - train_perf_y).mean() , "\n valid MAE : ", np.abs(valid_pred - valid_perf_y).mean())

model.fit(train_x, train_y)
train_y_pred = model.predict(train_x)

test_y_pred = model.predict(test_x)

<<<<<<< HEAD
submission['yield'] = test_y_pred
submission.to_csv('./blueberry/all_ridge.csv', index=False)
=======
#submission['yield'] = test_y_pred
#submission.to_csv('./blueberry/all_ridge.csv', index=False)
>>>>>>> f6225cb6866db1dbc484cf29f80d50d2e032083a



# ridge 회귀분석 - 이상치 제거, 다중공선성 제거 변수
kf = KFold(n_splits=30, shuffle=True, random_state=2024)

def rmse(model):
    score = np.sqrt(-cross_val_score(model, train_out_x, train_out_y, cv = kf,
                                     n_jobs=-1, scoring= "neg_mean_absolute_error").mean())
    return(score)                   # n_jobs=-1 : 계산 가능한 공간이 4개가 있는데 각 공간마다 fold1이 valid일 때의 계산, fold2가 valid일 때의 계산, ... 을 맡겨서 계산을 더 빠르게 하는 것임


# 각 알파 값에 대한 교차 검증 점수 저장
alpha_values = np.arange(0, 1, 1)
mean_scores = np.zeros(len(alpha_values))

k=0
for alpha in alpha_values:
<<<<<<< HEAD
    ridge = Ridge(alpha=alpha)
    mean_scores[k] = rmse(ridge)
=======
    lasso = Lasso(alpha=alpha)
    mean_scores[k] = rmse(lasso)
>>>>>>> f6225cb6866db1dbc484cf29f80d50d2e032083a
    k += 1


# 결과를 DataFrame으로 저장
df = pd.DataFrame({
    'lambda': alpha_values,
    'validation_error': mean_scores
})

df.head()


# 최적의 alpha 값 찾기
optimal_alpha = df['lambda'][np.argmin(df['validation_error'])]
print("Optimal lambda:", optimal_alpha)



model = Ridge(alpha=0)  # 얘도 람다 0 나옴

train_perf_x = train_x[:10193]
train_perf_y = train_y[:10193]
valid_perf_x = train_x[10193:]
valid_perf_y = train_y[10193:]
model.fit(train_perf_x, train_perf_y)
train_pred = model.predict(train_perf_x)
valid_pred = model.predict(valid_perf_x)

print("train MAE : ", np.abs(train_pred - train_perf_y).mean() , "\n valid MAE : ", np.abs(valid_pred - valid_perf_y).mean())

model.fit(train_x, train_y)
train_y_pred = model.predict(train_x)

test_y_pred = model.predict(test_x)

#submission['yield'] = test_y_pred
#submission.to_csv('./blueberry/all_ridge.csv', index=False)






# ridge 회귀분석 - boxcox, 표준화, 모든 변수
#train MAE :  385.3361866767742 
# valid MAE :  382.707335321868

kf = KFold(n_splits=30, shuffle=True, random_state=2024)

def rmse(model):
    score = np.sqrt(-cross_val_score(model, train_bs_x, train_bs_y, cv = kf,
                                     n_jobs=-1, scoring= "neg_mean_absolute_error").mean())
    return(score)                   # n_jobs=-1 : 계산 가능한 공간이 4개가 있는데 각 공간마다 fold1이 valid일 때의 계산, fold2가 valid일 때의 계산, ... 을 맡겨서 계산을 더 빠르게 하는 것임


# 각 알파 값에 대한 교차 검증 점수 저장
alpha_values = np.arange(0.04, 0.06, 0.001)
mean_scores = np.zeros(len(alpha_values))

k=0
for alpha in alpha_values:
<<<<<<< HEAD
    ridge = Ridge(alpha=alpha)
    mean_scores[k] = rmse(ridge)
=======
    lasso = Lasso(alpha=alpha)
    mean_scores[k] = rmse(lasso)
>>>>>>> f6225cb6866db1dbc484cf29f80d50d2e032083a
    k += 1


# 결과를 DataFrame으로 저장
df = pd.DataFrame({
    'lambda': alpha_values,
    'validation_error': mean_scores
})

df.head()


# 최적의 alpha 값 찾기
optimal_alpha = df['lambda'][np.argmin(df['validation_error'])]
print("Optimal lambda:", optimal_alpha)



model = Ridge(alpha=0.056000000000000015)  # 얘도 람다 0 나옴

train_perf_x = train_bs_x[:10193]
train_perf_y = train_bs_y[:10193]
valid_perf_x = train_bs_x[10193:]
valid_perf_y = train_bs_y[10193:]
model.fit(train_perf_x, train_perf_y)
train_pred = model.predict(train_perf_x)
valid_pred = model.predict(valid_perf_x)

print("train MAE : ", np.abs(train_pred - train_perf_y).mean() , "\n valid MAE : ", np.abs(valid_pred - valid_perf_y).mean())

model.fit(train_bs_x, train_bs_y)

test_y_pred = model.predict(test_bs_x)

submission['yield'] = test_y_pred
submission.to_csv('./blueberry/box_std_all_ridge.csv', index=False)

ridge_pred = test_y_pred


# -------------------------
# knn
knn_regressor = KNeighborsRegressor(n_neighbors=3)
knn_regressor.fit(train_bs_x, train_bs_x)
