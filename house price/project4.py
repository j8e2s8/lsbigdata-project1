# 라쏘 회귀분석 하이퍼파리미터 람다 찾기

import os
os.getcwd()
os.chdir('c://Users//USER//Documents//LS 빅데이터 스쿨//lsbigdata-project1/house price')


# 데이터 불러오기
import pandas as pd
import numpy as np
from sklearn.linear_model import Lasso, Ridge

train_df = pd.read_csv('train.csv')
test_df = pd.read_csv('test.csv')
submission = pd.read_csv('sample_submission.csv')







train_df.columns

train = train_df[['OverallCond', 'GrLivArea','SalePrice']]
train.isna().sum()


np.random.seed(20240826)
valid_index = np.random.choice(np.arange(len(train)), size= int(len(train)/4) , replace=False)

valid = train.loc[valid_index,]
train = train.drop(valid_index)

valid_x = valid[['OverallCond', 'GrLivArea']]
valid_y = valid['SalePrice']
train_x = train[['OverallCond', 'GrLivArea']]
train_y = train['SalePrice']






# 람다 값 결정하기
tr_result = []
val_result = []
lambda_result = []


for i in np.arange(1,100):
    model = Lasso(alpha = i*(0.1))
    model.fit(train_x, train_y)   # 임의의 람다로 모델 피팅

    # 모델 성능
    y_hat_train = model.predict(train_x)   # 모델 train set 예측값 구하기
    y_hat_val = model.predict(valid_x)    # 모델 valid set 예측값 구하기

    perf_train = sum((train_y - y_hat_train)**2)
    perf_val = sum((valid_y - y_hat_val)**2)
    tr_result.append(perf_train) 
    val_result.append(perf_val) 
    lambda_result.append(i*(0.1))


lasso_perf_df = pd.DataFrame({
    'train_perf' : tr_result,
    'valid_perf' : val_result,
    'lambda_value' : lambda_result
})

lasso_perf_df['valid_perf'].min()
lasso_perf_df[lasso_perf_df['valid_perf']==lasso_perf_df['valid_perf'].min()]
lasso_perf_df['train_perf'].min()
lasso_perf_df[lasso_perf_df['train_perf']==lasso_perf_df['train_perf'].min()]


model.fit(train_x, train_y)






# --------------------------------------
# 최적 람다 찾기
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import norm, uniform
from sklearn.linear_model import Lasso
from sklearn.model_selection import cross_val_score, KFold
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
from sklearn.metrics import make_scorer, mean_squared_error

len(train_df)

train_df.info()
train_df.columns
len(train_df.columns)
train_df['MiscVal']

df = pd.concat([train_df, test_df], axis=0)
len(df)
df['MSSubClass'] = df['MSSubClass'].astype(object)
numeric_df = df.select_dtypes(include = [int,float])
category_df = df.select_dtypes(include = [object])

numeric_df.columns
len(numeric_df.columns)
len(category_df.columns)


# 수치 컬럼 결측치 처리
nan_numeric_df = numeric_df.isna().sum() 
nan_numeric_df[nan_numeric_df>0]
fill_values = { 'LotFrontage' : 0 , 'MasVnrArea' :0 , 'GarageYrBlt' :0, 'BsmtFinSF1':0, 'BsmtFinSF2':0, 'BsmtUnfSF':0, 'TotalBsmtSF':0, 'BsmtFullBath':0, 'BsmtHalfBath':0, 'GarageCars':0, 'GarageArea':0 }
numeric_df = numeric_df.fillna(value= fill_values)

nan_numeric_df = numeric_df.isna().sum() 
nan_numeric_df[nan_numeric_df>0]   



qual_selected = category_df.columns[category_df.isna().sum() > 0]

for col in qual_selected:   # 이거 하니까 0.001 더 안 좋아짐
    category_df[col].fillna("unknown", inplace=True)
category_df[qual_selected].isna().sum()


# 범주 컬럼 결측치 처리
# 결측치가 다 없어서 그런 것 같아서 따로 안 채움
nan_category_df = category_df.isna().sum() 
nan_category_df[nan_category_df>0]


category_dummies = pd.get_dummies(
                        category_df, 
                        columns=category_df.columns,
                        drop_first=True)

category_dummies.columns

df2 = pd.concat([numeric_df , category_dummies], axis=1)
df2.columns
df2 = df2.drop(columns=('Id'))
df2.columns

train = df2[:1460]
test = df2[1460:]

train_x = train.drop(columns=('SalePrice'))
train_y = train['SalePrice']
test_x = test.drop(columns=('SalePrice'))




kf = KFold(n_splits=5, shuffle=True, random_state=2024)

def rmse(model):
    score = np.sqrt(-cross_val_score(model, train_x, train_y, cv = kf,
                                     n_jobs=-1, scoring = "neg_mean_squared_error").mean())
    return(score)                   # n_jobs=-1 : 계산 가능한 공간이 4개가 있는데 각 공간마다 fold1이 valid일 때의 계산, fold2가 valid일 때의 계산, ... 을 맡겨서 계산을 더 빠르게 하는 것임

lasso = Lasso(alpha=0.01)
rmse(lasso)

# 각 알파 값에 대한 교차 검증 점수 저장
alpha_values = np.arange(160, 170, 0.1)
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

# 결과 시각화
plt.plot(df['lambda'], df['validation_error'], label='Validation Error', color='red')
plt.xlabel('Lambda')
plt.ylabel('Mean Squared Error')
plt.legend()
plt.title('Lasso Regression Train vs Validation Error')
plt.show()

# 최적의 alpha 값 찾기
optimal_alpha = df['lambda'][np.argmin(df['validation_error'])]
print("Optimal lambda:", optimal_alpha)




# 더미 변수


# for i in category_df.columns:
#    if pd.isna(train_df[i].unique()).any():
#        category_test[i]= pd.Categorical(category_test[i], categories= train_df[i].unique()[~pd.isna(train_df[i].unique())])
#    else:
#        category_test[i]= pd.Categorical(category_test[i], categories= train_df[i].unique())

#category_test['BldgType']= pd.Categorical(category_test['BldgType'], categories= train_df['Electrical'].unique()[~pd.isna(train_df['Electrical'].unique())])




len(train_x.columns)
len(test.columns)

test_df['BsmtFinType1'].unique()
train_df['BsmtFinType1'].unique()

sum(train_x.columns == test.columns)
train_x.columns.sort_values()
test.columns.sort_values()

train_x = train_x[train_x.columns.sort_values()].head()
test = test[test.columns.sort_values()].head()

train_df['Condition2'].unique()
test.columns[50:100]



# 1차 : 164.3, 2차 : 164.1200000000328
model = Lasso(alpha = 163.89999999999978)

model.fit(train_x, train_y)
test_y_pred = model.predict(test_x)
submission['SalePrice'] = test_y_pred
submission.to_csv('lasso_allvars3.csv', index=False)







# ---- 릿지

kf = KFold(n_splits=5, shuffle=True, random_state=2024)

def rmse(model):
    score = np.sqrt(-cross_val_score(model, train_x, train_y, cv = kf,
                                     n_jobs=-1, scoring = "neg_mean_squared_error").mean())
    return(score)                   # n_jobs=-1 : 계산 가능한 공간이 4개가 있는데 각 공간마다 fold1이 valid일 때의 계산, fold2가 valid일 때의 계산, ... 을 맡겨서 계산을 더 빠르게 하는 것임

ridge = Ridge(alpha=0.01)
rmse(ridge)



model = Ridge(alpha = 164.3)

model.fit(train_x, train_y)
test_y_pred = model.predict(test_x)
submission['SalePrice'] = test_y_pred
submission.to_csv('lasso_allvars.csv', index=False)