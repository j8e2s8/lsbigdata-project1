# 표준화랑 one hot encoding 하다가 때려침


# 필요한 패키지 불러오기
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler, OneHotEncoder, MinMaxScaler, PolynomialFeatures
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline



## 필요한 데이터 불러오기
house_train=pd.read_csv("./lsbigdata-project1/house price/train.csv")
house_test=pd.read_csv("./lsbigdata-project1/house price/test.csv")
sub_df=pd.read_csv("./lsbigdata-project1/house price/sample_submission.csv")

house_train = house_train.drop(columns = ('Id'))
house_test = house_test.drop(columns = ('Id'))
house_train['MSSubClass'] = house_train['MSSubClass'].astype('object')
house_test['MSSubClass'] = house_test['MSSubClass'].astype('object')




## train 범주형 채우기
qualitative = house_train.select_dtypes(include = [object])
qualitative.isna().sum()
qual_selected = qualitative.columns[qualitative.isna().sum() > 0]

for col in qual_selected:
    house_train[col].fillna("unknown", inplace=True)
house_train[qual_selected].isna().sum()


## test 범주형 채우기
qualitative = house_test.select_dtypes(include = [object])
qualitative.isna().sum()
qual_selected = qualitative.columns[qualitative.isna().sum() > 0]

for col in qual_selected:
    house_test[col].fillna("unknown", inplace=True)
house_test[qual_selected].isna().sum()



df = pd.concat([house_train, house_test], axis=0)
col_list =df.select_dtypes(include = [object]).columns
col_list1 = col_list.drop(['ExterQual','ExterCond','BsmtQual','BsmtCond','BsmtFinType1','BsmtFinType2','HeatingQC','KitchenQual','FireplaceQu','GarageQual','GarageCond','PoolQC'])


for col in col_list1:
    df[col] = pd.Categorical(df[col], categories=df[col].unique())

df['ExterQual'] = pd.Categorical(df['ExterQual'], categories=['Po','Fa','TA','Gd','Ex'] , ordered=True)
df['ExterCond'] = pd.Categorical(df['ExterCond'], categories=['Po','Fa','TA','Gd','Ex'] , ordered=True)
df['BsmtQual'] = pd.Categorical(df['BsmtQual'], categories=['Po','Fa','TA','Gd','Ex'] , ordered=True)
df['BsmtCond'] = pd.Categorical(df['BsmtCond'], categories=['Po','Fa','TA','Gd','Ex'] , ordered=True)
df['BsmtFinType1'] = pd.Categorical(df['BsmtFinType1'], categories=['Unf','LwQ','Rec','BLQ','ALQ', 'GLQ'] , ordered=True)
df['BsmtFinType2'] = pd.Categorical(df['BsmtFinType2'], categories=['Unf','LwQ','Rec','BLQ','ALQ', 'GLQ'] , ordered=True)
df['HeatingQC'] = pd.Categorical(df['HeatingQC'], categories=['Po','Fa','TA','Gd','Ex'] , ordered=True)
df['KitchenQual'] = pd.Categorical(df['KitchenQual'], categories=['Po','Fa','TA','Gd','Ex'] , ordered=True)
df['FireplaceQu'] = pd.Categorical(df['FireplaceQu'], categories=['Po','Fa','TA','Gd','Ex'] , ordered=True)
df['GarageQual'] = pd.Categorical(df['GarageQual'], categories=['Po','Fa','TA','Gd','Ex'] , ordered=True)
df['GarageCond'] = pd.Categorical(df['GarageCond'], categories=['Po','Fa','TA','Gd','Ex'] , ordered=True)
df['PoolQC'] = pd.Categorical(df['PoolQC'], categories=['Po','Fa','TA','Gd','Ex'] , ordered=True)

house_train = df.iloc[:len(house_train),:]
house_test = df.iloc[len(house_train):,:]

house_test['ExterQual'].unique()

## NaN 채우기
# 각 숫치형 변수는 평균 채우기
# 각 범주형 변수는 Unknown 채우기
house_train.isna().sum()
house_test.isna().sum()

## train 숫자형 채우기
quantitative = house_train.select_dtypes(include = [int, float])
quantitative.isna().sum()
quant_selected = quantitative.columns[quantitative.isna().sum() > 0]

for col in quant_selected:
    house_train[col].fillna(house_train[col].mean(), inplace=True)
house_train[quant_selected].isna().sum()




## test 숫자형 채우기
quantitative = house_test.select_dtypes(include = [int, float])
quantitative.isna().sum()
quant_selected = quantitative.columns[quantitative.isna().sum() > 0]

for col in quant_selected:
    house_test[col].fillna(house_train[col].mean(), inplace=True)
house_test[quant_selected].isna().sum()


# 종속 변수와 특성 변수 분리
train_y = house_train['SalePrice'] 
train_x = house_train.drop(columns='SalePrice')
test_x = house_test.drop(columns='SalePrice')




# 각 변수를 위한 변환기 정의
numeric_transformer = StandardScaler()
categorical_transformer = OneHotEncoder()  # handle_unknown='ignore'

# ColumnTransformer 설정
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, train_x.select_dtypes(include=[int, float]).columns),
        ('cat', categorical_transformer, col_list)
    ])

# 전체 파이프라인 생성
pipeline = Pipeline(steps=[('preprocessor', preprocessor)])

# 데이터 변환
X_train_transformed = pipeline.fit_transform(house_train)
X_test_transformed = pipeline.transform(house_test)

# 변환된 데이터 출력
print("Transformed Training Data:")
print(X_train_transformed)

print("Transformed Test Data:")
print(X_test_transformed)

train_x.info()



house_train.shape
house_test.shape
train_n=len(house_train)

# 통합 df 만들기 + 더미코딩
# house_test.select_dtypes(include=[int, float])

df = pd.concat([house_train, house_test], ignore_index=True)
# df.info()
df = pd.get_dummies(
    df,
    columns= df.select_dtypes(include=[object]).columns,
    drop_first=True
    )
df

# train / test 데이터셋
train_df=df.iloc[:train_n,]
test_df=df.iloc[train_n:,]

## 이상치 탐색
# train_df=train_df.query("GrLivArea <= 4500")

## train
train_x=train_df.drop("SalePrice", axis=1)
train_y=train_df["SalePrice"]

## test
test_x=test_df.drop("SalePrice", axis=1)

from sklearn.linear_model import ElasticNet, Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV

eln_model= ElasticNet(random_state=20240911)
rf_model= RandomForestRegressor(n_estimators=100, random_state=20240911) # n_estimators : 트리 개수 (모델 개수)로 grid search로 최적의 값을 찾는 값이 아니라 내가 선택하는 것임 (하이퍼 파라미터 인건 맞는데, 하나하나 찾기엔 너무 시간이 오래걸림)

np.random.seed(20240911)                                                 
# eln
param_grid={
    'alpha': np.arange(0,0.2,0.01),  # 63.3 나옴
    'l1_ratio': [0.9]  # np.arange(0, 1.1, 0.1)  # [0, 0.1, 0.5, 1.0]  # 1.0 나옴
}

grid_search=GridSearchCV(
    estimator=eln_model,
    param_grid=param_grid,
    scoring='neg_mean_squared_error',
    cv=10
)

grid_search.fit(train_x, train_y)

grid_search.best_params_ # {'alpha': 0.1, 'l1_ratio': 0.9}  # {'alpha' : 63.3, 'l1_ratio' : 1.0}  # {'alpha': 100.0, 'l1_ratio': 1.0}
grid_search.cv_results_
result = pd.DataFrame(grid_search.cv_results_)
grid_search.best_score_  # (0.1, 0.9) -1032933901.414933   # (63.3, 1.0) -571887395.0411156
best_eln_model=grid_search.best_estimator_  # eln에 대해서는 베스트 모델 찾는 거 끝냄


# 랜포
np.random.seed(20240911)     
param_grid={
    'max_depth':  np.arange(12,15),  # np.arange(9,12) ,  #  np.arange(2,10), # 9나옴
    'min_samples_split': np.arange(4,10), #[10, 7, 5],  # np.arange(3,10),  # [20,10,5], # 5  # 쪼갤 때마다 최소한으로 남겨지는 데이터 개수
    'min_samples_leaf' : np.arange(2,6), #[3, 5, 10], # np.arange(3,10), # [5,10,20, 30], # 5  # 맨 마지막에 남겨지는 리프 노드의 최소한의 데이터 개수
    'max_features' : [None]  # ['sqrt', 'log2', None]  #  268개의 컬럼이 있는데, 고려되는 변수를 268개로 하거나, 그거의 루트 값과 log2 값을 이용도 해보겠다는 것임
}

grid_search=GridSearchCV(
    estimator=rf_model,
    param_grid=param_grid,
    scoring='neg_mean_squared_error',
    cv=10
)

grid_search.fit(train_x, train_y)

grid_search.best_params_  

# {'max_depth': 13,   # -919152736.2004478
#  'max_features': None,
#  'min_samples_leaf': 2,
#  'min_samples_split': 6}



# {'max_depth': 13, # -800777969.879989
#  'max_features': None,
#  'min_samples_leaf': 6,
#  'min_samples_split': 3}

# {'max_depth': 11,  # -736566267.4508291
#  'max_features': None,
#  'min_samples_leaf': 3,
#  'min_samples_split': 6}


# {'max_depth': 9,   # -785380413.946151
#  'max_features': None,
#  'min_samples_leaf': 5,
#  'min_samples_split': 5}

# {'max_depth': 7,
#  'max_features': None,
#  'min_samples_leaf': 5,
#  'min_samples_split': 10}

grid_search.best_score_ 
best_rf_model=grid_search.best_estimator_  # eln에 대해서는 베스트 모델 찾는 거 끝냄


# 근데 stacking 하고 싶으면 train_x로 예측해야함
train_pred_y_eln = best_eln_model.predict(train_x) # eln의 test 셋 예측값
train_pred_y_rf = best_rf_model.predict(train_x) # eln의 test 셋 예측값

stacking_train_x = pd.DataFrame({
    'y1' : train_pred_y_eln,
    'y2' : train_pred_y_rf
})


# 릿지 블랜더
from sklearn.linear_model import Ridge

np.random.seed(20240911)
rg_model = Ridge()
param_grid = {
    'alpha' : np.arange(0, 0.5, 0.01)
}

grid_search = GridSearchCV(
    estimator = rg_model,
    param_grid = param_grid,
    scoring='neg_mean_squared_error',
    cv = 10
)

grid_search.fit(stacking_train_x, train_y)

grid_search.best_params_  # 0 # {'alpha': 9.950000000000001}

grid_search.cv_results_
result = pd.DataFrame(grid_search.cv_results_)
grid_search.best_score_   # -174690304.5743541

blander_model = grid_search.best_estimator_

blander_model.coef_ # array([0.3583818 , 0.70122774]) <- 랜포가 더 좋다고 생각해서 비중을 더 키운 것임
blander_model.intercept_  # -10825.640652675705  <- 두 모델 오버슈팅 된 것 같아서 줄여줌




test_pred_y_eln=best_eln_model.predict(test_x) # test 셋에 대한 집값
test_pred_y_rf=best_rf_model.predict(test_x) # test 셋에 대한 집값

test_x_stack=pd.DataFrame({
    'y1': test_pred_y_eln,
    'y2': test_pred_y_rf
})

pred_y=blander_model.predict(test_x_stack)

# SalePrice 바꿔치기
sub_df["SalePrice"] = pred_y
sub_df

# # csv 파일로 내보내기
sub_df.to_csv("./lsbigdata-project1/house price/std_eln_rf_stacking.csv", index=False)












pred_y=best_model.predict(test_x) # test 셋에 대한 집값
pred_y




















# SalePrice 바꿔치기
sub_df["SalePrice"] = pred_y
sub_df

# csv 파일로 내보내기
sub_df.to_csv("./data/houseprice/sample_submission10.csv", index=False)