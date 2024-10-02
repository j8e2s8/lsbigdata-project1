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


house_train.isna().sum()
house_test.isna().sum()

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

plt.scatter(train_df[train_df.columns[45]],train_df['SalePrice'])
train_df.columns[34]

train_df=train_df.query("LotFrontage <= 300")
train_df=train_df.query("~ ((YearBuilt < 1900) & (SalePrice>400000))& ~((YearBuilt > 1980) & (SalePrice>700000))")


## train
train_x=train_df.drop("SalePrice", axis=1)
train_y=train_df["SalePrice"]

## test
test_x=test_df.drop("SalePrice", axis=1)




# 표준화 (표준화 안 한 걸로 하나, 표준화 해본 걸로 하나 해보기)
standard_scaler = StandardScaler()
standardized_data = standard_scaler.fit_transform(train_x)
train_x[[i + '_standard' for i in col]] = standardized_data

train_x.columns
len(train_x.columns) # 105
train_s_x = train_x.drop(columns=col)
len(train_s_x.columns)  # 102



standardized_data = standard_scaler.fit_transform(test_x[[i for i in col]])
test_x[[i + '_standard' for i in col]] = standardized_data

test_x.columns
len(test_x.columns) # 105
test_s_x = test_x.drop(columns=col)
len(test_s_x.columns)  # 102










from sklearn.linear_model import ElasticNet, Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV

train_eln_y = np.log(train_y)



eln_model= ElasticNet(random_state=20240911)
rf_model= RandomForestRegressor(n_estimators=100, random_state=20240911) # n_estimators : 트리 개수 (모델 개수)로 grid search로 최적의 값을 찾는 값이 아니라 내가 선택하는 것임 (하이퍼 파라미터 인건 맞는데, 하나하나 찾기엔 너무 시간이 오래걸림)
xg_model = xgb.XGBRegressor(random_state=42, objective='reg:squarederror' # 회귀 문제의 경우 'reg:squarederror' 
   , n_estimators=1000 )  # 트리 개수

np.random.seed(20240911)                                                 
# eln
param_grid={
    'alpha': np.arange(0,0.2,0.01), # 0.01   # 63.3 나옴
    'l1_ratio':  np.arange(0, 1.1, 0.1)  # [0, 0.1, 0.5, 1.0]  # 1.0 나옴
}

grid_search=GridSearchCV(
    estimator=eln_model,
    param_grid=param_grid,
    scoring='neg_mean_squared_error',
    cv=10
)

grid_search.fit(train_x, train_eln_y)

grid_search.best_params_ # {'alpha': 0.01, 'l1_ratio': 0.0}  # {'alpha': 0.1, 'l1_ratio': 0.9}  # {'alpha' : 63.3, 'l1_ratio' : 1.0}  # {'alpha': 100.0, 'l1_ratio': 1.0}
grid_search.cv_results_
result = pd.DataFrame(grid_search.cv_results_)
grid_search.best_score_ #(0.01, 0.0) -0.020308981272587047 #(0.01, 0.0) -0.015139883889228562  # (0.1, 0.9) -1032933901.414933   # (63.3, 1.0) -571887395.0411156
best_eln_model=grid_search.best_estimator_  # eln에 대해서는 베스트 모델 찾는 거 끝냄


# 랜포
np.random.seed(20240911)     
param_grid={
    'max_depth':  np.arange(9,15) ,  #  np.arange(2,10), # 9나옴
    'min_samples_split': np.arange(5,10), #[10, 7, 5],  # np.arange(3,10),  # [20,10,5], # 5  # 쪼갤 때마다 최소한으로 남겨지는 데이터 개수
    'min_samples_leaf' :  np.arange(3,6) , #[3, 5, 10], # np.arange(3,10), # [5,10,20, 30], # 5  # 맨 마지막에 남겨지는 리프 노드의 최소한의 데이터 개수
    'max_features' :   ['sqrt', 'log2', None]  #  268개의 컬럼이 있는데, 고려되는 변수를 268개로 하거나, 그거의 루트 값과 log2 값을 이용도 해보겠다는 것임
}

grid_search=GridSearchCV(
    estimator=rf_model,
    param_grid=param_grid,
    scoring='neg_mean_squared_error',
    cv=10
)

grid_search.fit(train_x, train_y)

grid_search.best_params_  

# {'max_depth': 13,   # -928830193.0184815
#  'max_features': None,
#  'min_samples_leaf': 3,
#  'min_samples_split': 7}

# {'max_depth': 13,   # -782549792.3502753
#  'max_features': None,
#  'min_samples_leaf': 4,
#  'min_samples_split': 9}


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




# xgboosting
# !pip install xgboost   # xgboost는 conda로 설치하면 에러남 pip으로 할 것
import xgboost as xgb

np.random.seed(20240911)     
param_grid={
    'learning_rate': [0.01, 0.1, 0.2],      # 학습률      
    'max_depth': np.arange(2,15)              # 트리의 최대 깊이
}

#    'n_estimators': [100, 200, 300],  # 트리 개수
#     'learning_rate': [0.01, 0.1, 0.2],  # 학습률
#     'max_depth': np.arange(2, 15),  # 트리의 최대 깊이
#     'min_child_weight': [1, 5, 10],  # 자식 노드에서 최소 샘플 수
#     'subsample': [0.8, 0.9, 1.0],  # 각 트리를 학습할 때 샘플 비율
#     'colsample_bytree': [0.8, 0.9, 1.0]  # 각 트리를 학습할 때 사용하는 피처 비율


grid_search=GridSearchCV(
    estimator=xg_model,
    param_grid=param_grid,
    scoring='neg_mean_squared_error',
    cv=10
)

grid_search.fit(train_x, train_y) 

grid_search.best_params_  # {'learning_rate': 0.1, 'max_depth': 2}


grid_search.best_score_   # -639005998.7144048
best_xg_model=grid_search.best_estimator_  # eln에 대해서는 베스트 모델 찾는 거 끝냄




# 근데 stacking 하고 싶으면 train_x로 예측해야함
train_pred_y_eln = best_eln_model.predict(train_x) # eln의 test 셋 예측값
train_pred_y_eln = np.exp(train_pred_y_eln)
train_pred_y_rf = best_rf_model.predict(train_x) # eln의 test 셋 예측값
train_pred_y_xg = best_xg_model.predict(train_x)  # eln의 test 셋 예측값

stacking_train_x = pd.DataFrame({
    'y1' : train_pred_y_eln,
    'y2' : train_pred_y_rf ,
    'y3' : train_pred_y_xg
})





# xg 블랜더
np.random.seed(20240911)     
xg_model = xgb.XGBRegressor(objective='reg:squarederror' # 회귀 문제의 경우 'reg:squarederror' 
   , n_estimators=1000 )  # 트리 개수

param_grid={
    'learning_rate': [0.01, 0.1, 0.2],      # 학습률      
    'max_depth': np.arange(2,15)              # 트리의 최대 깊이
}

grid_search=GridSearchCV(
    estimator=xg_model,
    param_grid=param_grid,
    scoring='neg_mean_squared_error',
    cv=10
)

grid_search.fit(stacking_train_x, train_y)

grid_search.best_params_   # {'learning_rate': 0.01, 'max_depth': 3}

blander_model = grid_search.best_estimator_






# 릿지 블랜더 (생략)

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


# ----------

test_pred_y_eln =best_eln_model.predict(test_x) # test 셋에 대한 집값
test_pred_y_rf =best_rf_model.predict(test_x) # test 셋에 대한 집값
test_pred_y_xg =best_xg_model.predict(test_x) # test 셋에 대한 집값


stacking_test_x =pd.DataFrame({
    'y1': test_pred_y_eln,
    'y2': test_pred_y_rf,
    'y3' : test_pred_y_xg
})

pred_y=blander_model.predict(stacking_test_x)

# SalePrice 바꿔치기
sub_df["SalePrice"] = pred_y
sub_df

# # csv 파일로 내보내기
sub_df.to_csv("./lsbigdata-project1/house price/eln_rf_xg_stacking(xg).csv", index=False)












pred_y=best_model.predict(test_x) # test 셋에 대한 집값
pred_y




















# SalePrice 바꿔치기
sub_df["SalePrice"] = pred_y
sub_df

# csv 파일로 내보내기
sub_df.to_csv("./data/houseprice/sample_submission10.csv", index=False)