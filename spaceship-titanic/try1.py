import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler, OneHotEncoder, MinMaxScaler, PolynomialFeatures
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

train_df = pd.read_csv('./spaceship-titanic/train.csv')
test_df = pd.read_csv('./spaceship-titanic/test.csv')
submission = pd.read_csv('./spaceship-titanic/sample_submission.csv')

train_df.head()
train_df.columns
# ['PassengerId', 'HomePlanet', 'CryoSleep', 'Cabin', 'Destination', 'Age',
#        'VIP', 'RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck',
#        'Name', 'Transported']

train_df.isna().sum()
test_df.isna().sum()
train_df.info()

train_df = train_df.drop(columns=('PassengerId'))
test_df = test_df.drop(columns=('PassengerId'))

## train 범주형 채우기
qualitative = train_df.select_dtypes(include = [object])
qualitative.isna().sum()
qual_selected = qualitative.columns[qualitative.isna().sum() > 0]

for col in qual_selected:
    train_df[col].fillna("unknown", inplace=True)
train_df[qual_selected].isna().sum()


## test 범주형 채우기
qualitative = test_df.select_dtypes(include = [object])
qualitative.isna().sum()
qual_selected = qualitative.columns[qualitative.isna().sum() > 0]

for col in qual_selected:
    test_df[col].fillna("unknown", inplace=True)
test_df[qual_selected].isna().sum()




## train 숫자형 채우기
quantitative = train_df.select_dtypes(include = [int, float])
quantitative.isna().sum()
quant_selected = quantitative.columns[quantitative.isna().sum() > 0]

for col in quant_selected:
    train_df[col].fillna(train_df[col].mean(), inplace=True)
train_df[quant_selected].isna().sum()




## test 숫자형 채우기
quantitative = test_df.select_dtypes(include = [int, float])
quantitative.isna().sum()
quant_selected = quantitative.columns[quantitative.isna().sum() > 0]

for col in quant_selected:
    test_df[col].fillna(train_df[col].mean(), inplace=True)
test_df[quant_selected].isna().sum()


train_df.columns
train_df.shape
test_df.shape
train_n=len(train_df)

# 통합 df 만들기 + 더미코딩
# test_df.select_dtypes(include=[int, float])
train_df1= train_df.drop(columns = ('Transported'))
df = pd.concat([train_df1, test_df], ignore_index=True)
# df.info()
df = pd.get_dummies(
    df,
    columns= df.select_dtypes(include=[object]).columns,
    drop_first=True
    )
df

# train / test 데이터셋
train_x=df.iloc[:train_n,]
train_y = train_df[['Transported']]
test_x=df.iloc[train_n:,]

## 이상치 탐색
# train_df=train_df.query("GrLivArea <= 4500")

plt.scatter(train_df[train_df.columns[0]],train_df['Transported'])
train_df.columns[34]

train_df=train_df.query("LotFrontage <= 300")
train_df=train_df.query("~ ((YearBuilt < 1900) & (SalePrice>400000))& ~((YearBuilt > 1980) & (SalePrice>700000))")




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







# 모델


from sklearn.tree import DecisionTreeClassifier 
from sklearn.model_selection import GridSearchCV


dt_model = DecisionTreeClassifier(random_state=20240906, criterion = 'entropy')
# criterion : gini, entropy, log_loss  (지표가 3가지 있음)


param_grid = {
    'max_depth' : np.arange(2,15),
    'min_samples_split' : np.arange(5, 15)
}

np.random.seed(20240906)
grid_search = GridSearchCV(
    estimator = model,
    param_grid = param_grid,
    scoring = 'accuracy',
    cv=5
)


grid_search.fit(train_x, train_y)
grid_search.best_params_  


from sklearn.ensemble import BaggingClassifier
bagging_model = BaggingClassifier(DecisionTreeClassifier(),  # 기본 모델로 결정 트리 사용
    n_estimators=100,                        # 사용할 기본 모델의 개수
    max_samples=100,                         # 각 기본 모델이 훈련하는 샘플의 최대 수
    n_jobs=-1,                               # 모든 CPU 코어를 사용하여 병렬 처리
    random_state=20240906                    # 재현성을 위한 랜덤 시드
)
  # for문 처럼 decisiontreeclassifier 모델을 50번 fit한다.(bagging에 사용할 모델 개수 50개) max_sample : 하나의 subdata를 행 100으로 한다. , n_jobs=-1 : 계산 메모리 4개를 동시에 사용해서 한 번에 모델 4개를 fit할 수 있다. ,random_state : 나중에 bootstrap으로 할 때 동일한 데이터로 하기 위해서
  # 이건 랜덤포레스트 아이디어임. (랜덤포레스트 옵션과, 이렇게 할 때 하는 옵션이 조금 다르긴 함)
  # baggingclassifier은 bootstrap도 되고, 열도 랜덤하게 선택할 수 있음(random sub space : sub space마다 열 달라짐) <- 랜덤포레스트의 더 general한 경우
  # 랜덤포레스트는 행 랜덤 추출만(bootstrap)

bagging_model.fit(train_x, train_y)
bagging_y_pred = bagging_model.predict(test_x)


#
from sklearn.ensemble import RandomForestClassifier
rf_model = RandomForestClassifier(n_estimators=100, random_state=20240906)
rf_model.fit(train_x, train_y)

y_pred = rf_model.predict(test_x)


# 6. 결과 평가
# 정확도를 계산하고, 분류 보고서를 출력합니다.
accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred)

print(f'Accuracy: {accuracy:.2f}')
print('Classification Report:')
print(report)




# SalePrice 바꿔치기
submission["Transported"] = y_pred
submission

# csv 파일로 내보내기
submission.to_csv("./spaceship-titanic/bagging.csv", index=False)