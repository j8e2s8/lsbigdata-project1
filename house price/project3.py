# house price
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

# 데이터 불러오기  (데이터 불러왔을 때 결측값 채워주는게 가장 깔끔한 듯함)
train_df = pd.read_csv('./train.csv')
test_df = pd.read_csv('./test.csv')
submission = pd.read_csv('./sample_submission.csv')

train_df.shape
test_df.shape


# Nan 채우기  (nan vs. NaN <- Not a Number)
# train과 test를 한번에 concat써서 결측값을 대체하면 안됨.  (test set에 train set 정보가 녹아들어감)
# train 따로 test 따로 결측치 채움
train_df.isna().sum()
test_df.isna().sum()

# 각 숫자변수는 평균 채우기
# 각 범주형 변수는 최빈값 채우기  (근데 변수 의미상 NaN가 데이터 수집이 안 된게 아니라, '없다'의 의미의 NaN이라면 'unknown' 'None' 으로 채워서 더미코딩에 포함되게 해야 할 수도 있음)

# 숫자형 결측치 처리 : 평균값 채우기
quantitative = train_df.select_dtypes(include=[int, float])  # quantitative : 수치형
quantitative.isna().sum()
quant_selected = quantitative.columns[quantitative.isna().sum()>0]

for col in quant_selected:
    train_df[col].fillna(train_df[col].mean(), inplace=True)
train_df[quant_selected].isna().sum()

# 범주형 결측치 처리  : 'unknown'이라는 범주로 채우기
qualitative = train_df.select_dtypes(include = [object])
qualitative.isna().sum()
qual_selected = qualitative.columns[qualitative.isna().sum()>0]

for col in qual_selected:
    train_df[col].fillna('unknown', inplace=True)
train_df[qual_selected].isna().sum()




# 더미변수 만들기
df = pd.concat([train_df, test_df], ignore_index = True)  # train, test를 합쳐서 한 번에 더미변수 만들려고

neighborhood_dummies = pd.get_dummies(
    df['Neighborhood'], drop_first = True
)

x = pd.concat([df[['GrLivArea', 'GarageArea']], neighborhood_dummies], axis=1)
y=df['SalePrice']

train_x = x.iloc[:1460, :]
test_x = x.iloc[1460:, :]
train_y = y[:1460]
test_y = y[1460:]



# validation set 3개 만들기
1460 * 0.3
np.random.randint(0, 1459, size =438)  # 중복되게 뽑힐 수 있음

np.random.seed(42)
val_index = np.random.choice(np.arange(1460), size = 438 ,replace=False)
valid_x = train_x.loc[val_index]
train_x = train_x.drop(val_index)
valid_y = train_y[val_index]
train_y = train_y.drop(val_index)


# 선형 회귀 모델 생성
model = LinearRegression()

model.fit(train_x, train_y)

y_hat = model.predict(valid_x)
np.mean((valid_y - y_hat)**2)
np.sqrt(np.mean((valid_y - y_hat)**2))




# --------------------------------------------------
# 이상치 제거 먼저하고, train test 구분하고, 모델 피팅해보기 <- 이렇게 하면 안됨. 이상치를 제거하기 전과 이상치 제거한 후의 모델 성능을 비교할 건데
# 이상치 제거를 먼저하게되면 valid set이 바뀌기 때문에 이상치 제거 전과 성능 비교를 할 수가 없게 됨.
# train test 구분하고, 이상치 제거 하고, 모델 피팅해보기



# 트레인, 테스트 합치기(더미변수 만드는거 한 번에 처리하기 위해서 더하는거.)
df = pd.concat([train_df, test_df], ignore_index = True) # ignore_index 옵션이 있음. # test_df에는 y컬럼이 없어도 concat으로 이어붙일 수 있음.

# 더미변수 만들기
df_dummies = pd.get_dummies(
    df,
    columns = ['Neighborhood'],
    drop_first=True
)

train_n = len(train_df) # 1460

# train과 test 나누기
train_dummies = df_dummies.iloc[:train_n,]
test_dummies = df_dummies.iloc[train_n:,]


## Validation set(모의고사 set) 만들기
np.random.seed(42)
val_index = np.random.choice(np.arange(train_n), size = 438,
                 replace = False) #30% 정도의 갯수를 랜덤으로 고르기.
valid = train_dummies.loc[val_index]  # 30% 438개
train = train_dummies.drop(val_index) # 70% 1022개
test = test_dummies.copy()

######## 이상치 탐색 및 없애기 (여기서 이상치는 train과 valid를 합친 데이터 상에서의 이상치임)
train = train.query("GrLivArea <= 4500") # 나중에 실행하지 말고도 구해보기.  
                           # 만약 이상치가 valid에 속해있어서 valid set에서는 제거되지 않아도, 우리가 비교할 모델들 또한 동일한 valid set으로 평가할 거기 때문에 공평함. 괜춘


# x, y 나누기
# train_df.columns.str.contains('Neighborhood_')  # 나중에 더미변수로 바꾼 Neighborhood_ 로 시작하는 컬럼을 불러올 수 있음.
# regex (Regular Expression, 정규방정식)
selected_columns = train.filter(regex= '^GrLivArea$|^GarageArea$|^Neighborhood_').columns # 특정 컬럼과 더미변수만 사용할거임
train_x = train[selected_columns]
train_y = train['SalePrice']

len(train.columns)
len(train_x.columns)



# 범주가 숫자여도 order 의미가 없으면 더미코딩 해줘야하고, 범주여도 order 의미가 있으면 숫자 코딩 해줘야함.



# -----------------------------------
# 범주컬럼 2개 이상 더미코딩 하는 법

# 트레인, 테스트 합치기(더미변수 만드는거 한 번에 처리하기 위해서 더하는거.)
df = pd.concat([train_df, test_df], ignore_index = True) # ignore_index 옵션이 있음. # test_df에는 y컬럼이 없어도 concat으로 이어붙일 수 있음.

# 더미변수 만들기
df_dummies = pd.get_dummies(
    df,
    columns = ['Neighborhood', 'MSZoning'],
    drop_first=True
)

train_n = len(train_df) # 1460

# train과 test 나누기
train_dummies = df_dummies.iloc[:train_n,]
test_dummies = df_dummies.iloc[train_n:,]


## Validation set(모의고사 set) 만들기
np.random.seed(42)
val_index = np.random.choice(np.arange(train_n), size = 438,
                 replace = False) #30% 정도의 갯수를 랜덤으로 고르기.
valid = train_dummies.loc[val_index]  # 30% 438개
train = train_dummies.drop(val_index) # 70% 1022개
test = test_dummies.copy()

######## 이상치 탐색 및 없애기 (여기서 이상치는 train과 valid를 합친 데이터 상에서의 이상치임)
train = train.query("GrLivArea <= 4500") # 나중에 실행하지 말고도 구해보기.  
                           # 만약 이상치가 valid에 속해있어서 valid set에서는 제거되지 않아도, 우리가 비교할 모델들 또한 동일한 valid set으로 평가할 거기 때문에 공평함. 괜춘


# x, y 나누기
# train_df.columns.str.contains('Neighborhood_')  # 나중에 더미변수로 바꾼 Neighborhood_ 로 시작하는 컬럼을 불러올 수 있음.
# regex (Regular Expression, 정규방정식)
selected_columns = train.filter(regex= '^GrLivArea$|^GarageArea$|^Neighborhood_').columns
train_x = train[selected_columns]
train_y = train['SalePrice']

len(train.columns)
len(train_x.columns)




# --------------------------------
# 모든 범주컬럼 다 더미코딩하기

# 트레인, 테스트 합치기(더미변수 만드는거 한 번에 처리하기 위해서 더하는거.)
df = pd.concat([train_df, test_df], ignore_index = True) # ignore_index 옵션이 있음. # test_df에는 y컬럼이 없어도 concat으로 이어붙일 수 있음.


# 더미변수 만들기
df_dummies = pd.get_dummies(
    df,
    columns = df.select_dtypes(include=[object]).columns,
    drop_first=True
)

train_n = len(train_df) # 1460

# train과 test 나누기
train_dummies = df_dummies.iloc[:train_n,]
test_dummies = df_dummies.iloc[train_n:,]


## Validation set(모의고사 set) 만들기
np.random.seed(42)
val_index = np.random.choice(np.arange(train_n), size = 438,
                 replace = False) #30% 정도의 갯수를 랜덤으로 고르기.
valid = train_dummies.loc[val_index]  # 30% 438개
train = train_dummies.drop(val_index) # 70% 1022개
test = test_dummies.copy()

######## 이상치 탐색 및 없애기 (여기서 이상치는 train과 valid를 합친 데이터 상에서의 이상치임)
train = train.query("GrLivArea <= 4500") # 나중에 실행하지 말고도 구해보기.  
                           # 만약 이상치가 valid에 속해있어서 valid set에서는 제거되지 않아도, 우리가 비교할 모델들 또한 동일한 valid set으로 평가할 거기 때문에 공평함. 괜춘


# x, y 나누기
# train_df.columns.str.contains('Neighborhood_')  # 나중에 더미변수로 바꾼 Neighborhood_ 로 시작하는 컬럼을 불러올 수 있음.
# regex (Regular Expression, 정규방정식)
train_x = train.drop('SalePrice', axis=1)
train_y = train['SalePrice']

valid_x = valid.drop('SalePrice', axis=1)
valid_y = valid['SalePrice']

len(train.columns)
len(train_x.columns)