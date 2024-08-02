
import pandas as pd

# 데이터 불러오기
train_df = pd.read_csv('./house price/train.csv')
test_df = pd.read_csv('./house price/test.csv')
submission = pd.read_csv('./house price/sample_submission.csv')

#sample_submission['SalePrice'] = price_mean
#sample_submission.head()

#sample_submission.to_csv('house price/sample_submission.csv' , index = False)


price_mean = df['SalePrice'].mean()


# groupby
group_train = train_df.groupby('YearBuilt',as_index=False).agg(price_mean=('SalePrice','mean'))
test = test_df[['Id','YearBuilt']]
test = pd.merge(test, group_train, how='left', on='YearBuilt')

null_id = test[test['price_mean'].isna()==True]['Id']
group_train['YearBuilt'].value_counts().sort_index()

for i in null_id:
    test.loc[test['Id']==i,'price_mean'] = int(test[test['YearBuilt']>int(test[test['Id']==i]['YearBuilt'])].sort_values('YearBuilt').iloc[0,:]['price_mean'])

# test['SalePrice'] = test['SalePrice'].fillna()  # 결측치 대체하기
 
test[test['price_mean'].isna()==True]  # 결측치 없다고 나옴


submission_merge = pd.merge(sample_submission, test, how='left', on='Id')[['Id','price_mean']]
submission_merge = submission_merge.rename(columns = {'price_mean' : 'SalePrice'})
submission_merge.info()


submission_merge.to_csv('sample_submission.csv', index=False)


# groupby2
import numpy as np
train_df['MSSubClass'].unique() != test_df['MSSubClass'].unique()


train_df.columns
train_df.info()
train_df['MSSubClass']=train_df['MSSubClass'].astype('object')

group_train = train_df.groupby('MSSubClass').agg(max_price = ('SalePrice', 'max')
                                 ,min_price = ('SalePrice', 'min')
                                 , mean_price = ('SalePrice', 'mean')).sort_values('max_price')
                                 
                                 
                                              
group_train = train_df.groupby(['YearBuilt','LandContour'],as_index=False).agg(price_mean=('SalePrice','mean'))
test = test_df[['Id','YearBuilt','LandContour']]
test = pd.merge(test, group_train, how='left', on=['YearBuilt','LandContour'])


test[test['price_mean'].isna()==True]
null_id = test[test['price_mean'].isna()==True]['Id']
group_train['YearBuilt'].value_counts().sort_index()

group_train2 = train_df.groupby(['LandContour'],as_index=False).agg(price_mean=('SalePrice','mean'))

for i in null_id:
    test.loc[test['Id']==i,'price_mean'] = int(test[test['YearBuilt']>int(test[test['Id']==i]['YearBuilt'])].sort_values('YearBuilt').iloc[0,:]['price_mean'])

test2 = pd.merge(test, group_train2, how='left', on='LandContour')
test2[test2['price_mean_y'].isna()==True]


submission_merge = pd.merge(sample_submission, test2, how='left', on='Id')[['Id','price_mean_y']]
submission_merge = submission_merge.rename(columns = {'price_mean_y' : 'SalePrice'})
submission_merge.info()


submission_merge.to_csv('sample_submission.csv', index=False)





# groupby3
                        
group_train = train_df.groupby(['YearBuilt','OverallQual'],as_index=False).agg(price_mean=('SalePrice','mean'))
test = test_df[['Id','YearBuilt','OverallQual']]
test = pd.merge(test, group_train, how='left', on=['YearBuilt','OverallQual'])


len(test[test['price_mean'].isna()==True])
test[test['price_mean'].isna()==True]['OverallQual'].value_counts()
test[test['price_mean'].isna()==True].groupby(['OverallQual', 'YearBuilt']).agg(value_count = ('YearBuilt', 'count'))

 #----------------------------------
test.groupby('OverallQual','Y).agg(year=('YearBuilt','min'))


null_id = test[test['price_mean'].isna()==True]['Id']
group_train['YearBuilt'].value_counts().sort_index()

group_train2 = train_df.groupby(['LotConfig'],as_index=False).agg(price_mean=('SalePrice','mean'))

for i in null_id:
    test.loc[test['Id']==i,'price_mean'] = int(test[test['YearBuilt']>int(test[test['Id']==i]['YearBuilt'])].sort_values('YearBuilt').iloc[0,:]['price_mean'])

test2 = pd.merge(test, group_train2, how='left', on='LandContour')
test2[test2['price_mean_y'].isna()==True]


submission_merge = pd.merge(sample_submission, test2, how='left', on='Id')[['Id','price_mean_y']]
submission_merge = submission_merge.rename(columns = {'price_mean_y' : 'SalePrice'})
submission_merge.info()


submission_merge.to_csv('sample_submission.csv', index=False)




# ----------------------------------------------------
# 스타일
# OverallQual: Rates the overall material and finish of the house
# 전체적인 자재 및 마감 품질. 주택의 전반적인 품질을 평가합니다.
#       10	Very Excellent
#       9	Excellent
#       8	Very Good
#       7	Good
#       6	Above Average
#       5	Average
#       4	Below Average
#       3	Fair
#       2	Poor
#       1	Very Poor

# ExterQual: Evaluates the quality of the material on the exterior  => 너무 당연한 결과가 나와서 패스
# 외부 재료 품질. 외부 재료의 품질을 평가합니다.
#       Ex	Excellent
#       Gd	Good
#       TA	Average/Typical
#       Fa	Fair
#       Po	Poor

# GarageType: Garage location
# 차고 위치		
#       2Types	More than one type of garage
#       Attchd	Attached to home
#       Basment	Basement Garage
#       BuiltIn	Built-In (Garage part of house - typically has room above garage)
#       CarPort	Car Port
#       Detchd	Detached from home
#       NA	No Garage



# 전반적인 품질별로 가장 많이 나오는 외부재료 품질 확인
df = train_df[['OverallQual','ExterQual']]
df.isna().sum()  
    
group_df = df.groupby(['OverallQual','ExterQual'], as_index=False).agg(exterqual_count=('ExterQual','count')).sort_values(['OverallQual','exterqual_count'], ascending=[False,False])
group_df2 = group_df.groupby('OverallQual', as_index=False).agg(exterqual_max_count = ('exterqual_count','max'))
group_df = group_df.rename(columns ={'exterqual_count':'exterqual_max_count'})
df2 = pd.merge(group_df2, group_df, how='left', on=['exterqual_max_count','OverallQual']).sort_values('OverallQual', ascending=False)
df2

import matplotlib.pyplot as plt
import seaborn as sns
plt.clf()
sns.barplot(data=df2, x='OverallQual', y='exterqual_max_count', hue='ExterQual')
plt.tight_layout()
plt.show()




# 전반적인 품질별로 가장 많이 나오는 차고 위치 확인
df = train_df[['OverallQual','GarageType']]
df.isna().sum()  # 'GarageType' 결측치 존재함
df = df.dropna()
df.isna().sum()  # 'GarageType' 결측치 제거 확인
    
group_df = df.groupby(['OverallQual','GarageType'], as_index=False).agg(garagetype_count=('GarageType','count')).sort_values(['OverallQual','garagetype_count'], ascending=[False,False])
group_df2 = group_df.groupby('OverallQual', as_index=False).agg(garagetype_max_count = ('garagetype_count','max'))
group_df = group_df.rename(columns ={'garagetype_count':'garagetype_max_count'})
df2 = pd.merge(group_df2, group_df, how='left', on=['garagetype_max_count','OverallQual']).sort_values('OverallQual', ascending=False)
df2

import matplotlib.pyplot as plt
import seaborn as sns
plt.clf()
sns.barplot(data=df2, x='OverallQual', y='garagetype_max_count', hue='GarageType')
plt.tight_layout()
plt.show()



# 전반적인 품질별로 가장 많이 나오는 차고 위치 확인 + 비율 값 확인
df = train_df[['OverallQual','GarageType']]
df.isna().sum()  # 'GarageType' 결측치 존재함
df = df.dropna()
df.isna().sum()  # 'GarageType' 결측치 제거 확인
    
group_df = df.groupby(['OverallQual','GarageType'], as_index=False).agg(garagetype_count=('GarageType','count')).sort_values(['OverallQual','garagetype_count'], ascending=[False,False])
overall_cnt = df.groupby('OverallQual', as_index=False).agg(overallqual_count=('OverallQual','count'))
df2 = pd.merge(group_df, overall_cnt, how='left', on='OverallQual')
df2['pct'] = round(df2['garagetype_count']/df2['overallqual_count']*100,1)
df2
group_df2 = df2.groupby('OverallQual', as_index=False).agg(garagetype_max_pct = ('pct','max'))
df3 = df2[['pct','GarageType','OverallQual']]
df3 = df3.rename(columns ={'pct':'garagetype_max_pct'})
df4 = pd.merge(group_df2, df3, how='left', on=['garagetype_max_pct','OverallQual']).sort_values('OverallQual', ascending=False)
df4

import matplotlib.pyplot as plt
import seaborn as sns
plt.clf()
sns.barplot(data=df4, x='OverallQual', y='garagetype_max_pct', hue='GarageType')
plt.tight_layout()
plt.show()




### 막대그래프, 범주별 점 그래프로 데이터 퍼져있는 상태 알아보기
Neighborhood_mean = train_df.groupby('Neighborhood', as_index = False) \
                     .agg(N_price_mean = ('SalePrice', 'mean'))

plt.clf()
# plt.grid()
sns.barplot(data = Neighborhood_mean, y = 'Neighborhood', x = 'N_price_mean', hue = 'Neighborhood')

# LandContour_scatter4 = train[["Neighborhood", "SalePrice"]]
plt.scatter(data = train_df , y="Neighborhood", x= "SalePrice", s = 1, color = 'red')

plt.xlabel("houseprice", fontsize=10)
plt.ylabel("neighborhood", fontsize=10)
#plt.yticks(rotation=45,fontsize=8)
plt.tight_layout()
plt.show()





# 8/1 수업
# 점을 잘 예측할 것 같은 직선을 만든 뒤, 예측하고 submit

# 곡선인 경우
a = 5
b = 100
x = np.linspace(0,5, 100)
y = np.exp(x) * a + b

# 직선인 경우
a1 = 80
b1 = -30
y = a1 * x + b1


my_df = train_df[['BedroomAbvGr','SalePrice']].head(10)
my_df['SalePrice'] = my_df['SalePrice']/1000
plt.clf()
plt.scatter(x=my_df['BedroomAbvGr'], y=my_df['SalePrice'], color='orange')
plt.plot(x , y)
plt.ylim([-1,400])
plt.show()


x = test_df[['BedroomAbvGr']]
y = a1 * x + b1
y1 = 1000*y

sample_submission['SalePrice'] = y1

sample_submission.to_csv('./house price/sample_submission_line_240801.csv', index=False)




# 직선 성능 평가
a1 = 80
b1 = -30
train_y_hat = (a1 * train_df['BedroomAbvGr'] + b1) *1000

sum(abs( train_df['SalePrice'] - train_y_hat ))
sum(( train_df['SalePrice'] - train_y_hat )**2)
 


# 회귀분석 직선 구하기
# !pip install scikit-learn

import numpy as np
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

# 예시 데이터 (x와 y 벡터)
x = np.array([1, 3, 2, 1, 5]).reshape(-1, 1)  # x 벡터 (특성 벡터는 2차원 배열이어야 합니다)
y = np.array([1, 2, 3, 4, 5])  # y 벡터 (레이블 벡터는 1차원 배열입니다)

# 선형 회귀 모델 생성
model = LinearRegression()

# 모델 학습
model.fit(x, y)  # fit() : 자동으로 기울기, 절편 값을 구해줌.

# 회귀 직선의 기울기와 절편
model.coef_  # 기울기 a
model.intercept_  # 절편 b


slope = model.coef_[0]    # 기울기가 여러개 값이 생길 수 있어서 인덱스 0만 출력해봄.
intercept = model.intercept_
print(f"기울기 (slope): {slope}")
print(f"절편 (intercept): {intercept}")

# 예측값 계산
y_pred = model.predict(x)   # model.predict() : 만든 직선에 x값 넣는 함수 , y값 반환해줌.

# 데이터와 회귀 직선 시각화
plt.clf()
plt.scatter(x, y, color='blue', label='실제 데이터')
plt.plot(x, y_pred, color='red', label='회귀 직선')
plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.show()

# 토론하기

x= np.array(train_df['BedroomAbvGr']).reshape(-1,1)
y = train_df['SalePrice']
model = LinearRegression()
model.fit(x,y)
y_pred = model.predict(x)

plt.clf()
plt.scatter(x,y,color='blue', label='실제 데이터')
plt.plot(x, y_pred, color='red', label='적합시킨 회귀 직선')
plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.show()


x_test = np.array(test_df['BedroomAbvGr']).reshape(-1,1)
y_test_pred = model.predict(x_test)

# 의미없는 시각화지만, 그려보기
plt.clf()
plt.scatter(x_test, y_test_pred, color='blue', label='예측한 데이터')
plt.plot(x_test, y_test_pred, color='red', label='적합시킨 회귀 직선')
plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.rcParams('font.family')= 'Malgun Gothic'
plt.show()



# submission 피일에 넣기
submission['SalePrice'] = y_test_pred
submission.to_csv('./house price/sample_submission_line_test_240828.csv', index=False)





from scipy.optimize import minimize

# 최소값을 찾을 다변수 함수 정의
def line_perform(par):
    y_hat = (par[0]* train_df['BedroomAbvGr']+ par[1])
    y=train_df['SalePrice']
    return np.mean((y-y_hat)**2)





# 다른 변수들로 회귀분석
train_df.columns
df = train_df[['ExterQual','Fireplaces','OverallQual','SalePrice']]
x1 = np.array(df['OverallQual']).reshape(-1,1)
y = df['SalePrice']
model = LinearRegression()
model.fit(x1,y)
y_train_pred = model.predict(x1)

plt.clf()
plt.scatter(x1, y, color='blue')
plt.plot(x1, y_train_pred , color='red')
plt.show()


print("MSE :",np.mean((df['SalePrice'] - y_train_pred)**2))
print("MAE",np.mean(abs(df['SalePrice'] - y_train_pred)))


x_test = np.array(test_df['OverallQual']).reshape(-1,1)
y_test_pred = model.predict(x_test)

submission['SalePrice'] = y_test_pred
submission.to_csv('./house price/sample_submission_line3_240802.csv', index=False)



# 여러 변수 (안됨 버려)
df = train_df[['ExterQual','Fireplaces','OverallQual','SalePrice']]
x = df[['ExterQual','Fireplaces','OverallQual']]
y = df['SalePrice']
model = LinearRegression()
model.fit(x,y)
y_train_pred = model.predict(x)

plt.clf()
plt.scatter(x1, y, color='blue')
plt.plot(x1, y_train_pred , color='red')
plt.show()


# TotalBsmtSF
train_df.columns
df = train_df[['TotalBsmtSF','SalePrice']].dropna()
x1 = np.array(df['TotalBsmtSF']).reshape(-1,1)
y = df['SalePrice']
model = LinearRegression()
model.fit(x1,y)
y_train_pred = model.predict(x1)

plt.clf()
plt.scatter(x1, y, color='blue')
plt.plot(x1, y_train_pred , color='red')
plt.show()


print("MSE :",np.mean((df['SalePrice'] - y_train_pred)**2))
print("MAE",np.mean(abs(df['SalePrice'] - y_train_pred)))


#nonna_test = np.array(test_df['TotalBsmtSF'].dropna()).reshape(-1,1)
#y_test_pred1 = model.predict(nonna_test)
#mean_y = np.mean(y_test_pred1)

mean_x = np.mean(train_df['TotalBsmtSF'])
test_df['TotalBsmtSF'].isna().sum()
test_df['TotalBsmtSF'] = test_df['TotalBsmtSF'].fillna(mean_x)

test_df['TotalBsmtSF'].isna().sum()

test_df.loc[test_df['TotalBsmtSF'].isna()==True,'TotalBsmtSF']
x_test = np.array(test_df['TotalBsmtSF']).reshape(-1,1)

y_test_pred = model.predict(x_test)


submission['SalePrice'] = y_test_pred
submission.to_csv('./house price/sample_submission_line5_240802.csv', index=False)




# 1stFlrSF
df = train_df[['1stFlrSF','SalePrice']].dropna()
x1 = np.array(df['1stFlrSF']).reshape(-1,1)
y = df['SalePrice']
model = LinearRegression()
model.fit(x1,y)
y_train_pred = model.predict(x1)

plt.clf()
plt.scatter(x1, y, color='blue')
plt.plot(x1, y_train_pred , color='red')
plt.show()


print("MSE :",np.mean((df['SalePrice'] - y_train_pred)**2))
print("MAE",np.mean(abs(df['SalePrice'] - y_train_pred)))


mean_x = np.mean(train_df['1stFlrSF'])
test_df['1stFlrSF'].isna().sum()
test_df['1stFlrSF'] = test_df['1stFlrSF'].fillna(mean_x)


test_df.loc[test_df['1stFlrSF'].isna()==True,'1stFlrSF']
x_test = np.array(test_df['1stFlrSF']).reshape(-1,1)

y_test_pred = model.predict(x_test)


submission['SalePrice'] = y_test_pred
submission.to_csv('./house price/sample_submission_line6_240802.csv', index=False





# YearBuilt
df = train_df[['YearBuilt','SalePrice']].dropna()
x1 = np.array(df['YearBuilt']).reshape(-1,1)
y = df['SalePrice']
model = LinearRegression()
model.fit(x1,y)
y_train_pred = model.predict(x1)

plt.clf()
plt.scatter(x1, y, color='blue')
plt.plot(x1, y_train_pred , color='red')
plt.show()


print("MSE :",np.mean((df['SalePrice'] - y_train_pred)**2))
print("MAE",np.mean(abs(df['SalePrice'] - y_train_pred)))


mean_x = np.mean(train_df['YearBuilt'])
test_df['YearBuilt'].isna().sum()
# test_df['YearBuilt'] = test_df['1stFlrSF'].fillna(mean_x)
#test_df.loc[test_df['YearBuilt'].isna()==True,'YearBuilt']

x_test = np.array(test_df['YearBuilt']).reshape(-1,1)

y_test_pred = model.predict(x_test)


submission['SalePrice'] = y_test_pred
submission.to_csv('./house price/sample_submission_line7_240802.csv', index=False)



# TotRmsAbvGrd
df = train_df[['TotRmsAbvGrd','SalePrice']].dropna()
x1 = np.array(df['TotRmsAbvGrd']).reshape(-1,1)
y = df['SalePrice']
model = LinearRegression()
model.fit(x1,y)
y_train_pred = model.predict(x1)

plt.clf()
plt.scatter(x1, y, color='blue')
plt.plot(x1, y_train_pred , color='red')
plt.show()


print("MSE :",np.mean((df['SalePrice'] - y_train_pred)**2))
print("MAE",np.mean(abs(df['SalePrice'] - y_train_pred)))


test_df['TotRmsAbvGrd'].isna().sum()
# mean_x = np.mean(train_df['TotRmsAbvGrd'])
# test_df['YearBuilt'] = test_df['1stFlrSF'].fillna(mean_x)
#test_df.loc[test_df['YearBuilt'].isna()==True,'YearBuilt']

x_test = np.array(test_df['TotRmsAbvGrd']).reshape(-1,1)

y_test_pred = model.predict(x_test)


submission['SalePrice'] = y_test_pred
submission.to_csv('./house price/sample_submission_line8_240802.csv', index=False)




# LotFrontage
df = train_df[['LotFrontage','SalePrice']].dropna()
x1 = df[['LotFrontage']]
y = df['SalePrice']
model = LinearRegression()
model.fit(x1,y)
y_train_pred = model.predict(x1)

plt.clf()
plt.scatter(x1, y, color='blue')
plt.plot(x1, y_train_pred , color='red')
plt.show()


print("MSE :",np.mean((df['SalePrice'] - y_train_pred)**2))
print("MAE",np.mean(abs(df['SalePrice'] - y_train_pred)))


test_df['LotFrontage'].isna().sum()
 mean_x = np.mean(train_df['LotFrontage'])
 test_df['LotFrontage'] = test_df['LotFrontage'].fillna(mean_x)
test_df.loc[test_df['LotFrontage'].isna()==True,'LotFrontage']

x_test = np.array(test_df['LotFrontage']).reshape(-1,1)

y_test_pred = model.predict(x_test)


submission['SalePrice'] = y_test_pred
submission.to_csv('./house price/sample_submission_line9_240802.csv', index=False)



# 이상치 확인
x1 = np.array(train_df['GrLivArea']).reshape(-1,1)
y = train_df['SalePrice']
model = LinearRegression()
model.fit(x1,y)
y_train_pred = model.predict(x1)

plt.clf()
plt.scatter(x1, y, color='blue')
plt.plot(x1, y_train_pred , color='red')
plt.xlim([0,6000])
plt.show()

print("MSE :",np.mean((train_df['SalePrice'] - y_train_pred)**2))
print("MAE",np.mean(abs(train_df['SalePrice'] - y_train_pred)))




train_df['GrLivArea'].value_counts().sort_index()
train_df.query('GrLivArea>=4500')[['GrLivArea','SalePrice']]
df = train_df.query('GrLivArea < 4500')

x1 = np.array(df['GrLivArea']).reshape(-1,1)
y = df['SalePrice']
model = LinearRegression()
model.fit(x1,y)
y_train_pred = model.predict(x1)

plt.clf()
plt.scatter(x1, y, color='blue')
plt.plot(x1, y_train_pred , color='red')
plt.xlim([0,6000])
plt.show()


print("MSE :",np.mean((df['SalePrice'] - y_train_pred)**2))
print("MAE",np.mean(abs(df['SalePrice'] - y_train_pred)))


test_df['GrLivArea'].isna().sum()
# mean_x = np.mean(train_df['LotFrontage'])
# test_df['LotFrontage'] = test_df['LotFrontage'].fillna(mean_x)
#test_df.loc[test_df['LotFrontage'].isna()==True,'LotFrontage']

x_test = np.array(test_df['GrLivArea']).reshape(-1,1)

y_test_pred = model.predict(x_test)


submission['SalePrice'] = y_test_pred
submission.to_csv('./house price/sample_submission_line10_240802.csv', index=False)


# 변수 여러개
x = np.array(df[['GrLivArea','GarageArea']]).reshape(-1,2)
x = df[['GrLivArea','GarageArea']]
y = df['SalePrice']
model = LinearRegression()
model.fit(x,y)
y_train_pred = model.predict(x)

model.coef_
model.intercept_



def f(x,y):
    return model.coef_[0]*x + model.coef_[1]*y + model.intercept_

f(df['GrLivArea'], df['GarageArea'])


f(test_df['GrLivArea'],test_df['GarageArea'])
f(test_df['GrLivArea'],test_df['GarageArea']).isna().sum()


test_df['nonna_test'] = test_df[['GarageArea']].fillna(test_df['GarageArea'].mean())
test_df['nonna_test'].isna().sum()
f(test_df['GrLivArea'],test_df['nonna_test'])
f(test_df['GrLivArea'],test_df['nonna_test']).isna().sum()



