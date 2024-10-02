# house price
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

# 데이터 불러오기
train_df = pd.read_csv('./house price/train.csv')
train_loc_df = pd.read_csv('./house price/houseprice-with-lonlat.csv')
test_df = pd.read_csv('./house price/test.csv')
submission = pd.read_csv('./house price/sample_submission.csv')

#sample_submission['SalePrice'] = price_mean
#sample_submission.head()

#sample_submission.to_csv('house price/sample_submission.csv' , index = False)




# 컬럼 알아보기
cols = train_df.columns
train_df[train_df['MSSubClass'] == 20]['YearBuilt'].sort_values()  # 'MSSubClass'=20은 1938, 1946~2010에 지어진 집임
train_df[train_df['MSSubClass'] == 30]['YearBuilt'].sort_values()  # 'MSSubClass'=30은 1885, 1910~1945, 1948에 지어진 집임










# 중복 집 알아보기
train_loc_df[['Longitude','Latitude']].value_counts().sort_values()

train_loc_df[['Longitude','Latitude']]











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
x = np.array(train_df[['GrLivArea','GarageArea']]).reshape(-1,2)
x = train_df[['GrLivArea','GarageArea']]
y = train_df['SalePrice']
model = LinearRegression()
model.fit(x,y)
y_train_pred = model.predict(x)

model.coef_
model.intercept_
for i in range(len(model.coef_)):
    print("베타",i+1,"_hat :", model.coef_[i])


def f(x,y):
    return model.coef_[0]*x + model.coef_[1]*y + model.intercept_

f(df['GrLivArea'], df['GarageArea'])


f(test_df['GrLivArea'],test_df['GarageArea'])
f(test_df['GrLivArea'],test_df['GarageArea']).isna().sum()


test_df['nonna_test'] = test_df[['GarageArea']].fillna(test_df['GarageArea'].mean())
test_df['nonna_test'].isna().sum()
f(test_df['GrLivArea'],test_df['nonna_test'])
f(test_df['GrLivArea'],test_df['nonna_test']).isna().sum()



# 변수 2개 다른걸로 해보기
train_x = train_df[['GrLivArea', 'OverallQual']]
train_y = train_df['SalePrice']

model = LinearRegression()
model.fit(x,y)
train_y_pred = model.predict(train_x)

test_x = test_df[['GrLivArea', 'OverallQual']]
test_x.isna().sum()
test_y_pred = model.predict(test_x)

submission['SalePrice'] = test_y_pred
submission.to_csv('./house price/sample_submission_line11.csv', index=False)


# 변수 2개 다른걸로 해보기 (하다 맒)
from sklearn.linear_model import LinearRegression
train_x = train_df[['GrLivArea', 'GarageArea']]
train_y = train_df['SalePrice']

model = LinearRegression()
model.fit(train_x,train_y)
train_y_pred = model.predict(train_x)

test_x = test_df[['GrLivArea', 'GarageArea']]
test_x.isna().sum()
test_y_pred = model.predict(test_x)

submission['SalePrice'] = test_y_pred
submission.to_csv('./house price/sample_submission_line12.csv', index=False)



# 모든 수치컬럼을 가지고 linearregression 해보기
train_df.info()
numeric_df = train_df.select_dtypes(include = [int, float])
numeric_df.info()
train_x = numeric_df.iloc[:,1:-1]  # ID와 SalePrice 컬럼 제거함
train_y = train_df['SalePrice']

train_x.isna().sum()  # LotFrontage, MasVnrArea, GarageYrBlt 결측치 대체하기

train_x['LotFrontage'] = train_x['LotFrontage'].fillna(train_x['LotFrontage'].mean())
train_x['MasVnrArea'] = train_x['MasVnrArea'].fillna(train_x['MasVnrArea'].mean())
train_x['GarageYrBlt'] = train_x['GarageYrBlt'].fillna(train_x['GarageYrBlt'].mean())

train_x.isna().sum()


model = LinearRegression()
model.fit(train_x, train_y)
train_y_pred = model.predict(train_x)

print("MSE :", sum((train_y_pred - train_y)**2))

numeric_test_x = test_df.select_dtypes(include = [int,float])
numeric_test_x = numeric_test_x.iloc[:,1:]

numeric_test_x.isna().sum()  # LotFrontage, MasVnrArea, BsmtFinSF1, BsmtFinSF2, BsmtUnfSF, TotalBsmtSF, BsmtFullBath, BsmtHalfBath, GarageYrBlt, GarageCars, GarageArea
numeric_test_x['LotFrontage'] = numeric_test_x['LotFrontage'].fillna(numeric_test_x['LotFrontage'].mean())
numeric_test_x['MasVnrArea'] = numeric_test_x['MasVnrArea'].fillna(numeric_test_x['MasVnrArea'].mean())
numeric_test_x['BsmtFinSF1'] = numeric_test_x['BsmtFinSF1'].fillna(numeric_test_x['BsmtFinSF1'].mean())
numeric_test_x['BsmtFinSF2'] = numeric_test_x['BsmtFinSF2'].fillna(numeric_test_x['BsmtFinSF2'].mean())
numeric_test_x['BsmtUnfSF'] = numeric_test_x['BsmtUnfSF'].fillna(numeric_test_x['BsmtUnfSF'].mean())
numeric_test_x['TotalBsmtSF'] = numeric_test_x['TotalBsmtSF'].fillna(numeric_test_x['TotalBsmtSF'].mean())
numeric_test_x['BsmtFullBath'] = numeric_test_x['BsmtFullBath'].fillna(numeric_test_x['BsmtFullBath'].mean())
numeric_test_x['BsmtHalfBath'] = numeric_test_x['BsmtHalfBath'].fillna(numeric_test_x['BsmtHalfBath'].mean())
numeric_test_x['GarageYrBlt'] = numeric_test_x['GarageYrBlt'].fillna(numeric_test_x['GarageYrBlt'].mean())
numeric_test_x['GarageCars'] = numeric_test_x['GarageCars'].fillna(numeric_test_x['GarageCars'].mean())
numeric_test_x['GarageArea'] = numeric_test_x['GarageArea'].fillna(numeric_test_x['GarageArea'].mean())

numeric_test_x.isna().sum()


test_y_pred = model.predict(numeric_test_x)
submission['SalePrice'] = test_y_pred
submission.to_csv('./house price/sample_submission_line_numeric_fillnamean_240805.csv', index=False)




# 다른 방식으로 결측치 대체
merge_train = train_df.copy()

# LotFrontage 결측치 대체
merge_train.loc[merge_train['LotFrontage'].isna()==True, ['LotFrontage', 'Street']]  
group1 = train_df.groupby('Street', as_index=False).agg(lotfrontage_mean = ('LotFrontage', 'mean'))
merge_train['LotFrontage'] = np.where((merge_train['LotFrontage'].isna() == True) & (merge_train['Street']=='Grvl'), 85.400000,
                                np.where((merge_train['LotFrontage'].isna() == True) & (merge_train['Street']=='Pave'), 69.985786, merge_train['LotFrontage']))
merge_train['LotFrontage'].isna().sum()  # 결측값 0개됨.


# MasVnrArea 결측치 대체
merge_train.loc[merge_train['MasVnrArea'].isna() == True, ['MasVnrArea', 'MasVnrType']]  # 'MasVnrArea', 'MasVnrType' 둘다 동일하게 결측치가 존재하기 때문에 MasVnrType 범주 기준으로 결측치 대체가 어려움. Exterior1st 범주 기준으로 결측치 대체하기로 함.

group2 = train_df.groupby('Exterior1st', as_index=False).agg(MasVnrArea_mean = ('MasVnrArea','mean'))
merge_train['MasVnrArea'].isna().sum() # 결측치가 8개 있음
merge_train.loc[merge_train['MasVnrArea'].isna() == True, :]
merge_train['MasVnrArea'] =  np.where((merge_train['MasVnrArea'].isna()==True) & (merge_train['Exterior1st'] == 'VinylSd'), 136.300000,
                             np.where((merge_train['MasVnrArea'].isna()==True) & (merge_train['Exterior1st'] == 'Wd Sdng'), 43.375610, 
                             np.where((merge_train['MasVnrArea'].isna()==True) & (merge_train['Exterior1st'] == 'CemntBd'), 144.440678, merge_train['MasVnrArea'])))

merge_train['MasVnrArea'].isna().sum()  # 결측치가 0개가 됨.



# GarageYrBlt 결측치 대체 (하던거)
merge_train.loc[merge_train['GarageYrBlt'].isna() == True, ['GarageYrBlt', 'GarageType','GarageFinish','GarageQual']]  # 'GarageYrBlt'는 garage 관련 변수들은 모두 결측치이기 때문에 관련 컬럼의 범주 기준으로 결측치 대체가 어려움. 전체 평균으로 대체 
merge_train['GarageYrBlt'].isna().sum()  # 결측치가 81개있음.
merge_train['GarageYrBlt'] = merge_train['GarageYrBlt'].fillna(merge_train['GarageYrBlt'].mean())
merge_train['GarageYrBlt'].isna().sum() # 결측치가 0개 됨.



numeric_df = merge_train.select_dtypes(include = [int, float])
numeric_df.info()
train_x = numeric_df.iloc[:,1:-1]  # ID와 SalePrice 컬럼 제거함
train_y = train_df['SalePrice']

train_x.isna().sum()  # 결측치가 0개임.


model = LinearRegression()
model.fit(train_x, train_y)
train_y_pred = model.predict(train_x)

print("MSE :", sum((train_y_pred-train_y)**2))




merge_test = test_df.copy()
merge_test['LotFrontage'].isna().sum() # 결측치 227개

group4 = merge_test.groupby('Street', as_index=False).agg(lotfrontage_mean = ('LotFrontage', 'mean'))
merge_test['LotFrontage'] = np.where((merge_test['LotFrontage'].isna() == True) & (merge_test['Street']=='Grvl'), 91.000000,
                                np.where((merge_test['LotFrontage'].isna() == True) & (merge_test['Street']=='Pave'), 68.488998, merge_test['LotFrontage']))
merge_test['LotFrontage'].isna().sum()  # 결측값 0개됨.



group5 = test_df.groupby('Exterior1st', as_index=False).agg(MasVnrArea_mean = ('MasVnrArea','mean'))
merge_test['MasVnrArea'].isna().sum() # 결측치가 15개 있음
merge_test.loc[merge_test['MasVnrArea'].isna() == True, 'Exterior1st']
merge_test['MasVnrArea'] =  np.where((merge_test['MasVnrArea'].isna()==True) & (merge_test['Exterior1st'] == 'VinylSd'), 124.637827,
                             np.where((merge_test['MasVnrArea'].isna()==True) & (merge_test['Exterior1st'] == 'WdShing'), 38.344828, 
                             np.where((merge_test['MasVnrArea'].isna()==True) & (merge_test['Exterior1st'] == 'CemntBd'), 158.265625, merge_test['MasVnrArea'])))

merge_test['MasVnrArea'].isna().sum()  # 결측치가 0개가 됨.




merge_test.loc[merge_test['GarageYrBlt'].isna() == True, ['GarageYrBlt', 'GarageType','GarageFinish','GarageQual']]  # 'GarageYrBlt'는 garage 관련 변수들은 모두 결측치이기 때문에 관련 컬럼의 범주 기준으로 결측치 대체가 어려움. 전체 평균으로 대체 
merge_test['GarageYrBlt'].isna().sum()  # 결측치가 78개있음.
merge_test['GarageYrBlt'] = merge_test['GarageYrBlt'].fillna(merge_test['GarageYrBlt'].mean())
merge_test['GarageYrBlt'].isna().sum() # 결측치가 0개 됨.



numeric_test_x = merge_test.select_dtypes(include = [int,float])
numeric_test_x.columns
numeric_test_x = numeric_test_x.iloc[:,1:]

numeric_test_x.isna().sum()  # BsmtFinSF1, BsmtFinSF2, BsmtUnfSF, TotalBsmtSF, BsmtFullBath, BsmtHalfBath, GarageCars, GarageArea
numeric_test_x['BsmtFinSF1'] = numeric_test_x['BsmtFinSF1'].fillna(numeric_test_x['BsmtFinSF1'].mean())
numeric_test_x['BsmtFinSF2'] = numeric_test_x['BsmtFinSF2'].fillna(numeric_test_x['BsmtFinSF2'].mean())
numeric_test_x['BsmtUnfSF'] = numeric_test_x['BsmtUnfSF'].fillna(numeric_test_x['BsmtUnfSF'].mean())
numeric_test_x['TotalBsmtSF'] = numeric_test_x['TotalBsmtSF'].fillna(numeric_test_x['TotalBsmtSF'].mean())
numeric_test_x['BsmtFullBath'] = numeric_test_x['BsmtFullBath'].fillna(numeric_test_x['BsmtFullBath'].mean())
numeric_test_x['BsmtHalfBath'] = numeric_test_x['BsmtHalfBath'].fillna(numeric_test_x['BsmtHalfBath'].mean())
numeric_test_x['GarageCars'] = numeric_test_x['GarageCars'].fillna(numeric_test_x['GarageCars'].mean())
numeric_test_x['GarageArea'] = numeric_test_x['GarageArea'].fillna(numeric_test_x['GarageArea'].mean())

numeric_test_x.isna().sum()

test_y_pred = model.predict(numeric_test_x)

submission['SalePrice'] = test_y_pred
submission.to_csv('./house price/sample_submission_line_fillna2_240805.csv', index=False)



# 실습

center_x = train_loc_df['Longitude'].median()
center_y = train_loc_df['Latitude'].median()

# 흰도화지 map 불러오기
map_sig = folium.Map(location = [center_y, center_x], zoom_start=12, tiles="cartodbpositron")
map_sig.save("./house price/map_seoul.html")  # 돌리면 프로젝트 폴더에 html 파일 생김. 들어가면 지도 나옴.


# 코로플릿 <- 구 경계선 그리기
folium.Choropleth(
    geo_data=geo,
    data=df_seoulpop,
    columns = ("code", "pop"),
    key_on = "feature.properties.SIG_CD").add_to(map_sig)

map_sig.save("map_seoul.html")

bins = list(train_loc_df['SalePrice'].quantile([0, 0.2, 0.4, 0.6, 0.8, 1]))  # 하위 0, 하위 0.2, 하위 0.4, 하위 0.6 ,... 에 해당하는 값을 반환해줌.
bins

folium.Choropleth(
    geo_data=geo,
    data=df_seoulpop,
    columns = ("code", "pop"),
    key_on = "feature.properties.SIG_CD",
    bins = bins).add_to(map_sig)

map_sig.save("map_seoul.html")



folium.Choropleth(
    geo_data=geo,
    data=df_seoulpop,
    columns = ("code", "pop"),
    key_on = "feature.properties.SIG_CD",
    fill_color = 'viridis',
    bins = bins).add_to(map_sig)

map_sig.save("map_seoul.html")


# 점 찍는 법

# 방법1    
for _, row in train_loc_df.iterrows():
    folium.Marker(location=[row['Latitude'], row['Longitude']] ).add_to(map_sig)
map_sig.save("./house price/map_seoul.html")

# 방법2
for i in range(len(train_loc_df)):
    folium.Marker(location=[float(train_loc_df['Latitude'][i]), float(train_loc_df['Longitude'][i])]).add_to(map_sig)
map_sig.save("./house price/map_seoul.html")    

# 방법3
for lat, lon in zip(train_loc_df['Latitude'])



def df_seoul(num):
    gu_name=geo['features'][num]['properties']['SIG_KOR_NM']
    coordinate_array = np.array(geo['features'][num]['geometry']['coordinates'][0][0])
    x = coordinate_array[:,0]
    y = coordinate_array[:,1]
    return pd.DataFrame({'gu_name' : gu_name, 'x' : x, 'y': y})
    
df_seoul(12)


from folim.plugins import MarkerCluster


marker_cluster = MarkerCluster().add_to(map_sig)

for i in range(len())



# 0808 수업 : 인터랙티브 시각화
#!pip install plotly
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd

df_covid19_100 = pd.read_csv('./data/df_covid19_100.csv')
df_covid19_100.info()


#속성은 data, layout, frame 3개있음

fig = go.Figure(  # 그림을 그릴 건데 () 안에 data 속성을 입력해줄거임. 근데 data 속성(dict :트레이스)을 2개 입력해줄 거임. 그럼 그래프 2개 겹쳐서 나옴. 속성(dict:트레이스) 1개만 입력해줘도 됨. 그럼 그래프 1개만 나옴.
    data = [ # dict가 여러개일 때는 리스트로 묶어줌.
        {'type' : 'scatter',
         'mode' : 'markers',
         'x' : df_covid19_100.loc[df_covid19_100['iso_code'] == 'KOR', 'date'],
         'y' : df_covid19_100.loc[df_covid19_100['iso_code'] == 'KOR', 'new_cases'],
         'marker' : {'color' : 'red'}
        },
        {'type' : 'scatter',
         'mode' : 'lines',
         'x' : df_covid19_100.loc[df_covid19_100['iso_code'] == 'KOR', 'date'],
         'y' : df_covid19_100.loc[df_covid19_100['iso_code'] == 'KOR', 'new_cases'],
         'line' : {'color' : 'red', 'dash' : 'dash'}
        }]
    , layout =  ## layout 속성이 1개밖에 없으니까 리스트로 안 묶어줌.
        {'title' : "코로나19 발생 현황",
         'xaxis' : {'title': '날짜', 'showgrid' : False},
         'yaxis' : {'title' : '확진자수'},
         'margin' :{ "l" : 25, "r" : 25, "t" : 50, "b" : 25}  # l : 왼쪽 여백, r : 오른쪽 여백, t : 위 여백, b : 아래 여백 
        }
    ).show()
    
    # 그림 보는법
    # zoom 하고 원하는 영역 지정하면 거기 확대해줌. 그리고 아무대나 더블 클릭하면 다시 돌아감.
    
    # frame : 애니메이션 기능
    # for문으로 각 장면을 만들어서 움직이게 보이도록 돌려주는 것임.
    
    
    


#프레임속성========
# 애니메이션 프레임 생성
frames = []
dates = df_covid19_100.loc[df_covid19_100["iso_code"] == "KOR", "date"].unique()

for date in dates:
    frame_data = {
        "data": [
            {
                "type": "scatter",
                "mode": "markers",
                "x": df_covid19_100.loc[(df_covid19_100["iso_code"] == "KOR") & (df_covid19_100["date"] <= date), "date"],
                "y": df_covid19_100.loc[(df_covid19_100["iso_code"] == "KOR") & (df_covid19_100["date"] <= date), "new_cases"],
                "marker": {"color": "red"}
            },
            {
                "type": "scatter",
                "mode": "lines",
                "x": df_covid19_100.loc[(df_covid19_100["iso_code"] == "KOR") & (df_covid19_100["date"] <= date), "date"],
                "y": df_covid19_100.loc[(df_covid19_100["iso_code"] == "KOR") & (df_covid19_100["date"] <= date), "new_cases"],
                "line": {"color": "blue", "dash": "dash"}
            }
        ],
        "name": str(date)
    }
    frames.append(frame_data)
    
len(frames)  # 2022-10-03 ~ 2023-01-11 동안 반복해서 101개 값을 가짐.



# x축과 y축의 범위 설정
x_range = ['2022-10-03', '2023-01-11']
y_range = [8900, 88172]


# 애니메이션을 위한 레이아웃 설정
layout = {
    "title": "코로나 19 발생현황",
    "xaxis": {"title": "날짜", "showgrid": False, "range": x_range},
    "yaxis": {"title": "확진자수", "range": y_range},
    "margin": {"l": 25, "r": 25, "t": 50, "b": 50},
    "updatemenus": [{
        "type": "buttons",
        "showactive": False,
        "buttons": [{
            "label": "Play",
            "method": "animate",
            "args": [None, {"frame": {"duration": 500, "redraw": True}, "fromcurrent": True}]
        }, {
            "label": "Pause",
            "method": "animate",
            "args": [[None], {"frame": {"duration": 0, "redraw": False}, "mode": "immediate", "transition": {"duration": 0}}]
        }]
    }]
}

# Figure 생성
fig = go.Figure(
    data=[
        {
            "type": "scatter",
            "mode": "markers",
            "x": df_covid19_100.loc[df_covid19_100["iso_code"] == "KOR", "date"],
            "y": df_covid19_100.loc[df_covid19_100["iso_code"] == "KOR", "new_cases"],
            "marker": {"color": "red"}
        },
        {
            "type": "scatter",
            "mode": "lines",
            "x": df_covid19_100.loc[df_covid19_100["iso_code"] == "KOR", "date"],
            "y": df_covid19_100.loc[df_covid19_100["iso_code"] == "KOR", "new_cases"],
            "line": {"color": "blue", "dash": "dash"}
        }
    ],
    layout=layout,
    frames=frames
)

fig.show()





import plotly.express as px
from palmerpenguins import load_penguins

penguins = load_penguins()
penguins.head()


penguins.columns


fig = px.scatter(
    penguins,
    x = "bill_length_mm",
    y = "bill_depth_mm",
    color = 'species'
    )
    
fig.update_layout(
    title = dict(text="팔머펭귄 종별 부리 길이 vs. 깊이"),
    paper_bgcolor = 'black',  # plot_bgcolor랑 같이 돌려야지 바뀜
    plot_bgcolor = 'black',
    legend = dict(font = dict(color = 'white'))
).show()
    




# 데이터 패키지 설치
# !pip install palmerpenguins
import pandas as pd
import numpy as np
import plotly.express as px
from palmerpenguins import load_penguins

penguins = load_penguins()
penguins.head()

# x: bill_length_mm
# y: bill_depth_mm  
fig = px.scatter(
    penguins,
    x="bill_length_mm",
    y="bill_depth_mm",
    color="species"
)
# 레이아웃 업데이트
fig.update_layout(
    title=dict(text="팔머펭귄 종별 부리 길이 vs. 깊이", 
               font=dict(color="white")),
               
    paper_bgcolor="black",
    plot_bgcolor="black",
    font=dict(color="white"),
    xaxis=dict(
        title=dict(text="부리 길이 (mm)", font=dict(color="white")), 
        tickfont=dict(color="white"),
        gridcolor='rgba(255, 255, 255, 0.2)'  # 그리드 색깔 조정
    ),
    yaxis=dict(
        title=dict(text="부리 깊이 (mm)", font=dict(color="white")), 
        tickfont=dict(color="white"),
        gridcolor='rgba(255, 255, 255, 0.2)'  # 그리드 색깔 조정
    ),
    legend=dict(font=dict(color="white")),
)

fig.show()





fig = px.scatter(
    penguins,
    x="bill_length_mm",
    y="bill_depth_mm",
    color="species",
    title="팔머펭귄 종별 부리 길이 vs. 깊이"
)

# Update layout with enhancements
fig.update_layout(
    title=dict(
        text="팔머펭귄 종별 부리 길이 vs. 깊이",
        font=dict(color="white", size=24)  # Increase title font size
    ),
    paper_bgcolor="black",
    plot_bgcolor="black",
    font=dict(color="white"),
    xaxis=dict(
        title=dict(text="부리 길이 (mm)", font=dict(color="white")), 
        tickfont=dict(color="white"),
        gridcolor='rgba(255, 255, 255, 0.2)'
    ),
    yaxis=dict(
        title=dict(text="부리 깊이 (mm)", font=dict(color="white")), 
        tickfont=dict(color="white"),
        gridcolor='rgba(255, 255, 255, 0.2)'
    ),
    legend=dict(
        title=dict(text="펭귄 종", font=dict(color="black")),  # Update legend title
        font=dict(color="black")
    ),
    margin=dict(l=40, r=40, t=60, b=40)  # Adjust margins if needed
)

# Update marker size
fig.update_traces(marker=dict(size=10))  # Increase marker size

# Show plot
fig.show()



from sklearn.linear_model import LinearRegression

model = LinearRegression()

penguins = penguins.dropna()
x = penguins[['bill_length_mm']]
y = penguins['bill_depth_mm']

model.fit(x,y)
linear_fit = model.predict(x)
model.coef_
model.intercept_

fig.add_trace(
    go.Scatter(
        mode = 'lines',
        x=x, y=linear_fit,
        name = '선형회귀직선',
        line=dict(dash="dot")
    )
).show()

fig.add_trace(
    go.Scatter(
        mode = 'lines',
        x=penguins['bill_length_mm'], y=linear_fit,
        name = '선형회귀직선',
        line=dict(dash="dot", color = 'white')  # 선 색 지정 가능
    )
).show()




fig = px.scatter(
    penguins,
    x="bill_length_mm",
    y="bill_depth_mm",
    color="species",
    trendline = 'ols'
).show()





# 범주형 변수로 회귀분석 진행하기
# 범주형 변수인 'species'를 더미 변수로 변환
penguins_dummies = pd.get_dummies(penguins, 
                                  columns=['species'],
                                  drop_first=False)
penguins_dummies.columns  # species 범주갯수만큼 칼럼이 늘어남.
penguins_dummies.iloc[:,-3:]


penguins_dummies2 = pd.get_dummies(penguins, 
                                  columns=['species'],
                                  drop_first=True)   # 첫번째 범주(더미변수)를 없애도 나머지 더미변수로 인해 뭘 의미하는지 알 수 있다. 
penguins_dummies2.columns
penguins_dummies2.iloc[:,-3:]





# x와 y 설정
x = penguins_dummies[["bill_length_mm", "species_Chinstrap", "species_Gentoo"]]  # 더미변수 2개만 사용
y = penguins_dummies["bill_depth_mm"]

# 모델 학습
model = LinearRegression()
model.fit(x, y)

model.coef_   # 10.565261622823762 + 0.20044313*bill_length_mm + (-1.93307791)*species_Chinstrap + (-5.10331533)*species_Gentoo
model.intercept_   

penguins.iloc[[0, 200],:]

# 첫 번째 관측치에 대해서 예측값은
# pred_y0 = 10.565261622823762 + 0.20044313 * 39.1 -1.93307791 * 0 -5.10331533 * 0 
# pred_y200 = 10.565261622823762 + 0.20044313 * 45.0 -1.93307791 * 0 -5.10331533 * 1 


regline_y_pred = model.predict(x)

import matplotlib.pyplot as plt

plt.clf()
#plt.plot(penguins['bill_length_mm'], regline_y_pred)
plt.scatter(penguins['bill_length_mm'], regline_y_pred)
plt.show()


import numpy as np
import pandas as pd
np.arange(10)

# 08/13 지도
import plotly.graph_objects as go

import plotly
plotly.__version__

