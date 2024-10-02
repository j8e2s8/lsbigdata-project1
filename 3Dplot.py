##3차원 평면 그래프 그리는 코드입니다

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from mpl_toolkits.mplot3d import Axes3D

# 데이터 로드
house_train = pd.read_csv("./house price/train.csv")
house_test = pd.read_csv("./house price/test.csv")
sub_df = pd.read_csv("./house price/sample_submission.csv")

# 이상치 제거
house_train = house_train.query('GrLivArea <= 4500')

# 피처와 타겟 변수 선택
x = house_train[['GrLivArea', 'GarageArea']]
y = house_train['SalePrice']

# 모델 학습
model = LinearRegression()
model.fit(x, y)

# 기울기와 절편 출력
slope_grlivarea = model.coef_[0]
slope_garagearea = model.coef_[1]
intercept = model.intercept_

print(f"GrLivArea의 기울기 (slope): {slope_grlivarea}")
print(f"GarageArea의 기울기 (slope): {slope_garagearea}")
print(f"절편 (intercept): {intercept}")

# 3D 그래프 그리기
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# 데이터 포인트
ax.scatter(x['GrLivArea'], x['GarageArea'], y, color='blue', label='data points')

# 회귀 평면
GrLivArea_range = np.linspace(x['GrLivArea'].min(), x['GrLivArea'].max(), 10)
GarageArea_range = np.linspace(x['GarageArea'].min(), x['GarageArea'].max(), 10)

GrLivArea_vals, GarageArea_vals = np.meshgrid(GrLivArea_range, GarageArea_range)
SalePrice_vals = intercept + slope_grlivarea * GrLivArea_vals + slope_garagearea * GarageArea_vals

ax.plot_surface(GrLivArea_vals, GarageArea_vals, SalePrice_vals, color='red', alpha=0.5, label='reg_surface') # alpha : 투명도 (0:완전투명, 1:완전불투명)

ax.scatter(GrLivArea_vals, GarageArea_vals, SalePrice_vals, color='green', s=2 )

a ,b =[3,4,5]
a
b
a , b=[[2,3,4],[5,6]]


# 축 라벨
ax.set_xlabel('GrLivArea')
ax.set_ylabel('GarageArea')
ax.set_zlabel('SalePrice')

plt.legend()
plt.show()