# 07/25

# P(X =k | n , p)
# n : 베르누이 확률변수 더한 갯수
# p : 1이 나올 확률
# binom.pmf(k, n, p)

from scipy.stats import binom

binom.pmf(0, n=2, p=0.3)
binom.pmf(1, n=2, p=0.3)
binom.pmf(2, n=2, p=0.3)

import numpy as np
for i in np.arange(31):
    print("X=",i,"일 때: ",binom.pmf(i, n=30, p=0.3))  # for문 안 쓰고 k값에 벡터 넣어도 됨
    
binom.pmf([1,2] , n=30, p=0.3)

np.arange(1,55).cumprod()
np.arange(1,4).cumprod()
np.cumprod([1,2,3])

n= np.arange(1,55)
np.log(n)
np.log([1,2,3,4])


import math 
math.comb(54,26)

np.log(math.factorial(54))

np.mean(np.array([2,3,4]))

math.comb(2,0) * (0.7**2)
math.comb(2,1) * (0.3) * 0.7
math.comb(2,2) * (0.3**2) 



# X~ B
binom.pmf(4, n=10,p=0.36)
binom.pmf(np.arange(5), n=10, p=0.36).sum()
binom.pmf(np.arange(3,9), n=10, p=0.36).sum()

binom.pmf(np.arange(4), n=30, p=0.2).sum()+ binom.pmf(np.arange(25,31), n=30, p=0.2).sum() 
binom.pmf(np.concatenate([np.arange(4),np.arange(25,31)]) , n=30, p=0.2).sum()
1-binom.pmf(np.arange(4,25), n=30, p=0.2).sum()


# rvs 함수 (random variable sample)
# 표본 추출 함수


binom.rvs(n=30, p=0.26, size=5)
30*0.26



import pandas as pd
import seaborn as sns
plt.clf()
sns.barplot(x=np.arange(31), y= binom.pmf(np.arange(31), n=30, p=0.26))
plt.show()

plt.clf()
sns.barplot(x=[0,1], y=bernoulli.pmf([0,1],p=0.3))
plt.show()

# cdf : cumulative dist fuction (누적 확률)
binom.pmf(np.arange(5,19), n=30, p=0.26).sum()
binom.cdf(18,  , z n=30, p=0.26) - binom.cdf(4, n=30, p=0.26)
binom.cdf(19, n=30, p=0.26) - binom.cdf(13, n=30, p=0.26)

plt.clf()
sns.barplot(x=np.arange(31),  y=binom.pmf(np.arange(31), n=30, p=0.26), color='blue')
plt.scatter(10, 0.05, color='red', zorder=10, s=10)
plt.show()

from scipy.stats import binom
import matplotlib.pyplot as plt
import numpy as np

x_1 = binom.rvs(n=30, p=0.26, size=3)
x_1
plt.clf()
sns.barplot(x=np.arange(31),  y=binom.pmf(np.arange(31), n=30, p=0.26), color='blue', zorder=1)
plt.scatter(x_1, np.repeat(0.003,3), color='red', zorder=10, s=10)
sns.barplot(x=30*0.26, y=binom.pmf(30*0.26, n=30, p=0.26) , color='yellow', zorder=11)
plt.axvline(x=30*0.26, color='green', linestyle='--', linewidth=2)
plt.show()



x_1 = binom.rvs(n=30, p=0.26, size=1)
x_1
plt.clf()
sns.barplot(x=np.arange(31),  y=binom.pmf(np.arange(31), n=30, p=0.26), color='blue', zorder=1)
plt.scatter(x_1, 0.003, color='red', zorder=3)
sns.barplot(x=30*0.26, y=binom.pmf(30*0.26, n=30, p=0.26) , color='yellow', zorder=11)
plt.axvline(x=30*0.26, color='green', linestyle='--', linewidth=2)
plt.show()





x_i = bernoulli.rvs(p=0.3)

plt.clf()
sns.barplot(x=[0,1], y=bernoulli.pmf([0,1], p=0.3))
plt.scatter(x=x_i, y=0.02, color = 'red',  s=10, zorder = 2) # 해당좌표에 점 찍기
            # x : x 좌표, y : y 좌표, color : 색 지정, s : 점 크기, zorder : 그래프를 여러개 설정했을 때 그래프를 쌓을 z축 좌표
plt.axvline(x=0.3, color = 'green', linestyle='--', linewidth=2)  # 선 표시하기 
            # x : x 좌표, color : 색 지정, linestyle : 선 스타일 지정, linewidth : 선 굵기 지정
plt.show()




binom.ppf(0.5, n=30, p=0.26)
binom.ppf(0.7, n=30, p=0.26)
binom.cdf(8, n=30, p=0.26)
binom.cdf(9, n=30, p=0.26)
binom.cdf(7, n=30, p=0.26)

math.pi

from scipy.stats import norm
norm.pdf(5, loc=3, scale=4)

np.linspace(-3,3,5)
norm.pdf(np.linspace(-3,3,5), loc=0, scale=1)
norm.pdf(4, loc=0, scale=1)

plt.clf()
plt.scatter(x=np.linspace(-3,3,5), y=norm.pdf(np.linspace(-3,3,5), loc=0, scale=1), color='blue')
plt.show()

plt.clf()
plt.scatter(x=np.linspace(-3,3,100), y=norm.pdf(np.linspace(-3,3,100), loc=0, scale=1), color='blue')
plt.show()


norm.pdf(np.linspace(-5,5,100), loc=0, scale=1)




plt.clf()
plt.plot(np.linspace(-5,5,100), norm.pdf(np.linspace(-5,5,100), loc=0, scale=1), color="black")  # x=, y= 설정하면 에러남
plt.show()


plt.clf()
plt.plot(np.linspace(-5,5,5), norm.pdf(np.linspace(-5,5,5), loc=0, scale=1), color="black")  # x=, y= 설정하면 에러남
plt.show()




plt.clf()
plt.plot(np.linspace(-5,5,100), norm.pdf(np.linspace(-5,5,100), loc=0, scale=1), color="black")
plt.plot(np.linspace(-5,5,100), norm.pdf(np.linspace(-5,5,100), loc=0, scale=2), color="red")
plt.plot(np.linspace(-5,5,100), norm.pdf(np.linspace(-5,5,100), loc=0, scale=0.5), color="blue")
plt.show()


norm.cdf(0, loc=0, scale=1)
norm.cdf(100, loc=0, scale=1)
norm.cdf(0.54, loc=0, scale=1) - norm.cdf(-2, loc=0, scale=1)
norm.cdf(1, loc=0, scale=1) + (1-norm.cdf(3, loc=0, scale=1))

norm.cdf(5, loc=3, scale=5) - norm.cdf(3, loc=3, scale=5)

norm.rvs(loc=0, scale=1, size=100)

from scipy.stats import norm
import numpy as np
sum(norm.rvs(loc=0, scale=1, size=1000)<0)/1000
np.mean(norm.rvs(loc=0, scale=1, size=1000) < 0)

import seaborn as sns
import matplotlib.pyplot as plt
plt.clf()
sns.histplot(norm.rvs(loc=3, scale=2, size=1000))
plt.show()

plt.clf()
sns.histplot(norm.rvs(loc=3, scale=2, size=1000), stat="density")
plt.show()

plt.clf()
sns.histplot(norm.rvs(loc=3, scale=2, size=1000), stat="frequency")
plt,show()




# ------------------

plt.clf()
x = norm.rvs(loc=3, scale=2, size=1000)
sns.histplot(x, stat="density")
plt.show()

plt.clf()
x = norm.rvs(loc=3, scale=2, size=1000)
sns.histplot(x, stat="density")
xmin,xmax = (x.min(), x.max())
x_values= np.linspace(xmin, xmax, 100)
pdf_values = norm.pdf(x_values, loc=3, scale=2)
plt.plot(x_values, pdf_values, color='red', linewidth=2, zorder=3)
plt.xlim([-10,10])
plt.ylim([0,0.3])
plt.show()

from scipy.stats import binom
x_1 = binom.rvs(n=30, p=0.26, size=3)
x_1
plt.clf()
sns.barplot(x=np.arange(31),  y=binom.pmf(np.arange(31), n=30, p=0.26), color='blue', zorder=1)
plt.scatter(x_1, np.repeat(0.003,3), color='red', zorder=10, s=10)
sns.barplot(x=30*0.26, y=binom.pmf(30*0.26, n=30, p=0.26) , color='yellow', zorder=11)
plt.axvline(x=30*0.26, color='green', linestyle='--', linewidth=2)
plt.show()


# 07/26
from scipy.stats import uniform
uniform.rvs(loc=2, scale=4, size=1)

import numpy as np
x=np.linspace(0,8,100)
u_pdf = uniform.pdf(x, loc=0, scale=1)
uniform.cdf(3.25, loc=2, scale=4)

uniform.cdf(8.39, loc=2, scale=4) - uniform.cdf(5, loc=2, scale=4)
uniform.ppf(0.93,loc=2,scale=4)

uniform.rvs(loc=2, scale=4, size=20, random_state=42).mean()

from scipy.stats import norm
norm.rvs(loc=2, scale=4, size=20, random_state=42)


x=uniform.rvs(loc=2, scale=4, size=20*1000, random_state=42)
x = x.reshape(-1,20)
x.shape
x.mean(axis=1)

x.mean(axis=1).shape


import seaborn as sns
import matplotlib.pyplot as plt

plt.clf()
sns.histplot(x.mean(axis=1))
plt.show()

uniform.var(loc=2,scale=4)
uniform.expect(loc=2, scale=4)








x=uniform.rvs(loc=2, scale=4, size=20*1000, random_state=42)
x = x.reshape(-1,20)
blue_x = x.mean()

xmin, xmax = (blue_x.min(), blue_x.max())

plt.clf()
x_values = np.linspace(3, 6, 100)
pdf_values = norm.pdf(x_values, loc=4, scale=np.sqrt(1.3333333/20))
plt.plot(x_values, pdf_values, color='red', linewidth=2)
plt.axvline(x=4, color='green', linestyle='--', linewidth=2)
# 표본평균(파란 벽돌) 점찍기
plt.scatter(uniform.rvs(loc=2, scale=4, size=20).mean(), y=0.02, color='blue', zorder=10, s=10)
x1=(blue_x + 2.57*np.sqrt(1.3333333/20))
x2=(blue_x - 2.57*np.sqrt(1.3333333/20))
# x1=(blue_x + 1.96*np.sqrt(1.3333333/20))  # 1.96 = norm.ppf(0.975, loc=0, scale=1*)
# x2=(blue_x - 1.96*np.sqrt(1.3333333/20))
# plt.axvline(x=blue_x+0.665 , color='blue', linestyle='--', linewidth=1)
# plt.axvline(x=blue_x-0.665 , color='blue', linestyle='--', linewidth=1)
plt.axvline(x=x1, color='blue', linestyle='--', linewidth=1) # 2.57은 이론 상 나온 정해진 값임.
plt.axvline(x=x2, color='blue', linestyle='--', linewidth=1)
plt.xlim([3,5])
plt.show()









norm.ppf(0.025, loc=4, scale=np.sqrt(1.3333333/20))
norm.ppf(0.975, loc=4, scale=np.sqrt(1.3333333/20))

norm.ppf(0.005, loc=4, scale=np.sqrt(1.3333333/20))
norm.ppf(0.995, loc=4, scale=np.sqrt(1.3333333/20))


(4-norm.ppf(0.025, loc=4, scale=np.sqrt(1.3333333/20))) / np.sqrt(1.3333333/20)



# 24.07.29
import numpy as np
import pandas as pd
np.random.seed(20240729)
new_seat = np.random.choice(np.arange(1,29), 28, replace=False)
result = pd.DataFrame({ "old_seat" : np.arange(1,29),
                        "new_seat" : new_seat})
result



# y=2x 그래프 그리기
import matplotlib.pyplot as plt
plt.clf()
plt.plot([1,-1] , [2,-2], color='red')
plt.axhline(y=0, color ='black')
plt.axvline(x=0, color='black')
plt.show()

# y=x^2를 점 3개 이용해서 그래프 그리기
plt.clf()
x=np.linspace(-5,5,500)
plt.plot(x, x**2, color='red')
plt.axhline(y=0, color ='black')
plt.axvline(x=0, color='black')
#plt.axis('equal')
plt.xlim([-5,5])
plt.ylim([-1,20])
plt.gca().set_aspect('equal', adjustable='box')
plt.show()


# 1-α : 신뢰수준 ex) 1-α : 90% -> α : 0.1
# 90% 신뢰구간 구하기
x = [79.1, 68.8, 62.0, 74.4, 71.0, 60.6, 98.5, 86.4, 73.0, 40.8, 61.2, 68.7, 61.6, 67.7, 61.7, 66.8]
x_bar=np.mean(x)
from scipy.stats import norm
z_005 = norm.ppf(0.95, loc=0, scale=1)
(x_bar - z_005*6/np.sqrt(16)  , x_bar + z_005*6/np.sqrt(16))

# 95% 신뢰구간 구하기
x = [79.1, 68.8, 62.0, 74.4, 71.0, 60.6, 98.5, 86.4, 73.0, 40.8, 61.2, 68.7, 61.6, 67.7, 61.7, 66.8]
x_bar=np.mean(x)
from scipy.stats import norm
z_0025 = norm.ppf(0.975, loc=0, scale=1)
(x_bar - z_005*6/np.sqrt(16)  , x_bar + z_005*6/np.sqrt(16))

z_0005 = norm.ppf(0.995, loc=0, scale=1)

# 데이터로 부터 E[X^2] 구하기. X~N(3,5^2)
x=norm.rvs(loc=3, scale=5, size=10000)
np.mean(x**2) # 표본이 얼마냐에 따라서 계산 값이 바뀜. 그래도 34 언저리에서 나올 거임.  # E[X^2] = Var[x] + E[X]^2 = 34 
sum(x**2)/(len(x)-1)  # 33.0732

# 데이터로 부터 E[(X-X^2)/(2X)] 구하기.
x=norm.rvs(loc=3, scale=5, size=10000)
np.mean((x-x**2)/(2*x))  # E[(X-X^2)/(2X)] 값을 정확히는 모르겠지만, 표본의 기댓값으로 대략 -0.974라는 것을 알 수 있음.

# 몬테카를로 적분 : 확률변수 기댓값을 구할 때, 표본을 많이 뽑은 후, 원하는 형태로 변형, 평균을 계산해서 기댓값을 구하는 방법
# 아무리 복잡한 확률변수여도 가능함. 각 확률변수가 독립이 아니어도 가능.

# 표본 10만개 추출해서 s^2 구해보기
np.random.seed(20240729)
x=norm.rvs(loc=3, scale=5, size=100000)
x_bar = np.mean(x)
sum((x-x_bar)**2)/(len(x)-1)
np.var(x, ddof=1)
np.var(x) # Var(X) 임.


# 교재 8장, p.212
import seaborn as sns
economics = pd.read_csv('data/economics.csv')
economics.head()
economics.info()

plt.clf()
sns.lineplot(data=economics, x='date', y='unemploy')
plt.show()

# 214
economics['date2'] = pd.to_datetime(economics['date'])
economics.info()
economics['year'] = economics['date2'].dt.year
economics['date2'].dt.year + 1
economics['date3'] = economics['date2'] + pd.DateOffset(months=1)
economics[['date2', 'date3']]

economics['date2'].dt.is_leap_year # 윤년 체크

plt.clf()
sns.lineplot(data=economics, x='year', y='unemploy')
plt.show()

plt.clf()
sns.scatterplot(data=economics, x='year', y='unemploy',s=2)
plt.show()


economics.head()
my_df = economics.groupby('year', as_index=False).agg(year_mean=('unemploy','mean')
                                                       ,year_std=('unemploy','std')
                                                       ,year_count=('unemploy','count'))
                              
my_df['left_ci'] = my_df['year_mean'] - 1.96*my_df['year_std']/np.sqrt(my_df['year_count'])
my_df['right_ci'] = my_df['year_mean'] + 1.96*my_df['year_std']/np.sqrt(my_df['year_count'])       
my_df                 

x=my_df['year']
y=my_df['year_mean']
plt.clf()
plt.plot(x,y)
plt.show()

np.std(economics.query('year==1967')['unemploy'])
np.sqrt(np.var(economics.query('year==1967')['unemploy']))
np.sqrt(np.var(economics.query('year==1967')['unemploy'], ddof=1))



from scipy.stats import binom
from scipy.stats import norm
norm.var(loc=10, scale=4)

x=norm.rvs(loc=10, scale=4)




#7/30
import pandas as pd

df = pd.read_csv("https://docs.google.com/spreadsheets/d/1RC8K0nzfpR3anLXpgtb8VDjEXtZ922N5N0LcSY5KMx8/gviz/tq?tqx=out:csv&sheet=Sheet2")
df.head()





# 7/31 수업 (복사 붙여넣기 하기)

from scipy.stats import norm
x = norm.ppf(0.25, loc=3, scale=7)
z = norm.ppf(0.25, loc=0, scale=1)
3 + z * 7
x  # x = 3+z*7

norm.cdf(5, loc=3, scale=7)
norm.cdf(2/7, loc=0, scale=1)

norm.ppf(0.975, loc=0, scale=1)



# 방법1
z = norm.rvs(loc=0, scale=1, size=1000)
x = norm.rvs(loc=3, scale=np.sqrt(2), size=1000)
z_min, z_max = (z.min(), z.max())
x_min, x_max = (x.min(), x.max())
plt.clf()
sns.histplot(z, stat='density')
sns.histplot(x, stat='density', color='grey', zorder=10)
plt.plot(np.linspace(z_min,z_max,500), norm.pdf(np.linspace(z_min,z_max,500), loc=0, scale=1), color='red')
plt.plot(np.linspace(x_min,x_max,500), norm.pdf(np.linspace(x_min,x_max,500), loc=3, scale=np.sqrt(2)), color='red')
plt.show()


# 방법2
z = np.sort(norm.rvs(loc=0, scale=1, size=1000))
plt.clf()
sns.histplot(z, stat='density')
plt.plot(z, norm.pdf(z, loc=0, scale=1), color='red')
plt.show()




# x ~ N(μ, σ^2) , (x-μ)/σ ~ N(0,1) 분포가 맞는지 확인
x = norm.rvs(loc=5, scale=3, size=1000)
z = (x-5)/3
z_min, z_max = (z.min(), z.max())
z2 = np.linspace(z_min, z_max, 500)
plt.clf()
sns.histplot(z, stat='density')
plt.plot(z2, norm.pdf(z2, loc=0, scale=1), color='red')
plt.rcParams['axes.unicode_minus'] = False
plt.show()



# x ~ N(μ, σ^2) , x_bar ~ N(μ, σ^2/n) 분포가 맞는지 확인
x = norm.rvs(loc=5, scale=3, size=1000*20)
x = x.reshape(-1,20)
x_bar = x.mean(axis=1)
x_bar_min, x_bar_max = (x_bar.min(), x_bar.max())
x2 = np.linspace(x_bar_min, x_bar_max, 500)

plt.clf()
sns.histplot(x_bar, stat='density')
plt.plot(x2, norm.pdf(x2,loc=5, scale=3/np.sqrt(20)), color='red')
plt.show()


# x ~ ?(μ, σ^2) , x_bar ~=  N(μ, σ^2/n) 분포가 맞는지 확인
from scipy.stats import uniform
x = uniform.rvs(loc=2, scale=6, size=1000*20)
mu = uniform.expect(loc=2, scale=6)
sigma = uniform.std(loc=2, scale=6)
x = x.reshape(-1,20)
x_bar = x.mean(axis=1)
x_bar_min, x_bar_max = (x_bar.min(), x_bar.max())
x2 = np.linspace(x_bar_min, x_bar_max, 500)

plt.clf()
sns.histplot(x_bar, stat='density')
plt.plot(x2, norm.pdf(x2, loc=mu, scale=sigma/np.sqrt(20)), color='red')
plt.show()


# y ~ Bernoulli(p) , x = sum(y) ~ B(n,p) , {(x/n) - p} / {p(1-p)/n} ~ N(0,1) (단, np≥5 or n(1-p)≥5) 분포가 맞는지 확인
from scipy.stats import binom, norm
x = binom.rvs(n=20, p=1/4, size=1000)
x_n = x/20
mu = binom.expect(args=(20,1/4))/20  # <- p와 값이 같아짐
sigma = binom.std(n=20,p=1/4) / 20  # <- sqrt(p(1-p)/n)와 값이 같아짐
z = (x_n-mu)/sigma
z_min, z_max = (z.min(),z.max())
z2 = np.linspace(z_min, z_max, 500)

plt.clf()
sns.histplot(z, stat='density')
plt.plot(z2, norm.pdf(z2, loc=0, scale=1), color='red')
plt.show()



# x ~ N(μ, σ^2) , x_bar ~ N(μ, σ^2/n) , (x_bar - μ)/(σ/sqrt(n)) ~ N(0,1)
from scipy.stats import norm
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
x = norm.rvs(loc=5, scale=3, size=1000*20)
x = x.reshape(-1,20)
x_bar = x.mean(axis=1)
z = (x_bar - 5)/(3/np.sqrt(20))
z_min, z_max = (z.min(), z.max())
z2 = np.linspace(z_min, z_max, 500)

plt.clf()
sns.histplot(z, stat='density')
plt.plot(z2, norm.pdf(z2,loc=0, scale=1), color='red')
plt.show()



# 질문2 : x ~ N(μ, σ^2) , x_bar ~ N(μ, s^2/n) , (x_bar - μ)/(s/sqrt(n)) ~ t(표본갯수 n-1)
# 안 맞는 것 같은데...
from scipy.stats import t
x_sample = norm.rvs(loc=5, scale=3, size=20)
s_sample = np.std(x_sample, ddof=1)

x = norm.rvs(loc=5, scale=3, size=1000*20)
x = x.reshape(-1,20)
x_bar = x.mean(axis=1)
z = (x_bar - 5)/(s_sample/np.sqrt(20))
z_min, z_max = (z.min(), z.max())
z2 = np.linspace(z_min, z_max, 500)

plt.clf()
sns.histplot(z, stat='density')
plt.plot(z2, t.pdf(z2, df=19), color='red')
plt.show()




#지금 이게 t분포 본거야? yes
x = norm.rvs(loc=5, scale=3, size=10)  # size가 커질 수록 표준정규분포에서 덜 떨어지는 것 같음. 가까워지는 것 같음. 
s = np.std(x, ddof=1)  # 모표준편차가 3인데 그 언저리로 표본표준편차는 2.48 나옴. (표본표준편차는 돌릴 때마다 달라질 것임.)
x2 = norm.rvs(loc=5, scale=3, size=1000)
z = (x2-5)/s
z_min, z_max = (z.min(), z.max())
z2 = np.linspace(z_min, z_max, 600)
plt.clf()
sns.histplot(z, stat='density')
plt.plot(z2, norm.pdf(z2, loc=0, scale=1), color='red')
plt.show()






# 질문  : 그럼 t1은 어떤 분포를 따르는지 t분포 선 그래프를 어떻게 그릴 수 있지?
# 히스토그램은 표준정규분포보다 작어졌다가 커졌다가 계속 바뀌는데, 특정 t분포를 따를 수가 있나?
from scipy.stats import t
x_sample = norm.rvs(loc=0, scale=1, size=10)
s_sample = np.std(x_sample, ddof=1)

x = norm.rvs(loc=0, scale=1, size=1000)  # 히스토그램 x축 계산할 때 이용
t1 = (x-5)/s_sample  # 히스토그램 x축
t1_min, t1_max = (t1.min(), t1.max())
t2 = np.linspace(t1_min, t1_max, 600)  # 표준정규분포 x축 
plt.clf()
sns.histplot(t1, stat='density')
plt.plot(t2, t.pdf(t2, df=1), color='red')   # 여기서도 df가 n-1임?
plt.plot(t2, norm.pdf(t2, loc=0, scale=1), color='black')
plt.show()





# 위에 꺼랑 비교 
x2 = norm.rvs(loc=5, scale=3, size=1000)
z = (x2-5)/3
z_min, z_max = (z.min(), z.max())
z2 = np.linspace(z_min, z_max, 600)
plt.clf()
sns.histplot(z, stat='density')
plt.plot(z2, norm.pdf(z2, loc=0, scale=1), color='red')
plt.show()








# t분포
from scipy.stats import t
t_values = np.linspace(-4,4, 100)
z2 = np.linspace(-4, 4, 100)

plt.clf()
plt.plot(t_values, t.pdf(t_values, df=5), color='red', linewidth=2)  # t(4)
plt.plot(t_values, t.pdf(t_values, df=1), color='blue', linewidth=2)
plt.plot(z2, norm.pdf(z2, loc=0, scale=1), color='black')  # 표준정규분포
plt.show()



x=norm.rvs(loc=15, scale=3, size=16, random_state=42)
x

x_bar = x.mean()
n=len(x)
x_bar + t.ppf(0.975, df=n-1)*np.std(x, ddof=1)/np.sqrt(n)
x_bar - t.ppf(0.975, df=n-1)*np.std(x, ddof=1)/np.sqrt(n)





# 질문 : x ~ ?(μ, σ^2) , x_bar ~ N(μ, σ^2/n) <- 여기서 n은 x_bar 1개를 구할 때 쓰인 x의 갯수?
x = norm.rvs(loc=5, scale=3, size=1000*20)
x = x.reshape(-1,20)
x_bar = x.mean(axis=1)
x_bar_min , x_bar_max = (x_bar.min(), x_bar.max())
z = np.linspace(x_bar_min, x_bar_max, 1000)

plt.clf()
sns.histplot(x_bar, stat='density')
plt.plot(z, norm.pdf(z,loc=5, scale=3/np.sqrt(20)), color='red')  # <- np.sqrt(20)이 맞아?
plt.show()


# 직선의 방정식
# y = ax+b
# y = 2x+3
a = 2
b = 3
x = np.linspace(-5,5,20)
y = a*x +b


plt.clf()
plt.plot(x, y)
plt.axvline(x=0, color='black')
plt.axhline(y=0, color='black')
plt.show()



# 08/02 수업
import numpy as np
from scipy.optimize import minimize

# 최소값을 찾을 다변수 함수 정의
def my_f(x):
    return x**2+3
# 초기 추정값
initial_guess = [1]
#최소값 찾기
result = minimize(my_f, initial_guess)
print("최소값:", result.fun)
print("최소값을 갖는 x 값:", result.x)


# 최소값을 찾을 다변수 함수 정의
def my_f2(x):
    return x[0]**2 + x[1]**2 +3
# 초기 추정값
initial_guess = [1,3]
#최소값 찾기
result = minimize(my_f2, initial_guess)
print("최소값:", result.fun)
print("최소값을 갖는 x 값:", result.x)




# 최소값을 찾을 다변수 함수 정의
def my_f3(x):
    return (x[0]-1)**2 + (x[1]-2)**2 + (x[2]-4)**2+7
# 초기 추정값
initial_guess = [-10,2,4]
#최소값 찾기
result = minimize(my_f2, initial_guess)
print("최소값:", result.fun)
print("최소값을 갖는 x 값:", result.x)
   
   

