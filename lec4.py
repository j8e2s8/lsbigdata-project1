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




# 체크1 : y ~ Bernoulli(p) , x = sum(y) ~ B(n,p) , {x - np} / sqrt{np(1-p)} ~ N(0,1) (단, np≥5 or n(1-p)≥5) 분포가 맞는지 확인
from scipy.stats import binom, norm
import matplotlib.pyplot as plt
import seaborn as sns

x = binom.rvs(n=10000, p=0.1, size=1000)
mu = binom.expect(args=(10000,0.1))  #<- np에 가까움
sigma = binom.std(n=10000, p=0.1)  #<- np(1-p)
z = (x-mu)/sigma
z_min, z_max = (z.min(),z.max())
z2 = np.linspace(z_min, z_max, 500)

plt.clf()
sns.histplot(z, stat='density')
plt.plot(z2, norm.pdf(z2, loc=0, scale=1), color='red')
plt.xlim([-4,4])
plt.ylim([0,0.6])
plt.show()

x = binom.rvs(n=20, p=0.4, size=1000)
mu = binom.expect(args=(20,0.4))  #<- np에 가까움
sigma = binom.std(n=20, p=0.4)  #<- np(1-p)
z = (x-mu)/sigma
z_min, z_max = (z.min(),z.max())
z2 = np.linspace(z_min, z_max, 500)

plt.clf()
sns.histplot(z, stat='density')
plt.plot(z2, norm.pdf(z2, loc=0, scale=1), color='red')
plt.xlim([-4,4])
plt.ylim([0,0.5])
plt.show()






# 체크2 : y ~ Bernoulli(p) , x = sum(y) ~ B(n,p) , {(x/n) - p} / sqrt{p(1-p)/n} ~ N(0,1) (단, np≥5 or n(1-p)≥5) 분포가 맞는지 확인
from scipy.stats import binom, norm
x = binom.rvs(n=10000, p=0.4, size=1000)
x_n = x/10000
mu = binom.expect(args=(10000,0.4))/10000  # <- p와 값이 같아짐
sigma = binom.std(n=10000,p=0.4) / 10000  # <- sqrt(p(1-p)/n)와 값이 같아짐
z = (x_n - mu)/sigma
z_min, z_max = (z.min(),z.max())
z2 = np.linspace(z_min, z_max, 500)

plt.clf()
sns.histplot(z, stat='density')
plt.plot(z2, norm.pdf(z2, loc=0, scale=1), color='red')
plt.xlim([-4,4])
plt.ylim([0,0.5])
plt.show()


from scipy.stats import bernoulli, norm
y = bernoulli.rvs(p=0.4, size=1000*10000)
y = y.reshape(-1,10000)
x = y.sum(axis=1)
mu = 0.4 * 10000
sigma = np.sqrt(0.4*0.6*10000)
z = (x - mu)/ sigma
z_min, z_max = (z.min(), z.max())
z2 = np.linspace(z_min, z_max, 500)

plt.clf()
sns.histplot(z, stat='density')
plt.plot(z2, norm.pdf(z2, loc=0, scale=1), color='red')
plt.xlim([-4,4])
plt.ylim([0,0.5]) 
plt.show()





# 체크3 :  y ~ Bernoulli(p) , x = sum(y) ~ B(n,p) , {(x/n) - p} / {p(1-p)/n} ~ N(0,1) (단, np≥5 or n(1-p)≥5) 분포가 맞는지 확인
from scipy.stats import bernoulli, binom, norm
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
x = bernoulli.rvs(p=1/4, size=1000*20)
x = x.reshape(-1,20)
x_bar = x.mean(axis=1)
mu = binom.expect(args=(20,1/4))/20  # <- p와 값이 같아짐
sigma = binom.std(n=20,p=1/4) / 20  # <- sqrt(p(1-p)/n)와 값이 같아짐
z = (x_bar - mu)/sigma
z_min, z_max = (z.min(),z.max())
z2 = np.linspace(z_min, z_max, 500)

plt.clf()
sns.histplot(z, stat='density')
plt.plot(z2, norm.pdf(z2, loc=0, scale=1), color='red')
plt.xlim(-4,4)
plt.ylim(0,1)
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



# x ~ N(μ, σ^2) , x_bar ~ N(μ, s^2/n) , (x_bar - μ)/(s/sqrt(n)) ~ t(표본갯수 n-1)
from scipy.stats import t, norm
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

x = norm.rvs(loc=5, scale=3, size=1000*5)
x = x.reshape(-1,5)
x_bar = x.mean(axis=1)
sample_std = x.std(axis=1)
z = (x_bar - 5)/(sample_std/np.sqrt(5))
z_min, z_max = (z.min(), z.max())
z2 = np.linspace(z_min, z_max, 500)

plt.clf()
sns.histplot(z, stat='density')
plt.plot(z2, t.pdf(z2, df=5-1), color='yellow')
plt.plot(z2, norm.pdf(z2, loc=0, scale=1), color='red')
plt.xlim([-7 , 7])
plt.ylim([0 , 0.5])
plt.show()



x = norm.rvs(loc=5, scale=3, size=1000*1000)
x = x.reshape(-1,1000)
x_bar = x.mean(axis=1)
sample_std = x.std(axis=1)
z = (x_bar - 5)/(sample_std/np.sqrt(1000))
z_min, z_max = (z.min(), z.max())
z2 = np.linspace(z_min, z_max, 500)

plt.clf()
sns.histplot(z, stat='density')

plt.plot(z2, norm.pdf(z2, loc=0, scale=1), color='red')
plt.plot(z2, t.pdf(z2, df=1000-1), color='yellow')
plt.xlim([-7 , 7])
plt.ylim([0 , 0.5])
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






# 그럼 t1은 어떤 분포를 따르는지 t분포 선 그래프를 어떻게 그릴 수 있지?
# 히스토그램은 표준정규분포보다 작어졌다가 커졌다가 계속 바뀌는데, 특정 t분포를 따를 수가 있나?
from scipy.stats import t
x_sample = norm.rvs(loc=0, scale=1, size=10)
s_sample = np.std(x_sample, ddof=1)

x = norm.rvs(loc=5, scale=3, size=1000)  # 히스토그램 x축 계산할 때 이용
sample_std = x.std(ddof=1)
t1 = (x-5)/sample_std  # 히스토그램 x축
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





#  x ~ ?(μ, σ^2) , x_bar ~ N(μ, σ^2/n)
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
   
   
# y=2x+3 그래프 그리기
from scipy.stats import norm
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

x=np.linspace(0,100,400)
y = 2*x+3 

# np.random.seed(20240805)
obs_x = np.random.choice(np.arange(100),20)
epsilon_i = norm.rvs(loc=0, scale=60, size=20)  # epsilon의 분산이 작아질 수록 빨간색 선이 검정색 선에서 안 벗어남.
obs_y = 2*obs_x + 3 +epsilon_i

#plt.clf()
#plt.plot(x, y, label='y=2x+3', color='black')
#plt.scatter(obs_x, obs_y, color='blue', s=3)
#plt.show()

model = LinearRegression()
obs_x = obs_x.reshape(-1,1)
model.fit(obs_x, obs_y)
y_pred = model.predict(obs_x)

#print("a hat :",model.coef_)
#print("b hat :",model.intercept_)

plt.clf()
plt.plot(x, y ,color='black')  # 실제 모수에 대한 회귀식
plt.plot(obs_x, model.coef_*obs_x + model.intercept_, color='green')
plt.plot(obs_x, y_pred, color='red')  # 우리가 obs_x, obs_y로 추정한 회귀식

plt.scatter(obs_x, obs_y, color='blue', s=3)  # 관측해서 얻은 obs_x, obs_y
plt.xlim([0,100])
plt.ylim([0,300])
plt.show()



# !pip install statsmodels    
import statsmodels.api as sm

obs_x = sm.add_constant(obs_x)
model = sm.OLS(obs_y, obs_x).fit()
print(model.summary())



from scipy.stats import ttest_1samp
import numpy as np
x = np.array([15.078, 15.752, 15.549, 15.56 , 16.098, 13.277, 15.462, 16.116, 15.214, 16.93 , 14.118, 14.927, 15.382, 16.709, 16.804])
ttest_1samp(x, 16, alternative='less')

?ttest_1samp


import pandas as pd
sample = [9.76, 11.1, 10.7, 10.72, 11.8, 6.15, 10.52,
14.83, 13.03, 16.46, 10.84, 12.45]
gender = ["Female"]*7 + ["Male"]*5
my_tab2 = pd.DataFrame({"score": sample, "gender": gender})
my_tab2
from scipy.stats import ttest_ind
male = my_tab2[my_tab2['gender'] == 'Male']
female = my_tab2[my_tab2['gender'] == 'Female']
ttest_ind(female['score'], male['score'], equal_var=True)






# 24.08.07
import json
geo = json.load(open('./data/SIG_Seoul.geojson',  encoding='UTF-8'))
df_pop = pd.read_csv("data/Population_SIG.csv")
df_pop.head()
df_pop.iloc[1:26]

df_seoulpop = df_pop.iloc[1:26]
df_seoulpop['code'] = df_seoulpop['code'].astype(str)
df_seoulpop.info()


geo['features']
type(geo)  # dict 임. key : value
len(geo)  # 길이가 4임.
geo.keys()  # 키가 type, name, crs, features 임.
geo['type'] # value : 'FeatureCollection' 1개
geo['name'] # value : 'sig_seoul' 1개
geo['crs'] # value : dict임.
geo['features']  # 지역 정보, 위치 정보를 가지는 dict값을 가지는 리스트임.


len(geo['features'])  # 구 위치 정보를 dict인 원소 값으로 가지는 리스트임. 리스트 원소 25개.

# 리스트 원소 1개 (종로구) dict를 분해해보기
len(geo['features'][0])  # type, properties, geometry 3개의 key를 가지는 dict임.
type(geo['features'][0])
geo['features'][0].keys() 

geo['features'][0]['type']  # 'Feature' 1개
geo['features'][0]['properties']  # 구 이름 정보를 가짐. dict : 'SIG_CD', 'SIG_ENG_NM', 'SIG_KOR_NM' 키 3개
geo['features'][0]['properties'].keys()
geo['features'][0]['geometry']  # 위치 정보를 가지는 dict :  'type', 'coordinates' 키 2개
geo['features'][0]['geometry'].keys()


# 우리가 필요한 값은
geo['features'][0]['properties']['SIG_KOR_NM'] # '종로구'
geo['features'][0]['geometry']['coordinates'] # 종로구의 위치 정보들을 원소 1개로 가지는 리스트임.
type(geo['features'][0]['geometry']['coordinates']) # 리스트임.
len(geo['features'][0]['geometry']['coordinates'])  # 원소 1개
len(geo['features'][0]['geometry']['coordinates'][0])  # 원소 1개
len(geo['features'][0]['geometry']['coordinates'][0][0])  # 원소 2332개 , 인덱스를 2번 한 게 위치 정보를 가져올 수 있는거임.
ignore_index=True

# 우리가 필요한 값은
geo['features'][0]['properties']['SIG_KOR_NM'] # '종로구'
geo['features'][0]['geometry']['coordinates'][0][0]  # 종로구 위치 정보 2332개 

# 구별로 정보를 얻고 싶다면
geo['features'][i]['properties']['SIG_KOR_NM'] # 'i구'
geo['features'][i]['geometry']['coordinates'][0][0]  # i구 위치 정보 2332개 


import numpy as np
coordinate_array = np.array(coordinate_list[0][0])
x = coordinate_array[:,0]
y = coordinate_array[:,1]

import matplotlib.pyplot as plt
plt.plot(x,y)  # 지도 (경계) 그려짐.
plt.show()
plt.clf()


plt.plot(x[::10],y[::10])  # 지도 (경계)가 좀 러프해짐
plt.show()
plt.clf()


# 숫자가 바뀌면 '구'가 바뀜
geo['features'][0]['properties']
geo['features'][1]['properties']
geo['features'][2]['properties']

# 함수로 만들기
def draw_seoul(num):
    gu_name=geo['features'][num]['properties']['SIG_KOR_NM']
    coordinate_array = np.array(geo['features'][num]['geometry']['coordinates'][0][0])
    x = coordinate_array[:,0]
    y = coordinate_array[:,1]
    
    plt.rcParams.update({"font.family" : "Malgun Gothic"})
    plt.plot(x,y)
    plt.title(gu_name)
    plt.show()
    plt.clf()
    
    return None
    
draw_seoul(12)




# 구별로 정보를 얻고 싶다면
geo['features'][i]['properties']['SIG_KOR_NM'] # 'i구'
geo['features'][i]['geometry']['coordinates'][0][0]  # i구 위치 정보 2332개 

type(geo['features'][0]['properties']['SIG_KOR_NM']) # 'i구' 문자열 값
type(geo['features'][0]['geometry']['coordinates'][0][0])  # i구 위치 정보 2332개 리스트임
len(geo['features'][0]['geometry']['coordinates'][0][0])



# 서울시 전체 지도 그리기 - 내 방법
num = len(geo['features'])

gu_name = []
gu_loc = []

for i in range(num):
    gu_name.append([geo['features'][i]['properties']['SIG_KOR_NM']]* len(geo['features'][i]['geometry']['coordinates'][0][0]) )
    gu_loc.append(geo['features'][i]['geometry']['coordinates'][0][0] )
    x = np.array(gu_loc)[:,0]
    y = np.array(gu_loc)[:,1]
    

len(gu_name)
len(x)
len(y)

    
df = pd.DataFrame({'x' : x,
                   'y' : y,
                    'gu_name' : gu_name})
                    
df.shape

import seaborn as sns

plt.clf()
sns.lineplot(data=df, x='x', y='y', hue='gu_name')
plt.show()

    
    

# 서울시 전체 지도 그리기 - 강사님 방법
def df_seoul(num):
    gu_name=geo['features'][num]['properties']['SIG_KOR_NM']
    coordinate_array = np.array(geo['features'][num]['geometry']['coordinates'][0][0])
    x = coordinate_array[:,0]
    y = coordinate_array[:,1]
    return pd.DataFrame({'gu_name' : gu_name, 'x' : x, 'y': y})
    
df_seoul(12)


result = pd.DataFrame({})
for i in np.arange(25):
    result = pd.concat([result, df_seoul(i)], ignore_index=True)

result['hue_group'] = np.where(result['gu_name']=='강남구', '강남구', 'etc')

plt.clf()
sns.scatterplot(data=result, x='x', y='y', hue='hue_group', palette=['grey', 'red'], s=2, legend=False) 
plt.show()


plt.clf()
sns.scatterplot(data=result, x='x', y='y', hue='hue_group', palette='viridis', s=2, legend=False) 
plt.show()


# !pip install folium
import folium

center_x = result['x'].mean()
center_y = result['y'].mean()

# 흰도화지 map 불러오기
my_map = folium.Map(location = [35.95, 127.7], zoom_start = 8)  # 우리가 구한 중심값이 numpy 값이라서 지도에 안 찍히는 듯
my_map = folium.Map(location = [center_y, center_x], zoom_start = 8)  # 경도 위도 순서로 써줘야함
map_sig = folium.Map(location = [center_y, center_x], zoom_start=12, tiles="cartodbpositron")
map_sig.save("map_seoul.html")  # 돌리면 프로젝트 폴더에 html 파일 생김. 들어가면 지도 나옴.


# 코로플릿 <- 구 경계선 그리기
folium.Choropleth(
    geo_data=geo,
    data=df_seoulpop,
    columns = ("code", "pop"),
    key_on = "feature.properties.SIG_CD").add_to(map_sig)

map_sig.save("map_seoul.html")

bins = list(df_seoulpop['pop'].quantile([0, 0.2, 0.4, 0.6, 0.8, 1]))  # 하위 0, 하위 0.2, 하위 0.4, 하위 0.6 ,... 에 해당하는 값을 반환해줌.
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
df_seoul(0).iloc[:, 1:3].mean()
a = df_seoul(0).iloc[:, 1:3].mean().y
b = df_seoul(0).iloc[:, 1:3].mean().x
folium.Marker([a, b], popup='강남구').add_to(map_sig)
map_sig.save("map_seoul.html")



# 08/12 수업
import os
cwd = os.getcwd()  # 현재 working directory가 어딘지
os.chdir(cwd)  # directory를 working directory로 변경


import plotly as px



# 08/21 
from matplotlib import pyplot as plt
from palmerpenguins import load_penguins
import pandas as pd
from sklearn.linear_model import LinearRegression
import numpy as np
import patsy

df = load_penguins()

model = LinearRegression()
penguins = df.dropna()

penguins_dummies = pd.get_dummies(
                        penguins, 
                        columns=['species'],
                        drop_first=True)

penguins_dummies.columns
x = penuins_dumies['bill_length_mm','species_Chinstrap', 'species_Gentoo']
y = penuins_dumies['bill_depth_mm']

x["bill_Chinstrap"] = x["bill_length_mm"] * x["species_Chinstrap"]
x["bill_Gentoo"] = x["bill_length_mm"] * x["species_Gentoo"]

model.fit(x, y)  # 상호작용 고려

model.coef_
model.intercept_


# patsy를 사용하여 수식으로 상호작용 항 생성
# 0 + 는 절편을 제거함
formula = 'bill_depth_mm ~ 0 + bill_length_mm + species'
y, x = patsy.dmatrices(formula, penguins, return_type='dataframe')
model.fit(x,y)
x   # intercept가 1이거나 0이 나옴. 즉, 베타0를 쓰는 애가 있고 안 쓰는 애가 있음.
model.coef_
model.intercept_


formula = 'bill_depth_mm ~ bill_length_mm + species'
y, x = patsy.dmatrices(formula, penguins, return_type='dataframe')
model.fit(x,y)
model.coef_
model.intercept_
x   # intercept가 다 1로 나옴. 즉, 베타0*1를 쓴다는 의미임.

                                                           # 변수1 * 변수2 로 작성하면, 다음과 같은 컬럼으로 분석해줌.
formula = 'bill_depth_mm ~ 0 + bill_length_mm * species'   # bill_length_mm  I(species='Adelie')  I(species='Chinstrap')  I(species='Gentoo')  bill_length_mm*I(species='Chinstrap')   bill_length_mm*I(species='Gentoo')
y, x = patsy.dmatrices(formula, penguins, return_type='dataframe')
model.fit(x,y)
pd.set_option('display.max_columns', None)
x


formula = 'bill_depth_mm ~ 1 + bill_length_mm * species'   # bill_length_mm  I(species='Adelie')  I(species='Chinstrap')  I(species='Gentoo')  bill_length_mm*I(species='Chinstrap')   bill_length_mm*I(species='Gentoo')
y, x = patsy.dmatrices(formula, penguins, return_type='dataframe')
model.fit(x,y)
x

formula = 'bill_depth_mm ~ 0 + bill_length_mm + body_mass_g + flipper_length_mm + species'   
model.fit(x,y)
x


formula = 'bill_depth_mm ~ 0 + bill_length_mm * body_mass_g *  flipper_length_mm * species'   
model.fit(x,y)
x


x = x.iloc[:1:]  # species(Adelie) 더미변수는 없어야 되는데 있기 때문에 없애줌
model.fit(x, y)
model.coef_
model.intercept_


# 08/22 수업
import numpy as np
import matplotlib.pyplot as plt

x = np.linspace(-8,8, 20)
a, b, c = -4, -5, 2
y = a * x**2 + b * x + c

plt.clf()
plt.plot(x, y)
plt.show()


x = np.linspace(-8,8, 20)
a, b, c, d = 2, 3, 5, -1
a, b, c, d = -2, -3, 5, -1
y = a * x**3 + b * x**2 + c * x + d

plt.clf()
plt.plot(x, y)
plt.show()


x = np.linspace(-8,8, 20)
a, b, c, d, e = 3, 3, 1, -1, 5
y = a * x**4 + b * x**3 + c * x**2 + d * x + e

plt.clf()
plt.plot(x, y)
plt.show()



from scipy.stats import norm, uniform
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt

# 검정색 모회귀곡선
k = np.linspace(-4,4,200)
k_y = np.sin(k)

# 파란 점들
epsilon_i = norm.rvs(loc=0, scale=3, size=20)
x = uniform.rvs(loc=-4, scale=8, size=20)
y = np.sin(x) + epsilon_i


plt.clf()
plt.plot(k, k_y, 'black')   # 모회귀곡선
plt.scatter(x,y)
plt.show()



# train test 데이터 만들기
np.random.seed(42)
x = uniform.rvs(size=30, loc=-4, scale=8)
epsilon_i = norm.rvs(size=30, loc=0, scale=0.3)
y = np.sin(x) + epsilon_i

import pandas as pd
df = pd.DataFrame({ 
    "x": x , "y": y})

train_df = df.loc[:19]
train_df

test_df = df.loc[20:]
test_df


plt.scatter(train_df['x'], train_df['y'] , color='blue')


from sklearn.linear_model import LinearRegression
model = LinearRegression()

train_x= train_df[['x']]
train_y= train_df['y']
model.fit(train_x, train_y)

model.coef_
model.intercept_


reg_line = model.predict(train_x)

plt.clf()
plt.plot(train_x, reg_line, color = 'red')
plt.scatter(train_x, train_y, color='blue')
plt.show()


train_df['x^2'] = train_df['x']**2
train_x2 = train_df[['x', 'x^2']]
train_y2 = train_df['y']

model.fit(train_x2, train_y2)

model.coef_
model.intercept_

k= np.linspace(-4,4,200)
train_k = pd.DataFrame({'x': k , 'x^2':k**2})

reg_curve = model.predict(train_k)

plt.plot(train_k['x'], reg_curve, color='red')
plt.scatter(train_x, train_y, color='blue')



train_df['x^3'] = train_df['x']**3

train_x3 = train_df[['x', 'x^2', 'x^3']]
train_y3 = train_df['y']

model.fit(train_x3, train_y3)

model.coef_
model.intercept_

train_k['x^3'] = train_k['x']**3

reg_curve2 = model.predict(train_k)

plt.plot(train_k['x'], reg_curve2, color='red')
plt.scatter(train_x, train_y, color='blue')



train_df['x^4'] = train_df['x']**4
train_x4 = train_df[['x', 'x^2', 'x^3', 'x^4']]
train_y4 = train_df['y']

model.fit(train_x4, train_y4)

model.coef_
model.intercept_

train_k['x^4'] = train_k['x']**4
reg_curve4 = model.predict(train_k)

plt.plot(train_k['x'], reg_curve4, color='red')
plt.scatter(train_x, train_y, color='blue')




# 9차 곡선
def reg_curve2(n):
    import numpy as np
    import pandas as pd
    from sklearn.linear_model import LinearRegression
    from scipy.stats import norm, uniform
    import matplotlib.pyplot as plt
    np.random.seed(42)
    x = uniform.rvs(loc=-4, scale=8,size=200)
    epsilon_i = norm.rvs(loc=0, scale=0.3, size=200)
    y = np.sin(x) + epsilon_i  # 전수 조사의 관측된 값들
    black_y = np.sin(x)
    whole_df = pd.DataFrame({
        "x" : x , "y" : y , "black_y" : black_y
    })
    index = np.random.randint(0,200, size=30)
    df = whole_df.loc[index,].sort_values('x')   # MSE 구하기 위해 예측값을 구할 데이터셋
    for i in range(2, n+1):
        df[f'x^{i}'] = df['x']**i
    train_df = df.iloc[:20,:]
    test_df = df.iloc[21:,:]

    # train_x = train_df[train_df.columns.difference(['y','y2'])] # columns 순서가 이상해져서 model.fit할 때 에러남
    train_x = train_df.drop(columns=['y','black_y'])
    train_y= train_df['y']
    
    model = LinearRegression()
    model.fit(train_x, train_y)
    
    model.coef_
    model.intercept_
    
    y_pred = model.predict(train_x)

    #k= np.linspace(-4,4,200)
    #k_df = pd.DataFrame({ 'x':k })
    for i in range(2, n+1):   # 부드러운 곡선 그림을 그리기 위해 예측값을 구할 데이터셋
        whole_df[f'x^{i}'] = whole_df['x']**i
    whole_df = whole_df.sort_values('x')
    whole_x = whole_df.drop(columns=(['y','black_y']))
    reg_plot = model.predict(whole_x)

    plt.plot(whole_x['x'], reg_plot, color='red')  # 추정된 회귀곡선
    plt.plot(whole_df['x'] , whole_df['black_y'] ,color='black')  # 모회귀곡선
    plt.scatter(train_x['x'], train_y, color='blue')  # 관측된 관측치

    # test_x = test_df[test_df.columns.difference(['y'])]  # columns 순서가 이상해져서 model.fit할 때 에러남
    test_x = test_df.drop(columns=['y','black_y'])
    test_y = test_df['y']
    test_y_pred = model.predict(test_x)

    return print("최대 차수 n :" , n,"\n train set MSE :",((y_pred - train_df['y'])**2).mean(), "\n train set SSE :", sum((y_pred - train_df['y'])**2), "\n test set MSE :", ((test_y_pred - test_y)**2).mean() , "\n test set SSE :" ,sum((test_y_pred - test_y)**2))



reg_curve2(2)  # 차수가 커질 수록 test set에서 성능이 안 좋아짐


# 위에꺼 아님. 뭔가 이상해, 원래 차수가 커질수록 test mse, sse가 10만자리까지 커졌음.
# 밑에꺼가 맞는 것 같은데 위에꺼 잘 수정하기



def reg_curve(n):
    import numpy as np
    import pandas as pd
    from sklearn.linear_model import LinearRegression
    from scipy.stats import norm, uniform
    import matplotlib.pyplot as plt
    np.random.seed(42)
    x = uniform.rvs(loc=-4, scale=8,size=30)
    epsilon_i = norm.rvs(loc=0, scale=0.3, size=30)
    y = np.sin(x) + epsilon_i  # 관측된 값들
    df = pd.DataFrame({
        "x" : x , "y" : y
    })
    for i in range(2, n+1):
        df[f'x^{i}'] = df['x']**i
    train_df = df.iloc[:20,:]
    test_df = df.iloc[21:,:]

    # train_x = train_df[train_df.columns.difference(['y'])] # columns 순서가 이상해져서 model.fit할 때 에러남
    train_x = train_df.drop(columns=['y'])
    train_y= train_df['y']
    
    model = LinearRegression()
    model.fit(train_x, train_y)
    
    model.coef_
    model.intercept_
    
    y_pred = model.predict(train_x)

    k= np.linspace(-4,4,200)
    k_df = pd.DataFrame({ 'x':k })
    for i in range(2, n+1):
        k_df[f'x^{i}'] = k_df['x']**i
    reg_plot = model.predict(k_df)

    plt.plot(k_df['x'], reg_plot, color='red')
    plt.scatter(train_x['x'], train_y, color='blue')

    # test_x = test_df[test_df.columns.difference(['y'])]  # columns 순서가 이상해져서 model.fit할 때 에러남
    test_x = test_df.drop(columns=['y'])
    test_y = test_df['y']
    test_y_pred = model.predict(test_x)

    return print("최대 차수 n :" , n,"\n train set MSE :",((y_pred - train_df['y'])**2).mean(), "\n train set SSE :", sum((y_pred - train_df['y'])**2), "\n test set MSE :", ((test_y_pred - test_y)**2).mean() , "\n test set SSE :" ,sum((test_y_pred - test_y)**2))



reg_curve(10)  # 차수가 커질 수록 test set에서 성능이 안 좋아짐
reg_curve2(10)

# train set MSE : 0.4896887899730774 
# train set SSE : 9.793775799461548 
# test set MSE : 0.6154594571937836 
# test set SSE : 5.539135114744052




# 08/26 수업
# 
import numpy as np

a = np.arange(1,4)
a
b = np.array([3,6,9]).reshape(-1,1)
b

a.dot(b)


# 행렬 * 벡터 (곱셈)
a = np.array([1,2,3,4]).reshape((2,2), order='F')
a

b = np.array([5,6]).reshape(2,1)
b

a.dot(b)
a @ b


# 행렬 * 행렬
a = np.array([1,2,3,4]).reshape((2,2), order='F')
a

b = np.array([5,6,7,8]).reshape((2,2), order= 'F')
b

a @ b


# Q1
a = np.array([1,2,1,0,2,3]).reshape((2,3))
a
b= np.array([1, 0, -1, 1, 2, 3]).reshape(3,2)
b

a @ b



# Q2
np.eye(3)
a = np.array([3,5,7,2,4,9,3,1,0]).reshape(3,3)

a @ np.eye(3)
np.eye(3) @ a  # 둘이 결과가 같음


# trainspose
a
a.transpose()

b=a[:,0:2]
b.transpose()

c = np.array([1,2,3])
c.transpose()


# 회귀분석 데이터 행렬
x = np.array([13,15,12,14,10,11,5,6]).reshape(4,2)
x
vec1 = np.repeat(1,4).reshape(4,1)
matX = np.hstack((vec1, x))
matX

beta_vec = np.array([2,0,1]).reshape(3,1)
beta_vec
matX @ beta_vec

y = np.array([20,19,20,12]).reshape(4,1)
(y - matX @ beta_vec).transpose()  @  (y- matX @ beta_vec)


# 역행렬 (inverse matrix)
a = np.array([1,5,3,4]).reshape(2,2)
b = np.array([4,-5,-3,1]).reshape(2,2)
b = (-1/11)*b
a_inv = np.linalg.inv(a)
a_inv

a @ b

a = np.array([-4, -6, 2, 5, -1, 3, -2, 4, -3]).reshape(3,3)
a_inv = np.linalg.inv(a)
a_inv


## 역행렬 존재하지 않는 경우 (선형 종속)
b = np.array([1,2,3,2,4,5,3,6,7]).reshape((3,3), order='F')
b_inv = np.linalg.inv(b)

np.linalg.det(b)





# 베타 추정하기
data = np.random.randint(-50,50, 120)
x1 = data[:30]
x2 = data[30:60]
x3 = data[60:90]
y = data[90:]

df = pd.DataFrame({
    'x1' : x1,
    'x2' : x2,
    'x3' : x3,
    'y' : y
})

train_x = df.drop(columns=('y'))
train_y = df['y']


# 베타 추정하기 - 1. (XTX)^(-1)XTy
b0 = np.repeat(1,30).reshape(-1,1)
X = np.hstack((b0,train_x))
XTX = X.transpose() @ X
XTX_inv = np.linalg.inv(XTX)
XTy = X.transpose() @ train_y
XTX_inv @ XTy   #  4.56185895, -0.02151196,  0.15317829, -0.22849104



# 베타 추정하기 - 2. sklearn.linear_model의 LinearRegression의 model fit
from sklearn.linear_model import LinearRegression
model = LinearRegression()
model.fit(train_x, train_y)

model.coef_   # -0.02151196,  0.15317829, -0.22849104
model.intercept_    #  4.561858950771647



# 베타 추정하기 - 3. 최소제곱법
from scipy.optimize import minimize
b0 = np.repeat(1,30).reshape(-1,1)
X = np.hstack((b0,train_x))

def line_beta(beta):  # 베타들은 리스트로 넣기
    beta = np.array(beta).reshape(-1,1)
    y = np.array(train_y).reshape(-1,1)
    e = (y- X @ beta)
    return e.transpose() @ e


# 초기 추정값
initial_guess = [0,0,0,0]  # 베타들은 리스트로 넣기  <- 베타0, 베타1, 베타2, 베타3 있어야 함. 차원 맞춰줘야함
result = minimize(line_beta, initial_guess)

print("최소 잔차제곱합 :", result.fun)
print("최소 잔차제곱합이 되는 베타값 :" , result.x)   # [ 4.56185882 -0.02151197  0.15317828 -0.22849104]






# minimize로 라쏘 베타 구하기
from scipy.optimize import minimize

def line_perform_lasso(beta):
    beta = np.array(beta).reshape(3,1)
    a = (y- matX @ beta)
    return (a.transpose() @ a) + 3*np.abs(beta).sum()

line_perform_lasso([8.55, 5.96, -4.38])

# 초기 추정값
initial_guess = [0,0,0]

# 최소값 찾기
result = minimize(line_perform_lasso, initial_guess)

# 결과 출력
print("최소값:" , result.fun)
print("최소값을 갖는 베타값: ", result.x)




# minimize로 릿지 베타 구하기
from scipy.optimize import minimize

def line_perform_ridge(beta):
    beta = np.array(beta).reshape(3,1)
    a = (y- matX @ beta)
    return (a.transpose() @ a) + 3*(beta**2).sum()

line_perform_ridge([8.55, 5.96, -4.38])

# 초기 추정값
initial_guess = [0,0,0]

# 최소값 찾기
result = minimize(line_perform_ridge, initial_guess)

# 결과 출력
print("최소값:" , result.fun)
print("최소값을 갖는 베타값: ", result.x)





import numpy as np

# 회귀분석 데이터행렬
x=np.array([13, 15,
           12, 14,
           10, 11,
           5, 6]).reshape(4, 2)
x
vec1=np.repeat(1, 4).reshape(4, 1)
matX=np.hstack((vec1, x))
y=np.array([20, 19, 20, 12]).reshape(4, 1)
matX

# minimize로 라쏘 베타 구하기
from scipy.optimize import minimize



def line_perform_lasso(beta):
    beta=np.array(beta).reshape(3, 1)
    a=(y - matX @ beta)
    return (a.transpose() @ a) + 0*np.abs(beta[1:]).sum()  # 라쏘는 베타를 베타1부터 넣음. 베타0 값이 커도 상관이 없다는 것

line_perform_lasso([8.55,  5.96, -4.38])  # 람다가 0일 때 손실함수가 최소값을 가지는 베타들
line_perform_lasso([8.14,  0.96, 0])  # 최소일 때임

# 초기 추정값
initial_guess = [0, 0, 0]

# 최소값 찾기
result = minimize(line_perform_lasso, initial_guess)

# 결과 출력
print("최소값:", result.fun)
print("최소값을 갖는 x 값:", result.x)

# 예측식 : y_hat = 8.55 + 5.96*X1 + -4.38*X2   [람다가 0일 때]








def line_perform_lasso(beta):
    beta=np.array(beta).reshape(3, 1)
    a=(y - matX @ beta)
    return (a.transpose() @ a) + 3*np.abs(beta[1:]).sum()  # 라쏘는 베타를 베타1부터 넣음. 베타0 값이 커도 상관이 없다는 것
                                                    # beta1 부터 넣는 이유 : 람다가 커질수록 베타들 하나씩 추정값이 0 이 됨. 
                                                    # 근데 beta0부터 넣게 되면 엄청 큰 람다에서는 beta0~ betan모두 0으로 추정되고
                                                    # 그러면 y_hat이 0으로 추정됨. 그래서 beta0를 제외하고 beta들을 넣고
                                                    # 아무리 람다가 커도 beta0는 살아있으니까 y_hat은 beta0로 추정됨.
                                                    # 보통 이럴 때 beta0는 y평균값에 가깝게 추정된다.
                                                    # 라쏘와 릿지는 서로 장단점이 있어서 뭐가 더 좋다 라고 할 수 없음
                                                    # 라쏘는 변수 선택 효과가 있는 좋은 점이 있고, 릿지는 ....
line_perform_lasso([8.55,  5.96, -4.38])
line_perform_lasso([8.14,  0.96, 0])  # 최소일 때임

# 초기 추정값
initial_guess = [0, 0, 0]

# 최소값 찾기
result = minimize(line_perform_lasso, initial_guess)

# 결과 출력
print("최소값:", result.fun)
print("최소값을 갖는 x 값:", result.x)

# 예측식 : y_hat = 8.14 + 0.96*X1 + 0*X2   [람다가 3일 때]



def line_perform_lasso(beta):
    beta=np.array(beta).reshape(3, 1)
    a=(y - matX @ beta)
    return (a.transpose() @ a) + 30*np.abs(beta[1:]).sum()  # 라쏘는 베타를 베타1부터 넣음. 베타0 값이 커도 상관이 없다는 것

line_perform_lasso([8.55,  5.96, -4.38])
line_perform_lasso([8.14,  0.96, 0])  # 최소일 때임

# 초기 추정값
initial_guess = [0, 0, 0]

# 최소값 찾기
result = minimize(line_perform_lasso, initial_guess)

# 결과 출력
print("최소값:", result.fun)
print("최소값을 갖는 x 값:", result.x)

# 예측식 : y_hat = 11.5 + 0*X1 + 0.54*X2  [람다가 30일 때]






def line_perform_lasso(beta):
    beta=np.array(beta).reshape(3, 1)
    a=(y - matX @ beta)
    return (a.transpose() @ a) + 500*np.abs(beta[1:]).sum()  # 라쏘는 베타를 베타1부터 넣음. 베타0 값이 커도 상관이 없다는 것

line_perform_lasso([8.55,  5.96, -4.38])
line_perform_lasso([8.14,  0.96, 0])  # 최소일 때임

# 초기 추정값
initial_guess = [0, 0, 0]

# 최소값 찾기
result = minimize(line_perform_lasso, initial_guess)

# 결과 출력
print("최소값:", result.fun)
print("최소값을 갖는 x 값:", result.x)

# 예측식 : y_hat = 11.5 + 0*X1 + 0*X2  [람다가 500일 때] X1, X2가 얼마든간데 11.5 예측값임.
# 람다만 잘 설정하면, 변수가 엄청 많아도 베타 추정이 0인 변수들이 생기면서 자연스럽게 변수 선택하게 되는 효과가 있음.


# 그럼 결국 valid set으로 모델 성능이 좋은 람다를 구하자.
# 람다 값에 따라 변수가 선택 된다.
# x 변수가 추가되면, trainX에서는 어떤 X 차수든지 성능이 항상 좋아짐.
# x 변수가 추가되면, validX에서는 X 차수가 높아질 수록 좋아졌다가 나빠짐 (나빠진다는 것은 오버피팅이 됐기 때문임)
# 람다 0부터 시작 : 내가 가진 모든 변수를 넣겠다
# 람다가 증가 할 수록 : 변수가 하나씩 빠지게 됨.
# 따라서 validX에서 가장 성능이 좋은 람다를 선택!
# 변수가 선택됨을 의미.


# 릿지는 계수 추정이 하나는 0, 하나는 1000000 이런식으로 추정되지 않고 안정적이게 추정됨. 


# 회귀 추정 방법 : (XTX)^(-1)XTy
# 이렇게 회귀 추정하려면 (XTX) 역행렬이 존재해야 함. (XTX) 역행렬이 존재하려면 X가 선형독립이어야 함.

# x의 칼럼에 선형 종속인 애들이 있다 : 다중공선성이 존재한다. -> 즉, (XTX) 역행렬이 없다는 거고 -> 즉, 베타 추정을 못한다는 것임 (추정이 되도 믿을 수 없는 것임)


import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
from scipy.stats import uniform
from sklearn.linear_model import LinearRegression

# 20차 모델 성능을 알아보자능
np.random.seed(2024)
x = uniform.rvs(size=30, loc=-4, scale=8)
y = np.sin(x) + norm.rvs(size=30, loc=0, scale=0.3)

import pandas as pd
df = pd.DataFrame({
    "y" : y,
    "x" : x
})
df

train_df = df.loc[:19]
train_df

for i in range(2, 21):
    train_df[f"x{i}"] = train_df["x"] ** i
    
# 'x' 열을 포함하여 'x2'부터 'x20'까지 선택.
train_x = train_df[["x"] + [f"x{i}" for i in range(2, 21)]]
train_y = train_df["y"]

from sklearn.linear_model import Lasso
model= Lasso(alpha=0.1)  # lambda가 alpha 표현됨
model.fit(train_x, train_y)

model.coef_   # 베타1부터 추정한 값  <- 파라미터 라고 함.  하이퍼파라미터는 데이터, 모델에서부터 결정하는 게 아니라, validation set 에서 얼마로 할지 나중에 결정하는 것임.
                                                        # 람다(alpha)가 하이퍼파리미터임.




model= Lasso(alpha=0)  # lambda가 alpha 표현됨
model.fit(train_x, train_y)



valid_df = df.loc[20:]
valid_df

for i in range(2, 21):
    valid_df[f"x{i}"] = valid_df["x"] ** i

# 'x' 열을 포함하여 'x2'부터 'x20'까지 선택.
valid_x = valid_df[["x"] + [f"x{i}" for i in range(2, 21)]]


# 모델 성능
y_hat_train = model.predict(train_x)   # train set에서의 성능과
y_hat_val = model.predict(valid_x)    # valid set에서의 성능을 각각 구해서 비교할 수 있음.

print("train MSE :",sum((train_df["y"] - y_hat_train)**2))
print("valid MSE",sum((valid_df["y"] - y_hat_val)**2))





# 람다 값 결정하기

val_result = np.repeat(0.0, 100)
tr_result = np.repeat(0.0, 100)

for i in np.arange(1,100):
    model = Lasso(alpha = i*(0.1))
    model.fit(train_x, train_y)

    # 모델 성능
    y_hat_train = model.predict(train_x)
    y_hat_val = model.predict(valid_x)

    perf_train = sum((train_df['y'] - y_hat_train)**2)
    perf_val = sum((valid_df['y'] - y_hat_val)**2)
    tr_result[i] = perf_train
    val_result[i] = perf_val


import seaborn as sns
import pandas as pd
df = pd.DataFrame({
    '1' : np.arange(0,100,0.1),
    'tr' : tr_result,
    'val' : val_result
})



df = pd.DataFrame({
    'lambda': np.arange(0, 1, 0.01), 
    'tr': tr_result,
    'val': val_result
})

# seaborn을 사용하여 산점도 그리기
sns.scatterplot(data=df, x='lambda', y='tr')
sns.scatterplot(data=df, x='lambda', y='val', color='red')
plt.xlim(0, 0.4)

val_result[0]
val_result[1]
np.min(val_result)

# alpha를 0.03로 선택!
np.argmin(val_result)
np.arange(0, 1, 0.01)[np.argmin(val_result)]  # alpha가 0.03으로 나옴







np.random.seed(2024)
x = uniform.rvs(size=30, loc=-4, scale=8)
y = np.sin(x) + norm.rvs(size=30, loc=0, scale=0.3)

import pandas as pd
df = pd.DataFrame({
    "y" : y,
    "x" : x
})
df

train_df = df.loc[:19]
train_df

valid_df = df.loc[20:]
valid_df

valid_x = valid_df['x']
valid_y = valid_df['y']

for i in range(2, 21):
    train_df[f"x^{i}"] = train_df["x"] ** i
    
# 'x' 열을 포함하여 'x2'부터 'x20'까지 선택.
train_x = train_df[["x"] + [f"x^{i}" for i in range(2, 21)]]
train_y = train_df["y"]


k = np.arange(-4,4,0.01)
k_df = pd.DataFrame({
    'x' : k
})

for i in range(2,21):
    k_df[f"x^{i}"] = k_df['x']**i

k_df= k_df.sort_values('x')


model = Lasso(alpha = 0.03)
model.fit(train_x, train_y)
y_pred = model.predict(train_x)

line_pred = model.predict(k_df)

plt.scatter(valid_x, valid_y, color='blue')
plt.plot(k_df['x'], line_pred, color='red')





for i in range(2, 21):
    valid_df[f"x^{i}"] = valid_df["x"] ** i
    
# 'x' 열을 포함하여 'x2'부터 'x20'까지 선택.
valid_x = valid_df[["x"] + [f"x^{i}" for i in range(2, 21)]]
valid_y = valid_df["y"]


k = np.arange(-4,4,0.01)
k_df = pd.DataFrame({
    'x' : k
})

for i in range(2,21):
    k_df[f"x^{i}"] = k_df['x']**i

k_df= k_df.sort_values('x')


model = Lasso(alpha = 0.03)
model.fit(valid_x, valid_y)
y_pred = model.predict(valid_x)

line_pred = model.predict(k_df)

plt.scatter(valid_x['x'], valid_y, color='blue')
plt.plot(k_df['x'], line_pred, color='red')



# -------------------------------------------------
# k-fold
import seaborn as sns
import pandas as pd
import pandas as pd

np.random.seed(2024)
x = uniform.rvs(size=30, loc=-4, scale=8)
y = np.sin(x) + norm.rvs(size=30, loc=0, scale=0.3)

df = pd.DataFrame({
    "y" : y,
    "x" : x
})
df

index = np.random.choice(np.arange(30),30, replace=False)
fold1 = index[:6]
fold2 = index[6:12]
fold3 = index[12:18]
fold4 = index[18:24]
fold5 = index[24:]

for i in range(2, 21):
    df[f"x^{i}"] = df["x"] ** i

# fold1 = validation
valid1 = df.loc[fold1]
train1 = df.drop(fold1)

valid1_x = valid1.drop(columns = ('y'))
valid1_y = valid1['y']
train1_x = train1.drop(columns= ('y'))
train1_y = train1['y']


train_result = []
valid_result = []
lambda_result = []

for i in np.arange(1000):
    model = Lasso(alpha = i*(0.01))
    model.fit(train1_x, train1_y)

    # 모델 성능
    y_hat_train = model.predict(train1_x)
    y_hat_val = model.predict(valid1_x)

    perf_train = sum((train1_y - y_hat_train)**2)
    perf_valid = sum((valid1_y - y_hat_val)**2)
    train_result.append(perf_train) 
    valid_result.append(perf_valid)
    lambda_result.append(i*(0.01))



perf_df = pd.DataFrame({
    'lambda_result' : lambda_result,
    'train_result' : train_result,
    'valid_result' : valid_result
})




# alpha를 0.01으로 선택!
np.argmin(valid_result)  # 최소값의 위치 인덱스 알아봐줌
perf_df['lambda_result'][np.argmin(valid_result)]  # lambda 값

perf_df['valid_result'].min()


alpha001_perf = []
alpha001_perf.append(perf_df['valid_result'].min())

good_lambda = [0.01]


# fold2
valid2 = df.loc[fold2]
train2 = df.drop(fold2)

valid2_x = valid2.drop(columns = ('y'))
valid2_y = valid2['y']
train2_x = train2.drop(columns= ('y'))
train2_y = train2['y']


train_result = []
valid_result = []
lambda_result = []

for i in np.arange(1000):
    model = Lasso(alpha = i*(0.01))
    model.fit(train2_x, train2_y)

    # 모델 성능
    y_hat_train = model.predict(train2_x)
    y_hat_val = model.predict(valid2_x)

    perf_train = sum((train2_y - y_hat_train)**2)
    perf_valid = sum((valid2_y - y_hat_val)**2)
    train_result.append(perf_train) 
    valid_result.append(perf_valid)
    lambda_result.append(i*(0.01))



perf_df2 = pd.DataFrame({
    'lambda_result' : lambda_result,
    'train_result' : train_result,
    'valid_result' : valid_result
})


# alpha를 0.05으로 선택!
np.argmin(valid_result)  # 최소값의 위치 인덱스 알아봐줌

perf_df2['valid_result'].min()
perf_df2['lambda_result'][np.argmin(valid_result)]

alpha005_perf = [0]
alpha005_perf.append(perf_df2['valid_result'].min())

good_lambda.append(0.05)


# fold3
valid3 = df.loc[fold3]
train3 = df.drop(fold3)

valid3_x = valid3.drop(columns = ('y'))
valid3_y = valid3['y']
train3_x = train3.drop(columns= ('y'))
train3_y = train3['y']


train_result = []
valid_result = []
lambda_result = []

for i in np.arange(1000):
    model = Lasso(alpha = i*(0.01))
    model.fit(train3_x, train3_y)

    # 모델 성능
    y_hat_train = model.predict(train3_x)
    y_hat_val = model.predict(valid3_x)

    perf_train = sum((train3_y - y_hat_train)**2)
    perf_valid = sum((valid3_y - y_hat_val)**2)
    train_result.append(perf_train) 
    valid_result.append(perf_valid)
    lambda_result.append(i*(0.01))



perf_df3 = pd.DataFrame({
    'lambda_result' : lambda_result,
    'train_result' : train_result,
    'valid_result' : valid_result
})


# alpha를 0.02으로 선택!
np.argmin(valid_result)  # 최소값의 위치 인덱스 알아봐줌

perf_df3['valid_result'].min()
perf_df3['lambda_result'][np.argmin(valid_result)]

alpha002_perf =[0,0]
alpha002_perf.append(perf_df3['valid_result'].min())

good_lambda.append(0.02)





# fold4
valid4 = df.loc[fold4]
train4 = df.drop(fold4)

valid4_x = valid4.drop(columns = ('y'))
valid4_y = valid4['y']
train4_x = train4.drop(columns= ('y'))
train4_y = train4['y']


train_result = []
valid_result = []
lambda_result = []

for i in np.arange(1000):
    model = Lasso(alpha = i*(0.01))
    model.fit(train4_x, train4_y)

    # 모델 성능
    y_hat_train = model.predict(train4_x)
    y_hat_val = model.predict(valid4_x)

    perf_train = sum((train4_y - y_hat_train)**2)
    perf_valid = sum((valid4_y - y_hat_val)**2)
    train_result.append(perf_train) 
    valid_result.append(perf_valid)
    lambda_result.append(i*(0.01))



perf_df4 = pd.DataFrame({
    'lambda_result' : lambda_result,
    'train_result' : train_result,
    'valid_result' : valid_result
})


# alpha를 0으로 선택!
np.argmin(valid_result)  # 최소값의 위치 인덱스 알아봐줌

perf_df4['valid_result'].min()
perf_df4['lambda_result'][np.argmin(valid_result)]

alpha0_perf =[0,0,0]
alpha0_perf.append(perf_df4['valid_result'].min())

good_lambda.append(0)



# fold5
valid5 = df.loc[fold5]
train5 = df.drop(fold5)

valid5_x = valid5.drop(columns = ('y'))
valid5_y = valid5['y']
train5_x = train5.drop(columns= ('y'))
train5_y = train5['y']


train_result = []
valid_result = []
lambda_result = []

for i in np.arange(1000):
    model = Lasso(alpha = i*(0.01))
    model.fit(train5_x, train5_y)

    # 모델 성능
    y_hat_train = model.predict(train5_x)
    y_hat_val = model.predict(valid5_x)

    perf_train = sum((train5_y - y_hat_train)**2)
    perf_valid = sum((valid5_y - y_hat_val)**2)
    train_result.append(perf_train) 
    valid_result.append(perf_valid)
    lambda_result.append(i*(0.01))



perf_df5 = pd.DataFrame({
    'lambda_result' : lambda_result,
    'train_result' : train_result,
    'valid_result' : valid_result
})


# alpha를 0으로 선택!
np.argmin(valid_result)  # 최소값의 위치 인덱스 알아봐줌

perf_df5['valid_result'].min()
perf_df5['lambda_result'][np.argmin(valid_result)]


alpha0_perf.append(perf_df5['valid_result'].min())

good_lambda.append(0)




good_lambda


# fold1의 0.05일 때 성능
model = Lasso(alpha = 0.05)
model.fit(train1_x, train1_y)

# 모델 성능
# y_hat_train = model.predict(train1_x)
y_hat_val = model.predict(valid1_x)

#perf_train = sum((train1_y - y_hat_train)**2)
perf_valid = sum((valid1_y - y_hat_val)**2)
# train_result.append(perf_train) 
perf_valid

alpha005_perf[0] = perf_valid


# fold3의 0.05일 때 성능
model = Lasso(alpha = 0.05)
model.fit(train3_x, train3_y)

# 모델 성능
# y_hat_train = model.predict(train1_x)
y_hat_val = model.predict(valid3_x)

#perf_train = sum((train1_y - y_hat_train)**2)
perf_valid = sum((valid3_y - y_hat_val)**2)
# train_result.append(perf_train) 
perf_valid

alpha005_perf.append(perf_valid)


# fold4 의 0.05 성능
model = Lasso(alpha = 0.05)
model.fit(train4_x, train4_y)

# 모델 성능
# y_hat_train = model.predict(train1_x)
y_hat_val = model.predict(valid4_x)

#perf_train = sum((train1_y - y_hat_train)**2)
perf_valid = sum((valid4_y - y_hat_val)**2)
# train_result.append(perf_train) 
perf_valid
alpha005_perf.append(perf_valid)


# fold5 의 0.05 성능
model = Lasso(alpha = 0.05)
model.fit(train5_x, train5_y)

# 모델 성능
# y_hat_train = model.predict(train1_x)
y_hat_val = model.predict(valid5_x)

#perf_train = sum((train1_y - y_hat_train)**2)
perf_valid = sum((valid5_y - y_hat_val)**2)
# train_result.append(perf_train) 
perf_valid
alpha005_perf.append(perf_valid)


good_lambda

# fold2의 0.01 성능
model = Lasso(alpha = 0.01)
model.fit(train2_x, train2_y)

# 모델 성능
# y_hat_train = model.predict(train1_x)
y_hat_val = model.predict(valid2_x)

#perf_train = sum((train1_y - y_hat_train)**2)
perf_valid = sum((valid2_y - y_hat_val)**2)
# train_result.append(perf_train) 
perf_valid
alpha001_perf.append(perf_valid)


# fold3의 0.01 성능
model = Lasso(alpha = 0.01)
model.fit(train3_x, train3_y)

# 모델 성능
# y_hat_train = model.predict(train1_x)
y_hat_val = model.predict(valid3_x)

#perf_train = sum((train1_y - y_hat_train)**2)
perf_valid = sum((valid3_y - y_hat_val)**2)
# train_result.append(perf_train) 
perf_valid
alpha001_perf.append(perf_valid)


# fold4의 0.01 성능
model = Lasso(alpha = 0.01)
model.fit(train4_x, train4_y)

# 모델 성능
# y_hat_train = model.predict(train1_x)
y_hat_val = model.predict(valid4_x)

#perf_train = sum((train1_y - y_hat_train)**2)
perf_valid = sum((valid4_y - y_hat_val)**2)
# train_result.append(perf_train) 
perf_valid
alpha001_perf.append(perf_valid)



# fold5의 0.01 성능
model = Lasso(alpha = 0.01)
model.fit(train5_x, train5_y)

# 모델 성능
# y_hat_train = model.predict(train1_x)
y_hat_val = model.predict(valid5_x)

#perf_train = sum((train1_y - y_hat_train)**2)
perf_valid = sum((valid5_y - y_hat_val)**2)
# train_result.append(perf_train) 
perf_valid
alpha001_perf.append(perf_valid)



good_lambda

# fold1의 0.02 성능
model = Lasso(alpha = 0.02)
model.fit(train1_x, train1_y)

# 모델 성능
# y_hat_train = model.predict(train1_x)
y_hat_val = model.predict(valid1_x)

#perf_train = sum((train1_y - y_hat_train)**2)
perf_valid = sum((valid1_y - y_hat_val)**2)
# train_result.append(perf_train) 
perf_valid
alpha002_perf[0] = (perf_valid)


# fold2의 0.02 성능
model = Lasso(alpha = 0.02)
model.fit(train2_x, train2_y)

# 모델 성능
# y_hat_train = model.predict(train1_x)
y_hat_val = model.predict(valid2_x)

#perf_train = sum((train1_y - y_hat_train)**2)
perf_valid = sum((valid2_y - y_hat_val)**2)
# train_result.append(perf_train) 
perf_valid
alpha002_perf[1] = (perf_valid)


# fold4의 0.02 성능
model = Lasso(alpha = 0.02)
model.fit(train4_x, train4_y)

# 모델 성능
# y_hat_train = model.predict(train1_x)
y_hat_val = model.predict(valid4_x)

#perf_train = sum((train1_y - y_hat_train)**2)
perf_valid = sum((valid4_y - y_hat_val)**2)
# train_result.append(perf_train) 
perf_valid
alpha002_perf.append(perf_valid)


# fold5의 0.02 성능
model = Lasso(alpha = 0.02)
model.fit(train5_x, train5_y)

# 모델 성능
# y_hat_train = model.predict(train1_x)
y_hat_val = model.predict(valid5_x)

#perf_train = sum((train1_y - y_hat_train)**2)
perf_valid = sum((valid5_y - y_hat_val)**2)
# train_result.append(perf_train) 
perf_valid
alpha002_perf.append(perf_valid)



# fold1의 0 성능
model = Lasso(alpha = 0)
model.fit(train1_x, train1_y)

# 모델 성능
# y_hat_train = model.predict(train1_x)
y_hat_val = model.predict(valid1_x)

#perf_train = sum((train1_y - y_hat_train)**2)
perf_valid = sum((valid1_y - y_hat_val)**2)
# train_result.append(perf_train) 
perf_valid
alpha0_perf[0] = perf_valid



# fold2의 0 성능
model = Lasso(alpha = 0)
model.fit(train2_x, train2_y)

# 모델 성능
# y_hat_train = model.predict(train1_x)
y_hat_val = model.predict(valid2_x)

#perf_train = sum((train1_y - y_hat_train)**2)
perf_valid = sum((valid2_y - y_hat_val)**2)
# train_result.append(perf_train) 
perf_valid
alpha0_perf[1] = perf_valid


# fold3의 0 성능
model = Lasso(alpha = 0)
model.fit(train3_x, train3_y)

# 모델 성능
# y_hat_train = model.predict(train1_x)
y_hat_val = model.predict(valid3_x)

#perf_train = sum((train1_y - y_hat_train)**2)
perf_valid = sum((valid3_y - y_hat_val)**2)
# train_result.append(perf_train) 
perf_valid
alpha0_perf[2] = perf_valid


good_lambda

whole_perf_df = pd.DataFrame({
    'good_lambda' : good_lambda,
    'alpha000' : alpha0_perf,
    'alpha001' : alpha001_perf,
    'alpha002' : alpha002_perf,
    'alpha005' : alpha005_perf,
})


print("람다 0 의 평균 성능", whole_perf_df['alpha000'].mean())   # 0의 성능이 더 좋음
print("람다 0.01 의 평균 성능", whole_perf_df['alpha001'].mean())
print("람다 0.02 의 평균 성능", whole_perf_df['alpha002'].mean())
print("람다 0.05 의 평균 성능", whole_perf_df['alpha005'].mean())  









# -------------------------------------------------
# k-fold
import seaborn as sns
import pandas as pd
import pandas as pd

np.random.seed(2024)
x = uniform.rvs(size=30, loc=-4, scale=8)
y = np.sin(x) + norm.rvs(size=30, loc=0, scale=0.3)

df = pd.DataFrame({
    "y" : y,
    "x" : x
})
df

for i in range(2, 21):
    df[f"x^{i}"] = df["x"] ** i


index = np.random.choice(30,30, replace=False)
fold1 = index[:10]
fold2 = index[10:20]
fold3 = index[20:30]

df.loc[index[0:10]]
df.drop(index[0:10])

fold_num = 0
fold_num = 1

def make_tr_val(fold_num, df, cv_num=3):  # cv_num이 def 정의에 사용되지 않아서 흐릿한 색으로 나옴. 그리고 기본값 3으로 설정되어 있어서, def 사용할 때 cv 입력 안해도 3으로 입력됨.
    np.random.seed(2024)
    myindex = np.random.choice(30,30,replace=False)

    val_index = myindex[(10*fold_num) : (10*fold_num+10)]

    valid_set = df.loc[val_index]
    train_set = df.drop(val_index)

    train_x = train_set.iloc[:, 1:]
    train_y = train_set.iloc[:,0]

    valid_x = valid_set.iloc[:, 1:]
    valid_y = valid_set.iloc[:, 0]

    return (train_x, train_y, valid_x, valid_y)



valid_result_total = np.repeat(0.0, 3000).reshape(3,-1)  # 값을 받을 matrix 만듦
train_result_total = np.repeat(0.0, 3000).reshape(3,-1)  # 값을 받을 matrix 만듦

for j in np.arange(0,3):
    train_x, train_y, valid_x, valid_y = make_tr_val(fold_num = j, df=df)

    train_result = []
    valid_result = []
    lambda_result = []

    for i in np.arange(1000):
        model = Lasso(alpha = i*0.01)
        model.fit(train_x, train_y)

        # 모델 성능
        y_hat_train = model.predict(train_x)
        y_hat_val = model.predict(valid_x)

        perf_train = sum((train_y - y_hat_train)**2)
        perf_valid = sum((valid_y - y_hat_val)**2)
        train_result.append(perf_train) 
        valid_result.append(perf_valid)
        lambda_result.append(i*0.01)

    train_result_total[j,:] = train_result
    valid_result_total[j,:] = valid_result


df = pd.DataFrame({
    'lambda' : lambda_result,
    'train' : train_result_total.mean(axis=0),
    'valid' : valid_result_total.mean(axis=0)
})


sns.scatterplot(data=df, x='lambda', y='train')
sns.scatterplot(data=df, x='lambda', y='valid', color='red')
plt.xlim(0, 0.4)   # lambda 범위가 0.4보다 클 듯 함

plt.xlim(0,10)
plt.ylim()


val_result[0]
val_result[1]
np.min(val_result)

# alpha를 0.03로 선택!
np.argmin(val_result)
np.arange(0, 1, 0.01)[np.argmin(val_result)]


model= Lasso(alpha=0.03)
model.fit(train_x, train_y)
model.coef_
model.intercept_
# model.predict(test_x)

k=np.linspace(-4, 4, 80)

k_df = pd.DataFrame({
    "x" : k
})

for i in range(2, 21):
    k_df[f"x{i}"] = k_df["x"] ** i
    
k_df

















# --------- 강사님 코드


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import norm
from scipy.stats import uniform
from sklearn.linear_model import LinearRegression

# 20차 모델 성능을 알아보자
np.random.seed(2024)
x = uniform.rvs(size=30, loc=-4, scale=8)
y = np.sin(x) + norm.rvs(size=30, loc=0, scale=0.3)

df = pd.DataFrame({
    "y" : y,
    "x" : x
})

for i in range(2, 21):
    df[f"x{i}"] = df["x"] ** i

df

def make_tr_val(fold_num, df):
    np.random.seed(2024)
    myindex=np.random.choice(30, 30, replace=False)

    # valid index
    val_index=myindex[(10*fold_num):(10*fold_num+10)]

    # valid set, train set
    valid_set=df.loc[val_index]
    train_set=df.drop(val_index)

    train_X=train_set.iloc[:,1:]
    train_y=train_set.iloc[:,0]

    valid_X=valid_set.iloc[:,1:]
    valid_y=valid_set.iloc[:,0]

    return (train_X, train_y, valid_X, valid_y)


from sklearn.linear_model import Lasso

val_result_total=np.repeat(0.0, 3000).reshape(3, -1)
tr_result_total=np.repeat(0.0, 3000).reshape(3, -1)

for j in np.arange(0, 3):
    train_X, train_y, valid_X, valid_y = make_tr_val(fold_num=j, df=df)

    # 결과 받기 위한 벡터 만들기
    val_result=np.repeat(0.0, 1000)
    tr_result=np.repeat(0.0, 1000)

    for i in np.arange(0, 1000):
        model= Lasso(alpha=i*0.01)
        model.fit(train_X, train_y)

        # 모델 성능
        y_hat_train = model.predict(train_X)
        y_hat_val = model.predict(valid_X)

        perf_train=sum((train_y - y_hat_train)**2)
        perf_val=sum((valid_y - y_hat_val)**2)
        tr_result[i]=perf_train
        val_result[i]=perf_val

    tr_result_total[j,:]=tr_result
    val_result_total[j,:]=val_result


import seaborn as sns

df = pd.DataFrame({
    'lambda': np.arange(0, 10, 0.01), 
    'tr': tr_result_total.mean(axis=0),
    'val': val_result_total.mean(axis=0)
})

df['tr']

# seaborn을 사용하여 산점도 그리기
# sns.scatterplot(data=df, x='lambda', y='tr')
sns.scatterplot(data=df, x='lambda', y='val', color='red')
plt.xlim(0, 10)

# alpha를 2.67로 선택!
np.argmin(val_result_total.mean(axis=0))
np.arange(0, 10, 0.01)[np.argmin(val_result_total.mean(axis=0))]


# 함수 만들기

def make_train_valid(fold_num, df, cv_num=3):  # cv_num이 def 정의에 사용되지 않아서 흐릿한 색으로 나옴. 그리고 기본값 3으로 설정되어 있어서, def 사용할 때 cv 입력 안해도 3으로 입력됨.
    np.random.seed(2024)
    index = np.random.choice(len(df),len(df),replace=False)
    cut_n = int(len(df)/cv_num)
    for i in cv_num:
        if i != cv_num:
            f"val{i}_index" = index[ cut_n * i : cut_n(i+1)]
        elif i == cv_num:
            f"val{i}_index" = index[ cut_n * i : ]


    valid_set = df.loc[val_index]
    train_set = df.drop(val_index)

    train_x = train_set.iloc[:, 1:]
    train_y = train_set.iloc[:,0]

    valid_x = valid_set.iloc[:, 1:]
    valid_y = valid_set.iloc[:, 0]

    return (train_x, train_y, valid_x, valid_y)



# 데이터가 적으면 오버피팅의 가능성이 높아진다. 따라서 라쏘 분석을 하면 람다 값이 커져서(오버피팅 방지를 위해 패널티를 빡세게 주기 위해서) 변수를 많이 지운다.






# 0828 수업
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import norm, uniform
from sklearn.linear_model import Lasso
from sklearn.model_selection import cross_val_score, KFold
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
from sklearn.metrics import make_scorer, mean_squared_error

# 데이터 생성
np.random.seed(2024)
x = uniform.rvs(size=30, loc=-4, scale=8)
y = np.sin(x) + norm.rvs(size=30, loc=0, scale=0.3)

# 데이터를 DataFrame으로 변환하고 다항 특징 추가
x_vars = np.char.add('x', np.arange(1, 21).astype(str))
X = pd.DataFrame(x, columns=['x'])
poly = PolynomialFeatures(degree=20, include_bias=False)  # include_bias : X행렬에서 맨 앞에 1을 붙이냐 안 붙이냐 그거임
                                                    # degree : 몇 차까지 계산할거냐
X_poly = poly.fit_transform(X)
X_poly=pd.DataFrame(
    data=X_poly,
    columns=x_vars
)

# 교차 검증 설정
kf = KFold(n_splits=3, shuffle=True, random_state=2024)

# 알파 값 설정
alpha_values = np.arange(0, 10, 0.01)

# 각 알파 값에 대한 교차 검증 점수 저장
mean_scores = []

for alpha in alpha_values:
    lasso = Lasso(alpha=alpha, max_iter=5000)  # minimize 할 때 5000번 반복해서 더 정확하게 구해라 라는 것임
    scores = cross_val_score(lasso, X_poly, y, cv=kf, scoring='neg_mean_squared_error')  # scoring : score 계산하는 방법
                                                            # neg_mean_squared_error : MSE에 마이너스 붙인 것 : 원래는 낮은 값이 높은 성능인건데, 높은 값이 높은 성능이다라고 하고 싶어서
    mean_scores.append(np.mean(scores))

# 결과를 DataFrame으로 저장
df = pd.DataFrame({
    'lambda': alpha_values,
    'validation_error': mean_scores
})

# 결과 시각화  # y축 값이 마이너스가 붙긴 했지만, 그래프에서도 작은 값이 좋은 건 똑같음. 가장 밑에 있는 값을 찾아라
plt.plot(df['lambda'], df['validation_error'], label='Validation Error', color='red')
plt.xlabel('Lambda')
plt.ylabel('Mean Squared Error')
plt.legend()
plt.title('Lasso Regression Train vs Validation Error')
plt.show()

# 최적의 alpha 값 찾기
optimal_alpha = df['lambda'][np.argmin(df['validation_error'])]
print("Optimal lambda:", optimal_alpha)




# 08/29 수업
# y = (x-2)^2 +1 그래프 그리기
import matplotlib.pyplot as plt
import numpy as np

x = np.linspace(-4, 8, 100)
y = (x-2)**2 + 1
plt.plot(x, y, color='black')
plt.xlim(-4, 8)
plt.ylim(-1,20)

# y=4x-11 그래프
line_y = 4*x -11
plt.plot(x, line_y, color='red')


# f'(x)=2x-4
# k=4의 기울기
k=4
l_slope = 2*k -4
f_k = (k-2)**2 + 1
l_intercept = f_k - l_slope*k


# 미분값의 의미2 : 함수값이 커지는 방향을 알려줌.
# x=4에서 미분값이 4가 나왔다면 오른쪽 방향으로 함수값이 커진다는 것임
# x=-4에서 미분값이 -2가 나왔다면 왼쪽방향으로 함수값이 커진다는 것임.

# 함수값이 최소값인 곳을 찾으려면 미분계수값이 알려주는 방향의 반대방향으로 가야 됨. 


# Q. y = x^2
# 초기값 : 10, 델타:0.9
# y' = 2x
k=10
for i in range(100):
    k = k -0.9 * 2 * k

print(k)


k=10
lstep = np.arange(100, 0 , -1)*0.01
for i in range(100):


for i in range(3):
    print(i)




import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# x, y의 값을 정의합니다 (-1에서 7까지)
x = np.linspace(-1, 7, 400)
y = np.linspace(-1, 7, 400)
x, y = np.meshgrid(x, y)   # x와 y의 조합을 계산해줌. (순서쌍) 400*400 = 160000개


# 함수 f(x, y)를 계산합니다.
z = (x - 3)**2 + (y - 4)**2 + 3

# 그래프를 그리기 위한 설정
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# 표면 그래프를 그립니다.
ax.plot_surface(x, y, z, cmap='viridis')

# 레이블 및 타이틀 설정
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('f(x, y)')
ax.set_title('Graph of f(x, y) = (x-3)^2 + (y-4)^2 + 3')

# 그래프 표시
plt.show()

# ==========================
# 등고선 그래프
import numpy as np
import matplotlib.pyplot as plt

# x, y의 값을 정의합니다 (-1에서 7까지)
x = np.linspace(-10, 10, 400)
y = np.linspace(-10, 10, 400)
x, y = np.meshgrid(x, y)

# 함수 f(x, y)를 계산합니다.
z = (x - 3)**2 + (y - 4)**2 + 3

# 등고선 그래프를 그립니다.
plt.figure()
cp = plt.contour(x, y, z, levels=20)  # levels는 등고선의 개수를 조절합니다.
plt.colorbar(cp)  # 등고선 레벨 값에 대한 컬러바를 추가합니다.

# 특정 점 (9, 2)에 파란색 점을 표시
plt.scatter(9, 2, color='red', s=50)

x=9; y=2
lstep=0.1

for i in range(100):
    x , y = np.array([x, y]) - lstep* np.array([2*x-6, 2*y-8])
    plt.scatter(float(x), float(y), color ='red', s= 25)

print(x,y)  # 어디서 멈췄는지 알 수 있음


# 축 레이블 및 타이틀 설정
plt.xlabel('X')
plt.ylabel('Y')
plt.title('Contour Plot of f(x, y) = (x-3)^2 + (y-4)^2 + 3')
plt.xlim(-10, 10)
plt.ylim(-10, 10)

# 그래프 표시
plt.show()








import numpy as np
import matplotlib.pyplot as plt


# x, y의 값을 정의합니다 (-1에서 7까지)
beta0 = np.linspace(-20, 20, 600)
beta1 = np.linspace(-20, 20, 600)
beta0, beta1 = np.meshgrid(beta0, beta1)

# 함수 f(x, y)를 계산합니다.
z =  (1 - (beta0 + beta1))**2 + \
    (4 - (beta0 + 2 * beta1))**2 + \
    (1.5 - (beta0 + 3 * beta1))**2 + \
    (5 - (beta0 + 4 * beta1))**2

# 등고선 그래프를 그립니다.
plt.figure()
cp = plt.contour(beta0, beta1, z, levels=20)  # levels는 등고선의 개수를 조절합니다.
plt.colorbar(cp)  # 등고선 레벨 값에 대한 컬러바를 추가합니다.



beta0 = 10 ; beta1 = 10
lstep=0.01
i=2
for i in range(1000):
    beta0, beta1 = np.array([beta0, beta1]) - lstep* np.array([-23+8*beta0+20*beta1, -67+20*beta0+60*beta1])
    plt.scatter(float(beta0), float(beta1), color ='red', s= 25)

print(beta0, beta1)  # 어디서 멈췄는지 알 수 있음 0.5000337122517691 0.949988533723301 에서 멈춤
                    # 즉 , 0.5+0.95x = y 회귀직선으로 가까워짐

# 축 레이블 및 타이틀 설정
plt.xlabel('X')
plt.ylabel('Y')
plt.title('Contour Plot of f(beta0, beta1) ')


# 그래프 표시
plt.show()



# 모델 fit으로 베타 구하기
import pandas as pd
from sklearn.linear_model import LinearRegression

df = pd.DataFrame({
    'x' : np.array([1,2,3,4]),
    'y' : np.array([1,4,1.5,5])
})

model = LinearRegression()
model.fit(df[['x']], df['y'])
model.coef_
model.intercept_   # 앞서 찾은 값과 동일함


# 최소 값을 찾았는데도 찾는 횟수(for문)이 남았다면 비효율적이게 계속 돌아가는거임
# 그래서 이전에 찾은 값과 방금 찾은 값이 같으면 찾는 걸 중단하는 early stopping을 이용함.



# k 최근접 모델 : k nearest neighborhood <- bagging 할 때 다른 모델들 결과와 함께 쓰면 예측 성능이 더 좋아질 수 있음.
# # 근접한 애들 몇명만 볼 것 인가가 하이퍼파라미터임. : k
# 거리를 구하기 위해서 계산이 오래걸림 (obs1에 대해서 나머지 obs들과의 거리를 하나하나 다 구해야함)
# 비슷한 애들을 보고 예측. 비슷하다 = 거리가 가깝다
# 상대적으로 거리가 가깝다는 거라서, 거리 값 자체는 멀어도 다른 애들보다 가까운면 그걸로 예측됨.
# 거리1 : sqrt(a^2 + b^2)  , 거리2 : a + b

df = pd.DataFrame({
    'y' : [13,15,17,15,16],
    'x1' : [16, 20, 22, 18, 17],
    'x2' : [7, 7, 5, 6, 7]
})

(df['x1'] - 15)**2 + (df['x2'] - 5.6)**2  # 1번 펭귄
np.abs(df['x1']-15) + np.abs(df['x2'] - 5.6)  # 1번 펭귄


knn_regressor = KNeighborsRegressor(n_neighbors=3)
knn_regressor.fit(X_train_scaled, y_train)




# 08/30 수업
def g(x=3):
    result= x+1
    return result

print(g)  # 함수에 대해서 알고 싶은데 안 나옴, 근데 g쪽에 마우스를 가져다대면 함수 정보 뜸

import inspect
print(inspect.getsource(g))   # 함수 내용 출력해줌


def calculation_1


class DataFrma  # <- Cammel 식




# if문  한 줄
x=3
y=1 if x>4 else 2
y


# 리스트 컴리헨션
x = [1, -2, 3, -4, 5]
result = ['양수' if value>0 else "음수" for value in x]
result

result = ['성공' if value > 0 else '실패' for value in [-1,5,-6, 2]]


import numpy as np

x = np.array([1, -2, 3 ,-4, 0])
conditions = [
    x>0, x==0, x<0
]
choices = ['양수', '0', '음수']
result = np.select(conditions, choices, x)
result


names = ["john","alice"]
ages = np.array([25,30])
greetings = [f"이름: {name}, 나이: {age}." for name, age in zip(names, ages)]

zipped = zip(names, ages)  # zip() 함수로 names와 ages를 병렬적으로 묶음

# 각 튜플을 출력
for name , age in zipped:
    print(f"Name: {name}, Age: {age}")


# while 문
i = 0
while i <= 10:
    i += 3
    print(i)


# while, break 문
i = 0
while True:
    i += 3
    if i > 10:
        break   # 그만 돌아감
    print(i)


import pandas as pd 
data = pd.DataFrame({
    'A' : [1,2,3],
    'B' : [4,5,6]
})

data.apply(max, axis=0)
data.apply(max, axis=1)


def my_func(x, const=3):
    return max(x)**2 + const

my_func([3,4,10], 5)

data.apply(my_func, axis=0)
data.apply(my_func, axis=0, const=5)



import numpy as np 

array_2d = np.arange(1, 13).reshape((3,4), order='F')
print(array_2d)


np.apply_along_axis(max, axis=0, arr=array_2d)



y=2 
def my_func(x):
    y=1
    result = x+y
    return result

my_func(3)
print(y)



def my_func(x):
    global y
    y= y+1
    result = x + y
    return result

my_func(3)
print(y)   # 함수 돌릴 때마다 y값이 업데이트 됨.



def my_func(x):
    global y

    def my_f(k):   # my_f(k)는 my_func가 정의된 공간 (함수 안에서) 정의된거라서 global에서 my_f를 접근할 수 없ㅏ.
        return k**2

    y = my_f(x) + 1
    result = x + y

    return result


my_f(3)  # 정의되지 않음
my_func(3)
print(y)


# 입력값이 몇 개일지 모를 땐 *를 앞에 붙인다 , 인수값을 하나하나씩 돌아간에 
def add_many(*args):
    result = 0
    for i in args:
        result = result + i
    return result


add_many(1, 2, 3)   # args는 값을 리스트처엄 받


def first_many(*args):
    return args[0]

first_many(1,2,3)
first_many(4,1,2,3)


def add_mul(choice, *args):  # *만 붙이면 args 상관없이 다른 단어 써도 됨.
    if choice == 'add':
        result = 0
        for i in args:
            result = result + i
    elif choice == 'mul':
        result = 1
        for i in args:
            result = result * i
    return result

add_mul("add",5,4,3,1)

add_mul("mul",5,4,3,1)


## 별표 두개 (**)는 입력값을 딕셔너리로 만들어줌
def my_twostar(choice, **kwargs):
    if choice == 'first':
        return print(kwargs[0])
    elif choice == 'second':
        return print(kwargs[1])
    else:
        return print(kwargs)
    
my_twostar('first', age=30, name='issac')
my_twostar('second', age=30, name='issac')



import numpy as np
x = np.array([2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12])
px = np.array([1/36 , 2/36, 3/36, 4/36 , 5/36, 6/36, 5/36 , 4/36, 3/36, 2/36 , 1/36])
x_bar = sum(x*px)
print("E[x] :", sum(x*px))

print("Var[x] :", sum((x-x_bar)**2*px))

print("E[2*x+3] :", sum(x*px)*2+3)

print("Std[2*x+3] :", np.sqrt(sum((x-x_bar)**2*px)*4))



from scipy.stats import norm, chi2
1 - norm.cdf(24, loc=30, scale=4)



norm.cdf((29.7-30)/(4/np.sqrt(8)), loc=0, scale=1) - norm.cdf((28-30)/(4/np.sqrt(8)), loc=0, scale=1)
# 0.33735241076117556

norm.cdf(29.7, loc=30, scale=4/np.sqrt(8)) - norm.cdf(28, loc=30, scale=4/np.sqrt(8))
# 0.33735241076117556


<<<<<<< HEAD
from scipy.stats import chi2, chisquare
import matplotlib.pyplot as plt
import numpy as np
?chi2.pdf

x = np.linspace(0,20,100)
y = chi2.pdf(x, df=7)
plt.plot(x, y)
plt
=======

x = np.linspace(0,20,100)
y = chi2.pdf(x, df=7)

?chi2.pdf

import matplotlib.pyplot as plt
plt.plot(x, y)
plt.show()
>>>>>>> f6225cb6866db1dbc484cf29f80d50d2e032083a



# 귀무가설 : 두 변수가 독립이다
# 대립가설 : 두 변수가 독립이 아니다 
mat_a = np.array([14,4,0,10]).reshape(2,2)
mat_a

from scipy.stats import chi2, chi2_contingency
<<<<<<< HEAD
chi2_value, p_value, df, expected = chi2_contingency(mat_a, correction=False)  # 해당 함수는 4개의 결과값을 알려줌
                                                            # correction = Fasle로 해야 우리가 일반적으로 아는 카이제곱 검정통계량이 나옴. correction = True는 연속형 어쩌구임.
=======
chi2_value, p_value, df, expected = chi2_contingency(mat_a)  # 해당 함수는 4개의 결과값을 알려줌
>>>>>>> f6225cb6866db1dbc484cf29f80d50d2e032083a
chi2_value.round(3)  # 검정통계량
p_value.round(4)  # p-value 0.0004
# 유의수준 0.05 하에 p-value가 0.05보다 작으므로, 귀무가설을 기각
# 즉, 두 변수는 독립이 아니다. (두 변수 간의 영향이 있다. 우연이 아니다)

# x ~ chi2(1)일 때, p(x > 12.6) 
1 - chi2.cdf(12.6, df=1)


<<<<<<< HEAD
np.sum((mat_a - expected)**2 / expected)  # 검정통계량 값이 맞음
1 - chi2.cdf(15.556, df=1)  # p-value 값임 


mat_b = np.array([[50, 30, 20], [45, 35, 20]])
chi2, p ,df , expected = chi2_contingency(mat_b, correction=False)
chi2.round(3)
p.round(4)  # 귀무가설 기각을 못함 동질적이지 않다고 하지 못함.
expected



# 귀무가설 : 정당 지지와 핸드폰 사용 유무는 독립이다.
# 대립가설 : 정당 지지와 핸드폰 사용 유무는 독립이 아니다.
# 주의 ) 모든 칸이 5 이상인지 확인해봐야 함.
mat_c = np.array([[49,47],[15,27],[32,30]])
chi2_value, p_value, df, expected = chi2_contingency(mat_c, correction=False)
chi2_value
p_value   # 0.2 > 0.05 이므로 귀무가설 기각 못함. 독립이라고 할 수 없음.







from scipy.stats import chisquare
import numpy as np

observed = np.array([13,23,24,20,27,18,15])
expected = np.repeat(20, 7)
statistic, p_value = chisquare(observed, f_exp=expected)  # 적합도 할 때는 chisquare 함수 이용해야 하고, 행렬로 검정할 때는 chi2 함수 이용해야 함

statistic.round(4)
p_value   # 0.27 > 0.05 이므로 귀무가설 기각 못함. 요일 별 신생아 출생 비율엔 차이가 없다. 
1- chi2.cdf(7.6, df=6)





mat_d = np.array([[176, 124], [193, 107], [159, 141]])
chi2, p, df, expected = chi2_contingency(mat_d)
expected




# 09/04 수업
import pandas as pd
import numpy as np
from palmerpenguins import load_penguins

penguins = load_penguins()
penguins.head()

df_w = penguins.dropna()
df_w = df_w[['bill_length_mm', 'bill_depth_mm']]
df_w = df_w.rename(columns = {'bill_length_mm': 'y','bill_depth_mm':'x' })

n1 = df.query("x < 15").shape[0]
n2 = df.query("x >= 15").shape[0]

# 1번 그룸 (x<15)은 얼마로 예측할까?
y_hat1 = df.query("x < 15").mean()[0]
y_hat2 = df.query("x >= 15").mean()[0]


# 각 그룹 mse는 얼마?
mse1 = np.mean((df.query("x < 15")['y'] - y_hat1)**2)
mse2 = np.mean((df.query("x >= 15")['y'] - y_hat2)**2)


# 기준 x=15의 mse 기중평균은? 
(mse1 * n1 + mse2 * n2) / (n1+n2)


# 원래 mse는?
np.mean((df['y'] - df['y'].mean())**2)


# 기준값 x를 넣으면 mse값이 나오는 함수는?
def mse1(h,k1):
    penguins = load_penguins()
    penguins.head()

    df = penguins.dropna()
    df = df[['bill_length_mm', 'bill_depth_mm']]
    df = df.rename(columns = {'bill_length_mm': 'y','bill_depth_mm':'x' })

    df1 = df[df['x']<h]
    df2 = df[df['x'] >= h]


    n11 = df1.query(f"x < {k1}").shape[0]
    n12 = df1.query(f"x >= {k1}").shape[0]


    # 1번 그룸 (x<n)은 얼마로 예측할까?
    y_hat11 = df.query(f"x < {k1}").mean()[0]
    y_hat12 = df.query(f"x >= {k1}").mean()[0]



    # 각 그룹 mse는 얼마?
    mse11 = np.mean((df.query(f"x < {k1}")['y'] - y_hat11)**2)
    mse12 = np.mean((df.query(f"x >= {k1}")['y'] - y_hat12)**2)


    # 기준 x=n의 mse 기중평균은? 
    return (mse11 * n11 + mse12 * n12) / (n11+n12) 



def mse2(h,k2):
    penguins = load_penguins()
    penguins.head()

    df = penguins.dropna()
    df = df[['bill_length_mm', 'bill_depth_mm']]
    df = df.rename(columns = {'bill_length_mm': 'y','bill_depth_mm':'x' })

    df1 = df[df['x']<h]
    df2 = df[df['x'] >= h]


    n21 = df2.query(f"x < {k2}").shape[0]
    n22 = df2.query(f"x >= {k2}").shape[0]

    # 1번 그룸 (x<n)은 얼마로 예측할까?
    y_hat21 = df.query(f"x < {k2}").mean()[0]
    y_hat22 = df.query(f"x >= {k2}").mean()[0]


    # 각 그룹 mse는 얼마?
    mse21 = np.mean((df.query(f"x < {k2}")['y'] - y_hat21)**2)
    mse22 = np.mean((df.query(f"x >= {k2}")['y'] - y_hat22)**2)


    # 기준 x=n의 mse 기중평균은? 
    return (mse21 * n21 + mse22 * n22) / (n21+n22)










mse(n=20)




# minimize를 이용해서 가장 작은 mse가 나오는 x 찾기
from scipy.optimize import minimize

initial_guess = [15]
result = minimize(mse, initial_guess)
print("최소값 : ", result.fun)
print("최소값을 갖는 x값 :", result.x)

df['x'].min()
df['x'].max()

x_values = np.arange(start=13.2, stop=21.4, step = 0.01)
result2 = []
for i in range(len(x_values)):
    result2.append(mse(x_values[i]))

result2[np.argmin(result2)]
x_values[np.argmin(result2)]  # 16.41





a = [print(2*x) for x in [1,2,3]]
a 











(16.4-13.2)/0.01


# 그룹1 함수 정의
def mse1(h,k1):
    penguins = load_penguins()
    penguins.head()

    df = penguins.dropna()
    df = df[['bill_length_mm', 'bill_depth_mm']]
    df = df.rename(columns = {'bill_length_mm': 'y','bill_depth_mm':'x' })

    df1 = df[df['x']<h]


    n11 = df1.query(f"x < {k1}").shape[0]
    n12 = df1.query(f"x >= {k1}").shape[0]


    # 1번 그룸 (x<n)은 얼마로 예측할까?
    y_hat11 = df1.query(f"x < {k1}").mean()[0]
    y_hat12 = df1.query(f"x >= {k1}").mean()[0]



    # 각 그룹 mse는 얼마?
    mse11 = np.mean((df1.query(f"x < {k1}")['y'] - y_hat11)**2)
    mse12 = np.mean((df1.query(f"x >= {k1}")['y'] - y_hat12)**2)


    # 기준 x=n의 mse 기중평균은? 
    return (mse11 * n11 + mse12 * n12) / (n11+n12) 



# 그룹2 함수 정의
def mse2(h,k2):
    penguins = load_penguins()
    penguins.head()

    df = penguins.dropna()
    df = df[['bill_length_mm', 'bill_depth_mm']]
    df = df.rename(columns = {'bill_length_mm': 'y','bill_depth_mm':'x' })

    h = 16.41
    k2 = 16.43

    df2 = df[df['x'] >= h]

    
    n21 = df2.query(f"x < {k2}").shape[0]
    n22 = df2.query(f"x >= {k2}").shape[0]

    # 1번 그룸 (x<n)은 얼마로 예측할까?
    y_hat21 = df2.query(f"x < {k2}").mean()[0]
    y_hat22 = df2.query(f"x >= {k2}").mean()[0]


    # 각 그룹 mse는 얼마?
    mse21 = np.mean((df2.query(f"x < {k2}")['y'] - y_hat21)**2)
    mse22 = np.mean((df2.query(f"x >= {k2}")['y'] - y_hat22)**2)


    # 기준 x=n의 mse 기중평균은? 
    return (mse21 * n21 + mse22 * n22) / (n21+n22)




x_values1 = np.arange(start=df[df['x']<16.42]['x'].min() , stop=df[df['x']<16.42]['x'].max() , step=0.01)[1:-1]
result2 = []
for i in range(len(x_values1)):
    result2.append(mse1(16.41, k1 = x_values1[i] ))

x_values2 = np.arange(start=df[df['x']>=16.42]['x'].min() , stop=df[df['x']>=16.42]['x'].max() , step=0.01)[1:-1]
result3 = []
for i in range(len(x_values2)):
    result3.append(mse2(16.41, k2 = x_values2[i] ))



result2[np.argmin(result2)]
x_values1[np.argmin(result2)]  # 14.01
result3[np.argmin(result3)]
x_values2[np.argmin(result3)]  # 19.4

import matplotlib.pyplot as plt

plt.scatter(df_w['x'], df_w['y'])
plt.axvlines(x=[16.41, 14.01, 16.42], red='color')

plt.axvline(x=[16.41], red='color')
plt.show()


mse(h=20)
mse()


# 전체 함수 만들기 도전
# MSE 계산 함수 정의
def mse(k, df):
    penguins = load_penguins()

    df = penguins.dropna()
    df = df[['bill_length_mm', 'bill_depth_mm']]
    df = df.rename(columns = {'bill_length_mm': 'y','bill_depth_mm':'x' })


    df1 = df.query(f'x < {k}')
    df2 = df.query(f'x >= {k}')
    
    n1 = len(df1)
    n2 = len(df2)

    yhat1 = df1["y"].mean()
    yhat2 = df2["y"].mean()

    MSE1 = ((df1["y"] - yhat1) ** 2).mean()
    MSE2 = ((df2["y"] - yhat2) ** 2).mean()
    return (MSE1 * n1 + MSE2 * n2) / (n1 + n2)


x_values1 = np.arange(start=df[df['x']<16.42]['x'].min() , stop=df[df['x']<16.42]['x'].max() , step=0.01)[1:-1]
result2 = []
for i in range(len(x_values1)):
    result2.append(mse(h=16.41, k1 = x_values1[i] ))

x_values2 = np.arange(start=16.41 , stop=21.4 , step=0.01)
result3 = []
for i in range(len(x_values2)):
    result3.append(mse(h=16.41, k2 = x_values2[i] ))



result2[np.argmin(result2)]
x_values1[np.argmin(result2)]  # 15.81
result3[np.argmin(result3)]
x_values2[np.argmin(result3)]  # 20.80



# 조건에 구간 등급 변수 만들기
np.where( < 1 , 'a', <2 , 'b')
=

df['group'] = np.digitize(df['x'], [14.01, 16.42, 19.4])  # 각 값을 경계로 해서 구간 0, 1, 2 중 어디에 해당하는 지 값을 지정해줌
y_mean = df.groupby('group').mean()['y']
k1 = np.linspace(13, 14.01, 100)
k2 = np.linspace(14.01, 16.42, 100)
k3 = np.linspace(16.42, 19.4, 100)
k4 = np.linspace(19.4, 22, 100)


plt.scatter(df['x'], df['y'])

plt.plot(k1, np.repeat(y_mean[0], 100), color='red')
plt.plot(k2, np.repeat(y_mean[1], 100), color='red')
plt.plot(k3, np.repeat(y_mean[2], 100), color='red')
plt.plot(k4, np.repeat(y_mean[3], 100), color='red')



# 의사결정 나무
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error, r2_score

# 랜덤 시드 고정 (재현성을 위해)
np.random.seed(42)

# x 값 생성: -10에서 10 사이의 100개 값
x = np.linspace(-10, 10, 100)

# y 값 생성: y = x^2 + 정규분포 노이즈
y = x ** 2 + np.random.normal(0, 10, size=x.shape)

# 데이터프레임 생성
df = pd.DataFrame({'x': x, 'y': y})

# 데이터 시각화
plt.scatter(df['x'], df['y'], label='Noisy Data')
plt.xlabel('x')
plt.ylabel('y')
plt.title('Noisy Quadratic Data')
plt.legend()
plt.show()

# 입력 변수와 출력 변수 분리
X = df[['x']]  # 독립 변수는 2차원 형태로 제공되어야 함
y = df['y']

# 학습 데이터와 테스트 데이터로 분할 (80% 학습, 20% 테스트)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 디시전 트리 회귀 모델 생성 및 학습
model = DecisionTreeRegressor(random_state=42, max_depth=4, )
model.fit(X_train, y_train)

# 테스트 데이터에 대한 예측
y_pred = model.predict(X_test)

# 모델 평가
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print(f'Mean Squared Error: {mse}')
print(f'R^2 Score: {r2}')

# 테스트 데이터의 실제 값과 예측 값 시각화
plt.scatter(X_test, y_test, color='blue', label='Actual Values')
plt.scatter(X_test, y_pred, color='red', label='Predicted Values')
plt.xlabel('x')
plt.ylabel('y')
plt.title('Decision Tree Regression')
plt.legend()
plt.show()



from scipy.stats import binom
binom.cdf(3, n=10, p=0.6)
1- binom.cdf(3, n=10, p=0.3)
1- binom.cdf(3, n=10, p=0.4)



# 0905 수업



# 최적의 하이퍼파라미터 찾기
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import ElasticNet
model = ElasticNet(alpha=0.03, l1_ratio=0.2)  # alpha: 람다, l1_ratio : 알파
model.fit(train_x, train_y)


np.random.seed(20240905)

# elasticnet <- l1_ratio에 1과 0도 있어서 라쏘랑 릿지도 비교해줌
model = ElasticNet()  # alpha: 람다, l1_ratio : 알파

param_grid = {    # 하이퍼 파라미터 후보군들
    'alpha' : [0.1, 1.0, 10.0, 100.0],   
    'l1_ratio' : [0, 0.1, 0.5, 1.0]
}


grid_search = GridSearchCV(
    estimator = model,
    param_grid = param_grid,
    scoring = 'neg_mean_squared_error',
    cv=5
)

grid_search.fit(train_x, train_y)




# 라쏘
model = ElasticNet(l1_ratio=1)  # alpha: 람다, l1_ratio : 알파

param_grid = {    # 하이퍼 파라미터 후보군들
    'alpha' : [0.1, 1.0, 10.0, 100.0],   
}


grid_search = GridSearchCV(
    estimator = model,
    param_grid = param_grid,
    scoring = 'neg_mean_squared_error',
    cv=5
)

grid_search.fit(train_x, train_y)  # 최적의 하이퍼 파라미터 찾을 때 사용할 전체 데이터 셋 (여기서 알아서 train / valid 나눠서 하이퍼 파라미터 찾아줌)




model = Lasso()  # alpha: 람다, l1_ratio : 알파

param_grid = {    # 하이퍼 파라미터 후보군들
    'alpha' : [0.1, 1.0, 10.0, 100.0],   
}


grid_search = GridSearchCV(
    estimator = model,
    param_grid = param_grid,
    scoring = 'neg_mean_squared_error',
    cv=5
)

grid_search.fit(train_x, train_y)


# 여기에 계산된 성능을 볼 수 있게 df만드는 코드만 짜면 될 듯.
grid_search.best_params_
grid_search.cv_results_
grid_search.best_score_  # neg_mean_squared_error 라서 음수값 나올 거임. 그냥 음수 빼고 생각하면 됨.

best_model = grid_search.best_estimator_  # 최적의 모델을 자동으로 찾아줌
best_model.predict(test_y)  # 바로 최적의 모델로 예측할 수 있음.
=======

>>>>>>> f6225cb6866db1dbc484cf29f80d50d2e032083a
