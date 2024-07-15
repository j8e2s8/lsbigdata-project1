# 데이터 타입
x =15.34
print(x, "는", type(x), "형식입니다.", sep='') #sep은 입력값들 사이사이에 채워넣는 것. ''는 아무것도 넣지 마라는 것임.
print(x, "는", type(x), "형식입니다.", sep='.')

print(x, type(x))

fruit = ['apple','banana','cherry']
type(fruit)

mixed_list = [1,'hello', [1,2,3]]
type(mixed_list)


a=(10,20,30)
b=(42)
b
type(b)

c=(42,)
c
type(c)

b=10
b

c=10
c

d = [10,20,30]
d[1]
d[1] = 25
d

e = (10,20,30)
e[1]
e[1] =25

a=(10,20,30,40,50)
a[3:]
a[:3] # 마지막건 포함 아님.
a[1:3] # 첫 인덱스 포함, 마지막 인덱스 불포함



#사용자 정의함수
def min_max(numbers) :
   return min(numbers), max(numbers) #이렇게 결과를 반환해라
 
a=[1,2,3,4,5]
result = min_max(a)
result
print("Minimum and maximum", result)

type(result)  # 함수를 만들어서 했더니, 결과값이 튜블임.


person = {
  'name' : ['John', 'Anne'],
  'age' : (30,29),
  'city' : 'New York'
}

person  # 딕셔너리는 리스트, 튜플, 문자열, 수치다 요소로 가질 수 있음.

person.get('name')  # 딕셔너리.get('키')
person.get('age')[0]  # 딕셔너리.get('키')[인덱스]

age = person.get('age')
age
age[0]


fruit = {'apple','banana','cherry','apple'}
fruit
type(fruit)


empty_set = set()
empty_set
empty_set.add("apple")
empty_set

empty_set.add("banana")
empty_set.add("apple")
empty_set

empty_set.remove("banana")
empty_set

empty_set.remove("cherry")
empty_set.discard("cherry")
empty_set


p =True
q =False
print(p, type(p))


# 조건문
a=3
if (a==2):    # boolen이 참이 되면 첫번째 결과를 실행함
   print("a는 2와 같습니다.")
else:
  print("a는 2와 같지 않습니다.")
  
  
  
#타입 변환
num = 123
str_num= str(123)
type(str_num)

num_again = float(str_num)
num_again
type(num_again)

lst = [1,2,3]
type(lst)
tup = tuple(lst)
type(tup)
tup

set_example = {'a','b','c'}
dict_from_set={key:True for key in set_example}
dict_from_set



# 교재 63페이지

!pip install seaborn
import seaborn as sns
import matplotlib.pyplot as plt


var = ["a","a","b","c"]
var
seaborn.countplot(x=var)
plt.show()

df = sns.load_dataset('titanic')
df
seaborn.countplot(data=df, x='sex')
plt.show()  # 그냥하면 이전에 그렸던 그래프 정보가 남아있는 상태에서 그림 그려줌
plt.clf()

seaborn.countplot(data=df, x='class')
plt.show()
plt.clf()

sns.countplot(data=df, x='class', hue='alive')
plt.show()
plt.clf()

sns.countplot(data=df, x='class', hue='alive', palette='dark:red')
plt.show()
plt.clf()

sns.countplot(data=df, x='age', hue='alive', orient='v')
plt.show()
?sns.countplot

!pip install scikit-learn

import sklearn.metrics 


# 교재 4장
import pandas as pd

df_exam = pd.read_excel('excel_exam.xlsx')
df_exam
sum(df_exam['english']) / len(df_exam['english'])

import numpy as np

np.mean(df_exam['english'])

a='hello'
len(a)

df_exam_novar = pd.read_excel('excel_exam_novar.xlsx', header=None)
df_exam_novar

df=pd.read_csv('exam.csv')
df

df.info()

df.describe()
import matplotlib.pyplot as plt
plt.boxplot(df['english'])
plt.show()
plt.clf()

string(df['nclass'])
str(df['nclass'])

df.describe()
df['nclass'] = str(df['nclass'])
df.describe(include='all')

pd.set_option('display.max_columns', None)


mpg = pd.read_csv('mpg.csv')
mpg.head()
mpg.shape
mpg.describe()

mpg.info()

mpg.describe(include= 'all')


mpg = mpg.rename(columns={'cty':'city'})
mpg

mpg['total'] = (mpg['city'] + mpg['hwy'])/2
mpg.head()

mpg.mean()
mpg['city', 'hwy'].mean()
mpg['city'].mean()

mpg['total'].plot.hist()
plt.show()
plt.clf()

import pandas as pd
mpg = pd.read_csv('mpg.csv')
mpg['total'] = (mpg['cty'] + mpg['hwy'])/2
import numpy as np
mpg['test'] = np.where(mpg['total'] >= 20, 'pass', 'fail')
mpg['test'].value_counts()
mpg['cty'].value_counts()
mpg.value_counts()

import matplotlib.pyplot as plt
mpg['test'].plot.bar()
plt.show()
plt.clf()


count_test = mpg['test'].value_counts()
count_test
count_test.plot.bar(rot=0)
plt.show()

mpg['cty'].plot.hist()
plt.clf()
plt.show()

mpg['grade'] = np.where(mpg['total'] >= 30, 'A', np.where(mpg['total'] >= 20, 'B', 'C'))


mpg
mpg['grade'].value_counts()  # 자동으로 빈도수가 많은 애부터 정렬됨
mpg['grade'].value_counts().plot.bar(rot=0) # 빈도수가 많은 애부터 정렬
mpg['grade'].value_counts().sort_index().plot.bar(rot=0)  # 범주 인덱스 순으로 정렬
plt.show()
plt.clf()
import matplotlib.pyplot as plt
type(mpg['grade'].value_counts())



mpg['size'] = np.where(mpg['category'] == 'compact'
                     | mpg['category'] == 'subcompact'
                     | mpg['category'] == '2seater')
                     , 'small', 'large')











a=(1,2,3)
a

a=1,2,3  # 리스트보다 튜플이 가벼우니까 기본적으로 튜플로 저장함
a

def min_max(numbers):
  return min(numbers), max(numbers)

a=[1,2,3,4,5]
min_max(a)   # def 정의가 값, 값 이라서 튜플로 반환함


def min_max(numbers):
  return [min(numbers), max(numbers)]

a=[1,2,3,4,5]
min_max(a)  # def 정의를 [값,값] 이라서 리스트로 반환함


a =[1,2,3]
b=a  # soft copy
a[1]
a[1] = 4
a
b   # soft copy 한 애들은 같이 수정됨


id(a)
id(b)  # a와 b 주소가 같다는 것을 알 수 있음


b=a.copy()
b= a[:]  # 동일한 방법임

id(a)
id(b)

b[1]=7
a
b




import math
sqrt_val = math.sqrt(16)
print("16의 제곱근은:", sqrt_val)

exp_val = math.exp(5)
print("e^5의 값은:", exp_val)

log_val = math.log(10,10)
print("10의 밑 10 로그 값은:", log_val)

def normal_def(x, mu, sigma):
  sqrt_two_pi = math.sqrt(2*math.pi)
  factor =1/(sigma*sqrt_two_pi)
  return factor*math.exp(-0.5*((x-mu)/sigma)**2)

def my_normal_pdf(x, mu, sigma):
  part_1 =1/(sigma*math.sqrt(2*math.pi))
  part_2 = math.exp( (-(x-mu)**2) / (2 * sigma**2))
  return part_1 * part_2
my_normal_pdf(3,3,1)


def f(x,y,z):
  return (x**2 + math.sqrt(y)+ math.sin(z)) * math.exp(x)

f(2,9,math.pi/2)

def my_g(x):
  return math.cos(x) + math.sin(x) * math.exp(x)
my_g(math.pi)

def fname(`indent('.') ? 'self' : ''`):
    """docstring for fname"""
    # TODO: write code...   
# 4day

def     (input):
    contents
    return
    

import pandas as pd
import numpy as np


# ctrl+shift+c : 주석처리


a = np.array([1,2,3,4,5])
b = np.array(["apple", "banana", "orange"])
c = np.array([True, False, True, True])

type(a)
a[3]
a[2:]
a[1:4]

b = np.empty(3)  # 빈 배열 만들기
b
b[0] = 1   # 실수로 저장됨
b[1] = 4
b[2] = 10
b
b[2]

vec1=np.array([1,2,3,4,5])
vec1=np.arange(100)  # 0이상 99미만에서 1단위로 숫자 만들기
vec1=np.arange(1,101, 0.5) # 0이상 101미만에서 0.5 단위로 숫자 만들기
vec1

l_space1 = np.linspace (0,1,5)  #0이상 1이하, 숫자 5개 만들어줘
l_space1

linear_space2 = np.linspace(0,1,5, endpoint=False) #0이상 1미만, 숫자 5개 만들어줘
linear_space2

?np.linspace

vec1 = np.arange(5)
type(vec1)
np.repeat(vec1,5)

#-100부터 0까지 
vec2 = np.arange(-100,1)
vec2

vec3 = np.arange(0, -100) # 이렇게는 안됨
vec3


vec4 = np.arange(0, -101, -1) # 간격을 음수로 하면 됨
vec4


# repeat vs tile
vec1 = np.arange(5)
np.repeat(vec1, 3)  # 원소마다 반복
np.tile(vec1,3)   # 전체적으로 반복

vec1 + vec1

vect = np.array([1,2,3,4])
vec1+vec1


max(vec1)
min(vec1)

# 35672이하 홀수들의 합은?
vec1 = np.arange(1,35673, 2)
vec1
sum(vec1)


np.arange(1,35673,2).sum()  # array의 메서드로 sum()이 존재함.
# 가독성을 위해서 .이 많으면 가독성이 좋지 않음, 가독성을 위해서 
x=np.arange(1,35673,2)
x.sum()


len(x)
x.shape  # 튜플 형태로 반환함

b=np.array([[1,2,3],[4,5,6]])
len(b)   # 원소가 2개니까 2.
b.shape   # 행렬로 본 것임. (2,3)
b.size   # 전체 값의 갯수


a=np.array([1,2])
b=np.array([1,2,3,4]) 
a+b  # 길이가 달라서 계산 안됨

np.tile(a,2) + b  # 길이가 같아서 계산 가능
np.repeat(a,2) +b

b
b == 3  # b의 각 원소들이랑 == 여부 비교하는 것임.

# 35672보다 작은 수 중에서 7로 나눠서 나머지가 3인 숫자들의 갯수는?
c= np.arange(0,35672) % 7
c == 3
len(c)
sum(c == 3)  # True인 갯수

sum(np.linspace(1,35671, 35671) % 7 == 3)  # True, False를 이용해서 해당하는 값의 갯수를 구할 수 있음.
# 벡터는 문자열로만, 숫자로만 가능함. 리스트는 섞어서도 가능한데 벡터는 안됨



# 7/15 수업
import numpy as np
a=np.array([0.1,2.0,3.0])
b = 2.0
a*b

a.shape
b.shape  # shape이 존재하지 않음. 숫자 1개니까

# 하나는 shape이 존재하고, 하나는 shape이 존재하지 않을 때 브로드캐스팅이 된다.


#2차원 배열 생성
matrix = np.array([[0.0, 0.0, 0.0],
                  [10.0, 10.0, 10.0],
                  [20.0, 20.0, 20.0],
                  [30.0, 30.0, 30.0]])
                  
matrix.shape

vector = np.array([1.0,2.0,3.0])   # (4,3) + (3,) 가능
vector.shape
result = matrix + vector
result
# 원소의 첫원소끼리 더하기, 원소의 두번째 원소끼리 더하기


vector = np.array([1.0, 2.0, 3.0, 4.0]).reshape(4,1)  # (4,3) + (4,1) 가능
vector
vector.shape
result = matrix + vector
result


vector = np.array([1.0, 2.0, 3.0, 4.0]).reshape(1,4)  # (4,3) + (1,4) 불가능. 차원이 같아야지 연산이 가능함.
vector
vector.shape
result = matrix + vector
result




#백터 슬라이싱 예제, a를 랜덤하게 채움

a = np.random.randint(1,21,10)  # 값 10개 생성, 돌릴 때마다 값 달라짐
a
b = np.random.randint(1,21)  # 값 1개만 생성
b

np.random.seed(42)   
a = np.random.randint(1,21,10)  # 값 10개 생성, 돌릴 때마다 값 동일함. 누가 어느 컴퓨터로 돌리든 동일함
a 

np.random.seed(2024)   # 올해 연도를 사용하면 편리함
a = np.random.randint(1,21,10)  # 값 10개 생성, 돌릴 때마다 값 동일함. 누가 어느 컴퓨터로 돌리든 동일함
a 



?np.random.seed
?np.random.randint

a[2:5]
a[-2]  # 맨 끝에서 두번째
a[::2]  # [첫 값 : 끝 값 : 스텝]  # 얘는 전체에서 스텝 2로 2씩 띄어서 추출임.
a[1:6:2]

#1에서부터 1000사이 3의 배수의 합은?
a=np.arange(1,1001)
a=np.arange(3,1001,3)
sum(a)

x = np.arange(3,1001)
x[::3]
sum(x[::3])


np.random.seed(2024)   # 올해 연도를 사용하면 편리함
a = np.random.randint(1,21,10)  # 값 10개 생성, 돌릴 때마다 값 동일함. 누가 어느 컴퓨터로 돌리든 동일함
a 
print(a[[0,2,4]])

np.delete(a, [1,3])



# 필터링
a>3
b= a[a>3]  # 인덱스 대신에 비교연산으로 인한 True, False로 True 값만 인덱싱할 수 있음 (필터링)
b


np.random.seed(2024)
a= np.random.randint(1, 10000, 300)  # 동일한 변수명 a로, 동일한 random.seed(2024)로 벡터를 여러번 생성했는데, 문장만 동일하면 항상 값이 동일하게 나옴.
a
a<5000
a[a<5000]


np.random.seed(2024)
a= np.random.randint(1, 10000, 5)
a
(a>2000) & (a<5000)
a[(a>2000) & (a<5000)]


import pydataset

df=pydataset.data('mtcars')
np_df = np.array(df['mpg'])  # df의 'mpg' 칼럼을 벡터로 불러옴.
np_df

# 15이상 25이하 데이터 갯수는?
np_df[(np_df>=15) & (np_df<=25)]
sum((np_df>=15) & (np_df<=25))

# 평균 mpg 보다 높은(이상) 자동차 대수는?
sum(np.mean(np_df)<=np_df)


#15보다 작거나 22이상인 데이터 갯수는?
sum((np_df < 15) | (np_df>=22))


np.random.seed(2024)
a = np.random.randint(1,1000,5)
b = np.array(["A","B","C","F","W"])
a[(a>600) & (a<650)]
(a>600) & (a<650)
b[(a>600) & (a<650)]  #a에 대한 True, False를 b에 적용해서 불러오겠다는 것임.


model_names = np.array(df.index)  #df.index가 자동차 모델 이름임.
model_names
model_names[(np_df>=15) & (np_df<=25)]
model_names[(np_df < 15) | (np_df>=22)]


# 연비가 좋은 애들만 뽑아보기
model_names[np_df >= np.mean(np_df)]
model_names[np_df < np.mean(np_df)]

df
df['mpg'][df['mpg']>30]



# 필터링한 결과에 값을 바꿀 수 있음
a
a[a<600] = 0
a

np.random.seed(2024)
a = np.random.randint(1,100,10)
a
np.where(a<50)  # True의 위치를 반환하는 함수

# 처음으로 22000보다 큰 숫자가 나왔을 때, 그 숫자의 위치와 그 숫자는 무엇인가요?
np.random.seed(2024)
a = np.random.randint(1,26346,1000)
a
a[np.where(a>22000)][0]
np.where(a>22000)

type(np.where(a>22000))  # 값을 1개만 가지는 튜플임
np.where(a>22000)[0]
type(np.where(a>22000)[0])
np.where(a>22000)[0][0]
type(np.where(a>22000)[0][0])


my_index=np.where(a>22000)[0][0]
a[my_index]

# 처음으로 24000보다 큰 숫자 나왔을 때, 그 숫자와 위치는 무엇인가요?
a
a[np.where(a>24000)][0]

np.where(a>24000)[0][0]
a[np.where(a>24000)[0][0]]

# 처음으로 10000보다 큰 숫자 나왔을 때, 50번째로 나오는 그 숫자와 위치는 무엇인가요?
a
a[np.where(a>10000)][49]

np.where(a>10000)[0][49]
a[np.where(a>10000)[0][49]]


# 500보다 작은 숫자들 중 가장 마지막으로 나오는 숫자 위치와 그 숫자는 무엇인가요?
np.where(a<500)[0][-1]
a[np.where(a<500)[0][-1]]


# np.nan로 빈칸을 표현 할 수 있다.
a= np.array([20, np.nan, 13, 24, 309])
a+3 
np.mean(a)   # nan을 무시 못해서, 평균 결과가 nan이 나옴
np.nanmean(a)  # nan을 무시하고 평균값 구해줌.

np.nan_to_num(a)  # nan을 0으로 바꿔줌.
?np.nan_to_num
np.nan_to_num(a, nan=1)  # 원하는 값으로 nan을 바꿔줌.


c = np.array([1, "A", True])
c
type(c)

c = np.empty(3)
c

c = np.arange(1, 50, 2)
c
type(c)
c.shape


a = None  # 변수 초기화
a
b = np.nan   # nan이라는 값을 가짐
b
a+1
b+1

a= np.array([20, np.nan, 13, 14])
np.isnan(a)

~np.isnan(a)
a_filtered = a[~np.isnan(a)]
a_filtered


str_vec = np.array(["사과","배","수박","참외"])
str_vec[[0,2]]
mix_vec = np.array(["사과", 12, "수박", "참외"])
mix_vec = np.array(["사과", 12, "수박", "참외"], dtype=str)  #dtype 설정가능
mix_vec
np.concatenate((str_vec, mix_vec))
np.concatenate([str_vec, mix_vec])


col_stacked = np.column_stack((np.arange(1,5),
                               np.arange(12,16)))   # 열기준으로 2줄 생김
col_stacked

row_stacked = np.row_stack((np.arange(1,5),
                            np.arange(12,16)))   # 행기준으로 2줄 생김. 근데 곧 지원하지 않을 거라고 경고 뜸.
row_stacked



uneven_stacked = np.column_stack((np.arange(1,5),
                                  np.arange(12,18)))
uneven_stacked



vec1 = np.arange(1,5)
vec2 = np.arange(12,18)

uneven_stacked = np.column_stack((vec1, vec2))

vec1 = np.resize(vec1, len(vec2))
vec1
uneven_stacked = np.column_stack((vec1, vec2))
uneven_stacked


# 홀수번째 원소
a = np.array([12,21,35,48,5])
a[0::2]
a[1::2]

# 최대값 찾기
a = np.array([1,22,93,64,54])
a.max()

a = np.array([21,31,58])
b = np.array([24,44,67])
a
b

#
c = np.empty(6)
c[1::2] = b
c[0::2] = a
c

import pandas as pd
df=pd.DataFrame({'name' : ['김지훈','이유진','박동현','김민지'],
              'english' : [90, 80 , 60, 70],
              'math' : [50, 60, 100, 20]})
type(df)         

df = pd.DataFrame({'제품' : ["사과","딸기","수박"],
                   '가격' : [1800, 1500, 3000],
                   '판매량' : [24,38,13]})
                   
df
df['가격'].mean()
df['판매량'].mean()

np.repeat(1, 3)
