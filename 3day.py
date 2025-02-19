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

03087b3

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

import numpy as np
a = np.array([1,2,3])
np.repeat(a, repeats = [1,2,3])
np.repeat([1,2,3], 4)
np.repeat(1,3)

np.tile(1,3)
np.tile([1,2,3], 4)
np.tile(a, 2)
np.tile(a, repeats=[2,3,4])

a
len(a)
b=np.array([1,2,3,4])
len(b)
b.size()
b.size

c=np.array([(1,2,3),(4,5,6)])
c=np.array([[1,2,3],[4,5,6]])
type(c)
len(c)
c.size
c.shape

a = np.array([1,2,3])
b = np.array([4,5,6])
a+b
b-a
a*b
b/a
c=2
a+c
z
a

a=np.array([[1,2,3],
           [4,5,6],
           [7,8,9]])
b=np.array([1,2,3])
a+b

a = np.array([1,2,3,4])
a.reshape(4,1)



df=pd.read_excel("data/excel_exam.xlsx")
df

df["math"]
df["english"]
df["science"]


df=pd.read_excel("data/excel_exam.xlsx", sheet_name='Sheet2')
df

df['total'] = df['math'] + df['english'] + df['science']
df
df['mean'] = (df['math'] + df['english'] + df['science'])/3
df

df[df['math']>50]
df[(df['math']>50) & (df['english']>50)]
df1 = df[(df['math']>np.mean(df['math'])) & (df['english']<np.mean(df['english']))]

df1[df1['nclass']==3]
df[df['nclass']==3][['math','english','science']]
df[1:4]
df[1:2]
df[1:10:2]  # 데이터 프레임도 인덱싱 규칙 적용 가능함. 행에 적용됨.
df.sort_values("math")
df.sort_values("math", ascending=False)
df.sort_values(["nclass","math"], ascending=[True,False])


np.where(a>3, "Up", "Down")
df['updown'] = np.where(df["math"]>50, "Up", "Down")
df

a= 1+2j
type(a)


import pandas as pd
df=pd.read_excel("data/excel_exam.xlsx", sheet_name='Sheet1')
df

df.query('nclass==1')
df[df['nclass']==1]
df.query('nclass!=1')
df[df['nclass']!=1]

df.query('english>90')
df.query('nclass in [1,2]')

df
df['english'] > 80
sum(df['english'] > 80)
type(df[['math','english']])

df[['math']]
type(df[['math']])
df.drop(columns = ['math', 'english'])
df[df['math']>=80]\
     ['english']\
     .head()
     
df.assign(total = df['math'] + df['english'] + df['science']
         , mean = lambda x: x['total']/3)
         
mean([1,2,3])



# 7/16
import pandas as pd
import numpy as np

exam=pd.read_csv('data/exam.csv')
exam.head()

# 데이터 탐색 함수
# head()
# tail(
# shape
# info()
# describe()


exam.head(10)
exam.tail(10)
exam.shape

exam2=exam.copy()
exam2=exam2.rename(columns={'nclass' : 'class'})

exam2['total'] = exam2['math'] + exam2['english'] + exam2['science']
exam2['test'] = np.where(exam2['total']>= 200, 'pass', 'fail')
exam2


exam2['test'].value_counts().plot.bar()
import matplotlib.pyplot as plt
plt.show()

?pandas.DataFrame.plot.bar

import numpy as np
exam2['test2'] = np.where(exam2['total']>= 200, 'A', np.where(exam2['total'] >= 100, 'B', 'C'))
exam2

exam2['test2'].isin(["A", "C"])  # 'A'인 것만 True로 , 아닌 건 False로



a=np.random.randint(1,21,10)
?np.random.randint


a=np.random.choice(np.arange(1,4), 100, True, np.array([2/5,2/5,1/5]))  # 처음 들어가는 범위가 np.arange여야 함. 생성 수 갯수, 중복 여부, 확률은 array로 넣기
a
sum(a==3)


# 데이터 전처리 함수
# query()
# df[]
# sort_values()
# groupby()
# assign()
# agg()
# merge()
# concat()

exam = pd.read_csv("data/exam.csv")
exam.query("nclass==1")

exam.query("math>50")
exam.query("math<50")
exam.query("english>=50")
exam.query("english<=80")
exam.query("nclass == 1 & math >= 50")
exam.query("nclass == 2 & english >= 80")
exam.query("math >= 90 | english >= 90")
exam.query("english<90 | science<50")
exam.query('nclass==1 | nclass==3 | nclass==5')
exam.query('nclass in [1,3,5]')
exam.query('nclass not in [1,3,5]')
exam[~exam['nclass'].isin([1,3,5])]

exam['nclass']
exam.drop(columns = ['math','english'])


exam = exam.assign(
    total=exam['math'] + exam['english'] + exam['science']
    ,mean = lambda x: x['total']/3
    ) \
 .sort_values('total', ascending = False)
exam2

exam2.groupby('class') \
    .agg(mean_math = ('math', 'mean')
        , mean_english = ('english', 'mean')
        , mean_science = ('science' , 'mean'))


df = pd.read_csv('mpg.csv')
df
df.query('category == "suv"') \
    .assign(total=(df['hwy'] + df['cty'])/2) \
    .groupby('manufacturer') \
    .agg(mean_tot = ('total', 'mean')) \ 
    .sort_values('mean_tot', ascending= False) \
    .head()







# 프로젝트
market = pd.read_excel('traditionalmarket.xlsx')
market

market.info()
market.describe()
market.head()
market1 = market.copy()
market1=market1.rename(columns={'시장명' : 'market_name',
                              '시장유형' : 'type',
                              '소재지도로명주소' : 'address_new',
                              '시장개설주기' : 'open_period',
                              '소재지지번주소' : 'address_old',
                              '점포수' : 'market_count',
                              '사용가능상품권' : 'certificate',
                              '공중화장실 보유여부' : 'public_toilet',
                              '주차장 보유여부' : 'parking_lot',
                              '개설년도' : 'year' ,
                              '데이터기준일자' : 'data_date'})
market1
market1['size'] = np.where(market1['count'] < 50 , 'small', np.where(market1['count']>=134, 'large', 'medium'))
market1


market1['level'] = np.where((market1['public_toilet'] == 'Y') & (market1['parking_lot'] == 'Y'), 'high', np.where((market1['public_toilet'] == 'N') & (market1['public_toilet'] == 'N') , 'low', 'intermediate'))
market1

import matplotlib.pyplot as plt
plt.clf()
market1['level'].value_counts().plot.bar(rot=0)
plt.show()

market1.groupby('type').agg(certificate_counts = ('certificate', 'count') )




# 아영
import pandas as pd
import numpy as np

market = pd.read_excel("data/traditionalmarket.xlsx")
market2 = market.copy()


# 변수명 바꾸기
market2 = market2.rename(columns = {"시장명" : "market_name", 
                                    "시장유형" : "type",
                                    "소재지도로명주소" : "open_period",
                                    "소재지지번주소" : "address_old",
                                    "점포수" : "market_count",
                                    "사용가능상품권" : "certificate",
                                    "공중화장실 보유여부" :"public_toilet",
                                    "주차장 보유여부" : "parking_lot",
                                    "개설년도" : "year",
                                    "데이터기준일자" : "data_date"})
market2.describe()
market2["public_toilet"].info()
market2["parking_lot"].info()

market2['market_count'].describe()
market2 = market2.assign(market_scale = np.where(market2["market_count"] >= 134, "large", 
                                        np.where(market2["market_count"] >= 50, "medium", "small")))
market2['level'] = np.where((market2['public_toilet'] == 'Y') & (market2['parking_lot'] == 'Y'), 'high', 
                   np.where((market2['public_toilet'] == 'N') & (market2['public_toilet'] == 'N') , 'low', 'intermediate'))
market2.head()
import seaborn as sns
import matplotlib.pyplot as plt
sns.countplot(data = market2, x = 'level', hue = 'level')
plt.show()
df = market2.groupby(["type", "market_scale"], as_index = False) \
            .agg(market_count = ("market_scale", "count"))
df.sort_values('market_count', ascending = False)




# hw2
# 자동차 배기량에 따라 고속도로 연비가 다른지 알아보려고 합니다. 
# displ(배기량)이 4 이하인 자동차와 5 이상인 자동차 중 어떤 자동차의 hwy(고속도로 연비) 평균이 더 높은지 알아보세요.
import pandas as pd
import numpy as np
df = pd.read_csv('mpg.csv')
pd.set_option('display.max_columns', None)
df
df.assign(type = np.where(df['displ']<=4, '4이하', np.where(df['displ']>=5, '5이상', '기타'))).groupby('type').agg(type_mean = ('hwy','mean'))


# 4이하 가 5이상보다 평균이 높다는 것을 알 수 있다.

#다른 방법
df.query('displ <= 4')['hwy'].mean()
df.query('displ >=5')['hwy'].mean()


# 자동차 제조 회사에 따라 도시 연비가 어떻게 다른지 알아보려고 합니다. 'audi'와 'toyota' 중 어느 manufacturer(자동차 제조 회사)의 cty(도시연비) 평균이 더 높은지 알아보세요.
df.groupby('manufacturer', as_index=False).agg(cty_mean = ('cty', 'mean')).query("manufacturer == 'audi' | manufacturer=='toyota'")
# toyota 제조회사가 도시연비 평균이 더 높다.
# 다른 방법
df.groupby('manufacturer', as_index=False).agg(cty_mean = ('cty', 'mean')).query("manufacturer in ['audi','toyota']")


# 'chevrolet', 'ford', 'honda' 자동차의 고속도로 연비 평균을 알아보려고 합니다. 세 회사의 데이터를 추출한 다음 hwy 전체 평균을 구해 보세요.
df.query("manufacturer in ['chevrolet','ford','honda']")['hwy'].mean()

# 'audi'에서 생산한 자동차 중에 어떤 자동차 모델의 hwy(고속도로 연비)가 높은지 알아보려고 합니다.
# 'audi'에서 생산한 자동차 중 hwy가 1~5위에 해당하는 자동차의 데이터를 출력하세요.
df.query('manufacturer == "audi"').groupby('model').agg(hwy_mean = ('hwy' , 'mean'))
df.query('manufacturer == "audi"').sort_values('hwy', ascending=False).head(5)

# mpg 데이터 복사본을 만들고, cty와 hwy를 더한 '합산 연비 변수'를 추가하세요.
mpg = df.copy()
mpg= mpg.assign(sum_cty_hwy = mpg['cty']+mpg['hwy'])
mpg

# 앞에서 만든 '합산 연비 변수'를 2로 나눠 '평균 연비 변수'를 추가하세요.
mpg = mpg.assign(avg_cty_hwy = mpg['sum_cty_hwy']/2)
mpg


# '평균 연비 변수'가 가장 높은 자동차 3종의 데이터를 출력하세요.
mpg.sort_values('avg_cty_hwy', ascending=False).head(3)

# 1~3번 문제를 해결할 수 있는 하나로 연결된 pandas 구문을 만들어 실행해 보세요. 데이터는 복사본 대신 mpg 원본을 이용하세요.
df.assign(sum_cty_hwy = df['cty']+df['hwy']
          , avg_cty_hwy = lambda x: x['sum_cty_hwy']/2) \
    .sort_values('avg_cty_hwy', ascending=False) \
    .head(3)
    
    



# 교재 6.7
import pandas as pd
test1 = pd.DataFrame({'id'       : [1,2,3,4,5]
                     , 'midterm' : [60,80,70,90,85]})
test2 = pd.DataFrame({'id'       : [1,2,3,4,5]
                     , 'midterm' : [70,83,65,95,80]})
                     
total = pd.merge(test1, test2, how = 'left', on = 'id')
total

import numpy as np

test1['midterm'][2] = np.nan
test1['id'][4] = np.nan
test1
test1.isna().sum()
test1.dropna(subset='midterm')
test1
test1.loc[[0,1],['midterm']]
test1.loc[0,'midterm']


test1 = pd.DataFrame({'id'       : [1,2,3,4,5,'a']
                     , 'midterm' : ['a','b','c','d','e',1]})
test1['midterm'] = np.where(test1['midterm'] == 'e' , np.nan, test1['midterm'])
test1
test1.isna()

test1['test'] = np.where(test1['midterm'] == 'e' , 'good', np.nan)

test1.isna()
test1

test1['midterm'].quantile(.25)
test1['midterm'].quantile(.50)
test1['midterm'].quantile(.75)


test1['midterm'] = np.where(test1['midterm']<=60 , 'a', np.nan)


df = pd.DataFrame({'x1' : [1,1,2,2]})
df['x2'] = np.where(df['x1']==1, 'a', 'etc')
df
df['x2'] = df['x2'].replace('etc', np.nan)
df
test1
test1.replace('a', 'etc')


test1 = pd.DataFrame({'id'        : [1,2,3,4,5]
                      , 'midterm' : [60,80,70,90,85]})
test2 = pd.DataFrame({'id'      : [1,2,3,4,5]
                      , 'final' : [70,83,65,95,80]})
                      
                      
test = pd.merge(test1, test2, how='left' , on='id')
test

test2 = test2.rename(columns=({'final' : 'medterm'}))
test2
total = pd.concat([test1,test2])
total


name = pd.DataFrame({'nclass' : [1,2,3,4,5]
                   , 'teacher' : ['kim', 'lee', 'park', 'choi', 'jung']})
                   name

df = pd.read_csv('exam.csv')                   
df
df1 = pd.merge(df, name, how='left' , on='nclass')
df1


score1 = pd.DataFrame({'id'        : [1,2,3,4,5]
                      , 'score' : [60,80,70,90,85]})
score2 = pd.DataFrame({'id'      : [6,7,8,9,10]
                      , 'score' : [70,83,65,95,80]})
                      
score = pd.concat([score1, score2])
score


test1 = pd.DataFrame({'id'        : [1,2,3,4,5]
                      , 'midterm' : [60,80,70,90,85]})
test2 = pd.DataFrame({'id'      : [6,7,8,9,10]
                      , 'final' : [70,83,65,95,80]})
test2
test2 = test2.rename(columns={'final':'midterm'})
test2
test = pd.concat([test1, test2])
test


df=pd.DataFrame({'sex' : ['M','F', np.nan, 'M','F']
                 , 'score' : [5,4,3,4,np.nan]})

pd.isna(df).sum()
df.isna()

# 결측치 제거하기
df.dropna(subset = ['score'])
df.dropna()

exam=pd.read_csv('data/exam.csv')
exam
exam.loc[3:5,]
exam.iloc[0,0:3]

df[df['score']==3]['score'] = 5
df


exam.iloc[[2,7,10],2] = np.nan
exam


# 수학 점수 50점 이하인 학생들 점수 50점으로 상향 조정!
import pandas as pd
exam=pd.read_csv('data/exam.csv')

exam.loc[exam['math']<=50,'math'] = 50

exam
exam['math'].unique()
exam.unique()




# 영어점수 90점 이상 90점으로 하향 조정   # iloc 쓸거면, 숫자 벡터여야 함.
exam=pd.read_csv('data/exam.csv')
exam.iloc[exam['english']>=90,3]   # 이건 에러가 뜨지만
exam.iloc[exam['english']>=90,3] = 90   # 돌아가기는 함.
exam.iloc[exam[exam['english']>=90].index,3] = 90
np.where(exam['english'] >= 90)  #결과가 튜플임
exam.iloc[np.where(exam['english'] >= 90)[0],3] = 90 
exam.iloc[np.array(exam['english'] >= 90), 3] = 90

exam

# math 점수 50 이하 "-" 변경
exam=pd.read_csv('data/exam.csv')
exam.loc[exam['math']<= 50, 'math'] = '-'   # 숫자 컬럼인데 문자값 넣어서 경고 뜸. 되긴 함

exam

# 결측치를 수학 점수 평균으로 바꾸고 싶음
exam.loc[exam['math'] == '-', 'math'] = exam.query('math not in ["-"]')['math'].mean()
exam


exam.loc[exam['math'] == "-", ['math']] = np.nan
exam
math_mean = exam['math'].mean()
math_mean
exam.loc[pd.isna(exam['math']), ['math']] = math_mean
exam

vector = np.array([np.nan if x == '-' else float(x) for x in exam['math']])  # 리스트를 array로 만들겠다
vector = np.array([float(x) if x!="-" else np.nan for x in exam['math']])

np.nanmean(vector)  # NaN 제외하고 평균값 구해줌
vector.mean()  # 이건 NaN 때문에 결과도 NaN 나옴. dataframe.mean() 이 아니라서 NaN이 나오나봄. dataframe.mean()는 자동으로 제거하고 평균 구하줌.


math_mean = exam[exam['math'] != '-']['math'].mean()
exam['math'] = exam['math'].replace("-", math_mean)
exam['math'] = exam['math'].replace('-', math_mean)
exam


exam['math'] = np.where(exam['math'] == '-', mean, exam['math'])


exam[exam['nclass'].isin([1,2])]


import pandas as pd
df = pd.read_csv("data/economics.csv")
df

sns.lineplot()



# 7/18 수업
import numpy as np

matrix = np.column_stack((np.arange(1,5), np.arange(12,16)))
np.column_stack((1,2))
np.column_stack(([1,2,3],[4,5,6])).shape
np.coulmn_stack(())


matrix = np.vstack((
    np.arange(1,5),
    np.arange(12,16))
)

len(matrix)
matrix
matrix.size
matrix.shape


matrix
print("행렬:\n", matrix)

matrix = np.vstack(np.array([1,2,3]), np.array([4,5,6]))
matrix = np.vstack(([1,2,3], [4,5,6]))
matrix = np.vstack((1,2))
matrix = np.vstack(([1,2,3],[4,5,6]))
matrix
type(np.arange(1,5))



import pandas as pd


np.zeros(5)
np.zeros([5,4])
np.arange(1,5).reshape((2,2))
np.arange(1,5)

np.arange(1,7).reshape((2,3))
np.arange(1,7).reshape((2,-1))   # -1을 쓰면 한쪽 행/ 열에 맞춰서 나머지 열/ 행도 맞게 출력해줌.
np.arange(1,7).reshape((2,-1), order = 'F') 
np.arange(1,7).reshape((2,-1), order = 'C')


# 0에서 99까지 수 중 랜덤하게 50개 숫자를 뽑아서 5 by 10 행렬 만드세요.
np.random.seed(2024)
np.random.randint(0,100, 50).reshape((5,-1))
np.random.randint(0,100, 50).reshape((5,-1), order='C')  # 가로 기준
np.random.randint(0,100, 50).reshape((5,-1), order='F')  # 세로 기준


mat_a = np.arange(1,21).reshape((4,5), order='F')
mat_a

mat_a[0,0]   # 인덱스하면 1차원 벡터로 출력됨.
mat_a[1,1]
mat_a[2,3]
mat_a[0:2,3]
mat_a[1:3,1:4]
mat_a[3,]
mat_a[3,:]
mat_a[3,::2]

mat_a[:,1]  # 벡터
mat_a[:,1].reshape((-1,1))
mat_a[:,(1,)]  # 행렬
mat_a[:,[1]]    # 행렬
mat_a[:,1:2]  # 행렬

# 짝수 행만 선택하려면?
mat_b=np.arange(1,101).reshape((20,-1))
mat_b[1::2,]

mat_b[[1,4,6,14],]


x = np.arange(1,11).reshape((5,2))*2
x
x[[True,True,False,False,True],0]


mat_b[mat_b[:,1]%7 == 0,:]   # 7로 딱 나누어 떨어지는 값이 존재하는 행을 불러오기
mat_b

mat_b[mat_b[:,1] > 50, :]


import matplotlib.pyplot as plt

np.random.seed(2024)
img1 = np.random.rand(3,3)
print("이미지 행렬 img1:\n", img1)
plt.clf()
plt.imshow(img1, cmap='gray', interpolation = 'nearest')
plt.show()


import urllib.request
img_url = "https://bit.ly/3ErnM2Q"
urllib.request.urlretrieve(img_url, "jelly.png")  # 해당 주소에서 이미지를 다운로드 하되, "" 이름으로 저장해라.


!pip install imageio
import imageio
jelly = imageio.imread("jelly.png")
print("이미지 클래스 :" , type(jelly) )
print("이미지 차원 :", jelly.shape)    # (88, 50, 4)  # 4: 장 수의 의미임
print("이미지 첫 4x4 픽셀, 첫 번째 채널 :", jelly)

jelly.shape
jelly.transpose().shape
jelly[:,:,0].shape

jelly[:,:,0].transpose().shape



plt.clf()
plt.imshow(jelly)
plt.imshow(jelly[:,:,0].transpose())
plt.imshow(jelly[:,:,0])  # R
plt.imshow(jelly[:,:,1])  # G
plt.imshow(jelly[:,:,2])  # B
plt.imshow(jelly[:,:,3])  # 투명도
plt.axis('off')  # 축 정보 없애기
plt.show()

# 두 개의 2x3 행렬 생성
mat1 = np.arange(1,7).reshape(2,3)
mat2 = np.arange(7,13).reshape(2,3)

my_array = np.array([mat1, mat2])   
my_array
my_array.shape    # (2,2,3)  <- 2 / (2,3)  <- (2,3)이 2행으로 반복된다. <- 3 : 열

first_slice = my_array[0,:,:]
first_slice


my_array2 = np.array([my_array, my_array]) 
my_array2
my_array2.shape # (2, 2, ,2,,3 ) <- 2 / 2 / (2,3)  <- (2,3)이 2행으로 반복되는게 2행으로 반복됨 <- 3: 열

filtered_array = my_array[:, :, :-1]  <- 마지막 열이 1개 빠짐.
filtered_array

my_array[:,:,-2]

filtered_array2 = my_array[:,:,[0,2]]
filtered_array2

my_array[:,0,:]
my_array[0,1,[1,2]]
my_array[0,1,1:3]


mat_x = np.arange(1,101)
mat_x.reshape((5,5,4))  # 5*4 20 씩 <- 작은 행렬 5번 반복
mat_x.reshape((10,5,2)) # 5*2 10씩 <- 작은 행렬 10번 반복
mat_x.reshape((-1,5,2))


#넘파이 배열 메서드
a = np.array([[1,2,3], [4,5,6]])
a

a.sum()  # 모든 원소의 원소 합
a.sum(axis=0)  # 열별 합계
a.sum(axis=1)  # 행별 합계

a.mean() # 모든 원소의 원소 평균
a.mean(axis=0)
a.mean(axis=1)


# 가장 큰 수는?
mat_b.max()

# 행별 가장 큰수는?
mat_b.max(axis=1)

# 열별 가장 큰수는?
mat_b.max(axis=0)

a=np.array([1,3,2,5]).reshape((2,2))
a.cumsum()

mat_b.cumsum(axis=1)
mat_b.cumprod(axis=1)

mat_b.flatten()  # flatten() : 1차원으로 만드는 것

d = np.array([1,2,3,4,5])
d.clip(2,4)  # 최소값이 2가 될 수 있게 1->2, 최댓값이 4가 될 수 있게 4->5
d.tolist()


import numpy as np
np.random.rand(1)

def X(n):
    return np.random.rand(n)

X(3)



# 베르누이 확률변수 모수 : p 만들어보기
def B(p):
    return np.where(np.random.rand(1) < p, 1,0)

B(0.4)


# 확률변수 X 
def X(num, p):
    return np.where(np.random.rand(num)<p, 1, 0)

X(10,0.4)

X(5, 0.4)
X(100,0.5).mean()  # 참인 비율을 알 수 있음.  # 이게 기하분포 확률 변수가 되는 것임?
X(10000,0.5).mean()
X(100000,0.5).mean()

# 새로운 확률변수: 가질 수 있는 값 0,1,2
# 확률은 20%, 50%, 30%
def Y(num):
    x=np.random.rand(num)
    return np.where(x<0.2, 0, np.where(x<0.7,1,2))
    
    
Y(1)


def Y(f, p):
    x=np.random.rand(num)
    p_cumsum = p.cumsum()
    return np.where(x<p_cumsum[0], 0, np.where(x<p_cumsum[1],1,2))


p=np.array([0.2,0.5,0.3])
Y(10,p)

b=np.array([1,2,3,4,5,6]).reshape((2,3))
b


import numpy as np
def Y(num, p):
    return sum(np.where(np.random.rand(num) < p, 1 , 0))
    
Y(10,0.3)


# E[X]
import numpy as np
np.array([0,1,2,3]) * np.array([1/6,2/6,2/6,1/6])
sum(np.array([0,1,2,3]) * np.array([1/6,2/6,2/6,1/6]))


a = {1,2}
a
a.add(3)
a
a.remove(4)
a.remove(3)
a
a.discard(5)
a
a.discard(2)
a
a.add(2)
a
a.add(3)
a
a.pop(3)
a.pop()
a.clear()
a

b=set((4,5,6))
b
a=set((1,2,3))
a.update(b)
a.union(b)


# 07/19
import numpy as np
import matplotlib.pyplot as plt
data = np.random.rand(10000)



def X(num):
    return np.mean(np.random.rand(num))

data=[]
for n in np.arange(0,10001):
    data.append((float(X(5)))
    
data



plt.clf()
plt.hist(data1, bins=500, alpha=0.5, color='blue')
plt.title('Histogram of Numpy Vector')
plt.xlabel('Value')
plt.ylabel('Frequency')
plt.grid(True)
plt.show()


# 방법 2
data2 = np.random.rand(50000).reshape(-1,5).mean(axis = 1)

plt.clf()
plt.hist(data2, bins=500, alpha=0.5, color='blue')
plt.title('Histogram of Numpy Vector')
plt.xlabel('Value')
plt.ylabel('Frequency')
plt.grid(True)
plt.show()


# 교재 9장
!pip install pyreadstat  # 다른 프로그램 데이터 가져올 수 있음?


import pandas as pd
import numpy as np
import seaborn as sns

raw_welfare = pd.read_spss



# 7/22
fruits = ["apple","banana","cherry"]


empty_list = []
empty_list2 = list()

numbers = [1,2,3,4,5]
range_list = list(range(5))
range_list


range_list[3] = "LS 빅데이터 스쿨"
range_list
range_list[1] = [1,2,3]
range_list
range_list[1][2]


# 리스트 내포(comprehension)
# 1. 대괄호로 쌓여져있다 -> 리스트다.
# 2. 넣고 싶은 수식 표현을 x를 사용해서 표현
# 3. for .. in .. 을 사용해서 원소 정보 제공
range(10)
list(range(10))
squares = [x**2 for x in range(10)]

[x**2 for x in [3,5,2,15]]
[x**2 for x in (3,5,2,15)]
[x**2 for x in {3,5,2,15}]
[x**2 for x in np.array([3,5,2,15])]
[x**2 for x in np.arange(0,4)]

# pandas serise
import pandas as pd
exam = pd.read_csv("exam.csv")


numbers = [5,3,6]
repeated_list = [x for x in numbers for _ in range(4)]
repeated_list

# _ 의미
# 1. 앞에 나온 값을 가리킴
5+4
_+6  # _는 9를 의미

# 값 생략, 자리 차지(placeholder)
a, _, b = (1,2,4)
a; b
_


_ = None
_
del _
_
del _
_



# for 루프 문법
# for i in 범위:
# 작동방식
for x in [4,1,2,3]:
    print(x)
    
for x in range(5):
    print(x**2)


# 리스트를 하나 만들어서 for 루프를 사용해서 2,4,6,8, ..., 20 수를 채워넣기
a =[]
for i in np.arange(1,11):
    a.append(i*2)

a

[i for i in np.arange(2,21,2)]


mylist = [0]*5
mylist





numbers = [5,2,3]

## looping ten times uping _
for x in numbers:
    for y in range(4):
        print(x)
    

mylist = []
for i in [1,2,3]:
    
    
mylist = [0]*10
mylist
for i in range(10):
    mylist[i] = 2*(i+1)
    
mylist


# 인덱스 공유해서 카피하기
mylist_b = [2,4,6,80,10,12,24,35,23,20]
for i in range(10):
    mylist[i] = mylist_b[i]
    
mylist



# 퀴즈 : mylist_b의 홀수번째 위치에 있는 숫자들만 mylist에 가져오기
mylist = [0]*5
for i in range(5):
    mylist[i] = mylist_b[2*i]
    
mylist


# 리스트 컴프리헨션으로 바꾸는 방법
# 바깥은 무조건 대괄호로 묶어줌: 리스트 반환하기 위해서.
# for 루프의 : 는 생략한다.
# 실행부분을 먼저 써준다.
# 중복부분 제외시킴.
# 결과를 받는 부분 제외시킴.

[2*x for x in range(1,11)]


for i in range(3):
    for j in range(2):
        print(i,j)
        
for i in [0,1,2]:
    for j in [0,1]:
        print(i,j)
        
for i in [0,1,2]:
    for j in [3,4]:
        print(i)
        
[x for x in range(3) for y in [3,4]]


# 원소 체크
fruits = ['apple', 'apple', 'banana', 'cherry']
"banana" in fruits


# 바나나의 위치를 뱉어내게 하려면?
fruits = ['apple', 'apple', 'banana', 'cherry']

import numpy as np

for i in range(len(fruits)):
    np.where(fruits[i] == 'banana',print(i), print(None))

fruits = np.array(fruits)
int(np.where(fruits == 'banana')[0][0])


# 원소 순서를 거꾸로 써주는 reverse()
fruits = ['apple', 'apple', 'banana', 'cherry']
fruits.reverse()
fruits 


fruits.append("pineapple")
fruits

fruits.insert(2,"orange")
fruits


# 원소 제거
fruits.remove("apple") # 동일한 값이 있다면 가장 먼저 있는 값을 지움.
fruits

# 제거할 항목 리스트
fruits = np.array(['apple', 'apple', 'banana', 'cherry'])
items_to_remove = np.array(['banana','apple'])
items_to_remove

mask = ~np.isin(fruits, items_to_remove)   # 둘다 array이어야 함.
mask = ~np.isin(fruits, ['banana','apple'])
mask

filtered_fruits = fruits[mask]
filtered_fruits


import pandas as pd
mpg = pd.read_csv('data/mpg.csv')
mpg.shape

!pip install seaborn
import seaborn as sns
import matplotlib.pyplot as plt

plt.clf()
sns.scatterplot(data=mpg, x="displ", y="hwy").set(xlim=[3,6], ylim=[10,30])
plt.show()

# 막대그래프
df_mpg = mpg.groupby("drv", as_index=False).agg(mean_hwy = ('hwy', 'mean'))
df_mpg

plt.clf()
sns.barplot(data=df_mpg, x='drv', y='mean_hwy', hue='drv')
plt.show()


drv_count = mpg.groupby('drv', as_index=False).agg(n= ('drv', 'count'))
plt.clf()
sns.barplot(data=drv_count, x='drv', y='n')
plt.show()



# 9장 교재
import pandas as pd
welfare = pd.read_spss('data/Koweps_hpwc14_2019_beta2.sav')

welfare.shape
welfare.info()
welfare.describe()

welfare = welfare.rename(columns = { 'h14_g3' : 'sex'
                                   , 'h14_g4' : 'birth'
                                   , 'h14_g10' : 'marriage_type'
                                   , 'h14_g11' : 'religion'
                                   , 'p1402_8aql' : 'income'
                                   , 'h14_eco9' : 'code_job'
                                   , 'h14_reg7' : 'code_region'})
welfare.head()

import numpy as np
welfare['sex'] = np.where(welfare['sex'] == 9 , np.nan , welfare['sex'])
welfare
welfare['sex'] = np.where(welfare['sex'] == 1, 'male' , 'female')
welfare['sex'].value_counts()

sns.countplot(data=welfare, x='sex')



np.arange(33).sum()/33
sum(np.unique((np.arange(33)-16)**2)*2)/33


# 분산
import numpy as np
x=np.arange(4)
pro_x = np.array([1/6, 2/6, 2/6, 1/6])
Ex = sum(x*pro_x)
Exx = sum(x**2 * pro_x)
sum((x - Ex)**2 * pro_x)

# 문제
# x = 0~98까지 정수
import numpy as np
x = np.arange(0,99)
j1 = np.arange(1,51)
j2 = np.arange(49,0, -1)
j3 = np.concatenate((j1,j2))

pro_x = np.array(j3) / 2500
sum(x*pro_x)


j1 = np.arange(1,51)
j2 = np.arange(49,0, -1)
j3 = np.concatenate((j1,j2))
j3




# 7/24 수업
a =set((1,2,3))
a
b = set((3,4,5))
c = set((3,4))
a.difference(b)
b.difference(a)
a.symmetric_difference(b)
b.symmetric_difference(a)
a.isdisjoint(b)
a.issubset(b)
c.issubset(b)
b.issubset(c)
c.issuperset(b)
b.issuperset(c)

from scipy.stats import bernoulli
bernoulli.pmf(1, 0.3)  # 베르누이 확률변수 값이 1일 때의 확률이 0.3
bernoulli.pmf(0, 0.3)  # 베르누이 확률변수 값이 0일 때의 확률이 0.3

bernoulli.pmf([0,1], 0.3)

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

# lec4에서 이어서 하겠음
