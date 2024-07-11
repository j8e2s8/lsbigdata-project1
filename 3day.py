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


mpg['grade'].value_counts()
mpg['grade'].plot.bar()

# 4day

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
