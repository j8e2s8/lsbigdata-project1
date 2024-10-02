# 9-7 교재
#!pip install pandas
#!pip install numpy
#!pip install seaborn
#!pip install pyreadstat
#!pip install matplotlib
!pip install scipy

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

raw_welfare = pd.read_spss('data/koweps/Koweps_hpwc14_2019_beta2.sav')
welfare = raw_welfare.copy()

welfare.shape
welfare.info()
welfare.describe()

welfare = welfare.rename(columns = {'h14_g3' : 'sex',
                                    'h14_g4' : 'birth',
                                    'h14_g10' : 'marriage_type',
                                    'h14_g11' : 'religion',
                                    'p1402_8aq1' : 'income',
                                    'h14_eco9' : 'code_job',
                                    'h14_reg7' : 'code_region'})

welfare2 = welfare[['sex', 'birth', 'marriage_type', 'religion', 'income', 'code_job', 'code_region']]
welfare2.columns



# sex 이상치 확인 -> 해결
welfare2['sex'].dtypes
welfare2['sex'].value_counts()
welfare2['sex'] = np.where(welfare2['sex']==9, np.nan, welfare2['sex'])
welfare2['sex'].isna().sum()  # 이상치 없음.


# sex 범주 이름 바꿔주기
welfare2['sex'] = np.where(welfare2['sex']==1 , 'male', 'female')
welfare2['sex'].value_counts()


# sex 빈도 막대 그래프 만들기
import seaborn as sns
import matplotlib.pyplot as plt
plt.clf()
sns.countplot(data=welfare2, x='sex', hue='sex')
plt.show()


# income 정보 확인
welfare2['income'].dtypes
welfare2['income'].describe() 
welfare2['income'].isna().sum() # 결측값 확인됨

# income 결측값 처리
sex_income = welfare2.dropna(subset='income').groupby('sex', as_index=False).agg(mean_income = ('income','mean'))
sex_income

plt.clf()
sns.barplot(data=sex_income, x='sex', y='mean_income', hue='sex')
plt.show()




# birth 정보 확인
welfare['birth'].describe()

plt.clf()
sns.histplot(data=welfare, x='birth')  # 빈도 히스토그램
plt.show()

welfare2['birth'].isna().sum()  # 결측치 갯수
welfare2["birth"].isna().sum()  # snippet 이용해서 결측치 갯수 확인

welfare2['birth'] = np.where(welfare2['birth']==9999, np.nan, welfare['birth'])
welfare2['birth'].value_counts()



# 새로운 age 변수 만들기
welfare2 = welfare2.assign(age = 2024 - welfare2['birth']+1)
welfare2['age']

welfare2['age'].describe()
welfare2['age'].value_counts().sort_index()
welfare2['age'].value_counts().sort_index(ascending=False)

plt.clf()
sns.histplot(data=welfare2, x='age') # 빈도 히스토그램 
plt.show()



# age로 groupby 데이터 만들기
age_income = welfare2.dropna(subset='income').groupby('age').agg(mean_income=('income', 'mean'))
age_income.head()

plt.clf()
sns.lineplot(data=age_income, x='age', y='mean_income')
plt.show()



# income에 대답을 안 한 사람들에 대해서 알아보자
# income에 대답을 안 한 사람들 중 성별별로 몇 명인지 알아보자 
my_df = welfare2.query('income.isna()==True').groupby('age', as_index=False).agg(isna_count=('sex','count'))
plt.clf()
sns.barplot(data=my_df, x='age' , y='isna_count')
plt.xlim([0,100])
plt.ylim([0,300])
plt.show()


my_df2 = welfare2.assign(income_na=welfare2['income'].isna()).groupby('age', as_index=False).agg(n=('income_na','sum')) # 'count'로 하면 달라짐. 왜?
plt.clf()
sns.barplot(data=my_df2, x='age' , y='n')
plt.xlim([0,100])
plt.ylim([0,300])
plt.show()



# 결측값이 있을 때 count함수와 sum함수의 차이
welfare2['income'].isna().count()  # count는 True, False 상관없이 다 세어줌
welfare2['income'].isna().sum()   # sum은 True값만 1로 더해줌.
len(welfare['income'])  # 전체 행 갯수



# 240 페이지 : age를 기준으로 범주화 ageg 하기 (young, middle, old)
welfare2 = welfare2.assign(ageg = np.where(welfare2['age'] < 30, 'young'
                                 , np.where(welfare2['age'] <= 59, 'middle'
                                 , 'old')))


plt.clf()
sns.countplot(data=welfare2, x='ageg', hue='ageg')
plt.show()



# ageg를 기준으로 groupby 데이터 
ageg_income = welfare2.dropna(subset='income').groupby('ageg', as_index=False).agg(mean_income = ('income', 'mean'))

plt.clf()
sns.barplot(data=ageg_income, x='ageg', y='mean_income', hue='ageg')
plt.show()

plt.clf()
sns.barplot(data=ageg_income, x='ageg', y='mean_income', hue='ageg', order=['young','middle','old'])
plt.show()



# ageg, sex를 기준으로 groupby 데이터
sex_income = welfare2.dropna(subset='income').groupby(['ageg','sex'], as_index=False).agg(mean_income = ('income', 'mean'))
sex_income

plt.clf()
sns.barplot(data=sex_income, x='ageg', y='mean_income', hue='sex', order=['young','middle','old'])
plt.show()


# age, sex를 기준으로 groupby 데이터
sex_age = welfare2.dropna(subset='income').groupby(['age', 'sex'], as_index=False).agg(mean_income = ('income', 'mean'))

plt.clf()
sns.barplot(data=sex_age, x='age', y='mean_income', hue='sex')
plt.show()

plt.clf()
sns.lineplot(data=sex_age, x='age', y='mean_income', hue='sex')
plt.show()


# age를 10단위씩 범주화하기. 비효율 코드임.
welfare2 = welfare2.assign(age2 = np.where(welfare2['age']<10), 0  # 비효율적인 코드
                                 , np.where(welfare2['age']<20, 10
                                 , np.where(welfare2['age']<30, 20
                                 , np.where(welfare2['age']<40, 30
                                 , np.where(welfare2['age']<50, 40
                                 , np.where(welfare2['age']<60, 50
                                 , np.where(welfare2['age']<70, 60
                                 , np.where(welfare2['age']<80, 70
                                 , np.where(welfare2['age']<90, 80
                                 , np.where(welfare2['age']<100, 90))



# cut 사용해서 연령 구분할 수 있음.
vec_x = np.random.randint(1,100,50)           
bin_cut = np.array([0,9,19,29,39,49,59,69,79,89,99,109,119])
# bin_cut = [9,19,29,39,49,59,69,79,89,99,109,119]
pd.cut(vec_x, bins=bin_cut)  # 각 값이 어느 계급에 해당하는지 계급을 반환해줌.



# ---------------------------------------



# 한결이 코드 (너무 복잡)
age_min, age_max = (welfare2['age'].min(), welfare2['age'].max())
vec_x = np.random.randint(1,100,50)
bin_cut = [0] + [10*i+9 for i in np.arange(age_max//10 +1)]
pd.cut(vec_x, bins=bin_cut)


# 내 코드 (나이대별 수입 분석을 위해 연령 구별 하기)
# !pip install numpy --upgrade
# np.version.version
# version이 2.0.1 로 업데이트가 되긴 했는데, 다른 패키지랑 충돌됨. 다시 가상환경을 삭제했다가 새로 만들어서 연결함.
bin_cut = np.arange(13)*10-1
bin_cut[0] = 0
# str(x) + "대"
label_cut = bin_cut.astype(str) + "대"
vec_x = np.random.randint(1,100,50)
pd.cut(vec_x, bins=bin_cut)

list(map(str,[1,2,3]))
list(map(str,np.array([1,2,3])))


# 나이대 age_group 변수 만들기
bin_cut = np.arange(13)*10-1
bin_cut[0] = 0
label_cut = (np.arange(12)*10).astype(str) + "대"
welfare2 = welfare2.assign(age_group = pd.cut(welfare2['age'], bins = bin_cut, labels =label_cut ))
welfare2.head()


# 나이대 age_group별 groupby 데이터 만들기
age_income = welfare2.dropna(subset='income').groupby('age_group', as_index=False).agg(mean_income = ('income','mean'))
age_income
welfare2.dropna(subset='income').query('age_group == "0대"')  # 0대 나이대 데이터 자체가 없음.
welfare2.dropna(subset='income').query('age_group == "10대"')  # 10대 나이대 데이터 자체가 없음.
welfare2.dropna(subset='income').query('age_group == "20대"')  # 20대 나이대부터 데이터가 존재함.


# 시각화 (한글 깨지는 거 코드로 해결해보기)
from matplotlib import font_manager, rc
plt.clf()

# 한글 폰트 설정
font_path = "C:/Windows/Fonts/malgun.ttf"  # 예시: 윈도우 시스템에 있는 맑은 고딕 폰트 경로
font_name = font_manager.FontProperties(fname=font_path).get_name()
rc('font', family=font_name)
sns.barplot(data=age_income, x='age_group', y='mean_income')
plt.show()






# 나이대 age_group과 sex별 groupby 데이터 만들기
sex_income = welfare2.dropna(subset= 'income').groupby(['age_group', 'sex'], as_index=False).agg(mean_income = ('income','mean')) # 에러 

welfare2.info()  # age_group 컬럼의 타입이 category임.

# 판다스 데이터 프레임을 다룰 때, 변수의 타입이 카테고리로 설정되어 있는 경우, groupby+agg 콤보 안 먹힘.
# 그래서 object 타입으로 바꿔 준 후 수행
welfare2['age_group'] = welfare2['age_group'].astype('object')
welfare2.info() 

sex_age_income = welfare2.dropna(subset='income').groupby(['age_group', 'sex'], as_index=False).agg(mean_income=('income','mean'))
sex_age_income


plt.clf()
sns.barplot(data=sex_age_income, x='age_group', y='mean_income', hue='sex')  # 폰트 에러 해결하기
plt.show()





# 옵션을 사용해야 하는 사용자 정의 함수
def custom_mean(series, dropna=True):
    if dropna:
        return series.dropna().mean()
    else:
        return series.mean()

welfare2.groupby(['age_group','sex'], as_index=False).agg(top = ('income',lambda x: custom_mean(x, dropna=False)))
welfare2.groupby(['age_group','sex'], as_index=False).agg(top = ('income',lambda x: custom_mean(x, dropna=True)))  # 생기는 nan은 제일 먼저 데이터가 dropna되면서 데이터가 없어서 생긴 nan인 듯 

np.quantile(x, q=0.95)  # ppf처럼 상위 5%는 0.95로 잡아줘야함.


# 연령대별, 성별 상위 4% 수입 찾아보기.
sex_age_income = welfare2.dropna(subset='income')sort_values('age_group').groupby(['age_group','sex'], as_index=False).agg(top4per_income = ('income', lambda x: np.quantile(x,q=0.96)))
sex_age_income                                                                                                             # x는 imcome을 대신함. (lambda 앞에 있는애를 대신함)


plt.clf()
sns.barplot(data=sex_age_income, x="age_group", y="top4per_income", hue='sex')
plt.show()



# groupby의 다른 코드법
welfare2.dropna(subset = 'income').groupby('sex', as_index = False)[['income']].agg(['mean','std'])



# 9-6장
welfare['code_job']
welfare['code_job'].value_counts()

list_job = pd.read_excel('./data/koweps/Koweps_Codebook_2019.xlsx', sheet_name='직종코드')
list_job.head()

welfare2 = welfare2.merge(list_job, how='left', on='code_job')
welfare3 = welfare2.dropna(subset = ['job','income'])[['income','job', 'sex']]

job_income = welfare3.groupby('job', as_index=False).agg(mean_income = ('income','mean')).sort_values('mean_income', ascending=False).head(10)


plt.clf()
sns.barplot(data=job_income, y='job', x='mean_income' , hue='job')
#windows(width=4, height=4, rescale="R", title="R") 
plt.show()




job_income2 = welfare3.groupby(['job','sex'], as_index=False).agg(mean_income = ('income','mean')).sort_values('mean_income', ascending=False).head(10)

plt.clf()
sns.barplot(data=job_income2, y='job' ,x='mean_income', hue='sex')
plt.tight_layout()
plt.show()


df_female = welfare3.query("sex == 'female'").groupby('job', as_index=False).agg(mean_income = ('income','mean')).sort_values('mean_income', ascending=False).head(10)
plt.clf()
sns.barplot(data=job_income2, y='job' ,x='mean_income', hue='job')
plt.tight_layout()
plt.show()



# 9-8
welfare2.info()
welfare2['marriage_type']
df = welfare2.query('marriage_type != 5').groupby('religion', as_index=False)['marriage_type'].value_counts(normalize=True)

df.query('marriage_type == 1').assign(proportion = df['proportion']*100).round(1)

