# 패키지 불러오기
import numpy as np
import pandas as pd

tab3 = pd.read_csv('./data/tab3.csv')
tab3



tab1 = pd.DataFrame({'id' :np.arange(1,13),
                    'score' : tab3['score']})


sex = ["female"]*7 + ["male"]*5
tab2 = pd.DataFrame({'id' :np.arange(1,13),
                    'score' : tab3['score'],
                    'gender' : sex })



                    
# 1표본 t검정 (그룹 1개)
# 귀무가설 vs 대립가설
# H0: mu = 10 vs H1: mu != 10
# 유의수준 5%로 설정
from scipy.stats import ttest_1samp
ttest_1samp(tab1['score'], 10)
ttest_1samp(tab1['score'], 10).pvalue < 0.05  # H0 기각 못함.
# 유의확률 0.0648이 유의수준 0.05보다 크므로 귀무가설을 기각하지 못한다.
# 귀무가설이 참일 때, 표본평균이 관찰될 확률이 6.48%이므로 이것은 우리가 생각하는 보기 힘들다고 판단하는 기준인 유의수준 0.05보다 크므로 

ttest_1samp(tab1['score'], 10).confidence_interval(confidence_level=0.95)


# 2표본 t검정 (그룹 2)
## 귀무가설 vs 대립가설
## H0: mu_m = mu_f vs Ha: mu_m > mu_f
## 유의수준 1%로 설정
from scipy.stats import ttest_ind, t, norm
female = tab2[tab2['gender'] == 'female']['score']
male = tab2[tab2['gender'] == 'male']['score']
result = ttest_ind(female, male, equal_var = True, alternative = 'less')  # mu_f - mu_m < 0
result
result.confidence_interval(confidence_level=0.95)

x_bar = female.mean()
y_bar = male.mean()
sx_2 = female.var(ddof=1)
sy_2 = male.var(ddof=1)
n1 = len(female)
n2 = len(male)
sp_2 = (sx_2*(n1-1) + sy_2*(n2-1)) / (n1 + n2 -2)
z = ((x_bar - y_bar) - 0)/ np.sqrt(sp_2/n1 + sp_2/n2)

# 질문:
t.cdf(z, df=n1+n2-2)
norm.cdf(z, loc=0, scale=1)


# 대응표본 t검정 (짝지을 수 있는 표본)
tab3_data = tab3.pivot_table(index='id', columns='group', values='score').reset_index()
tab3_data2 = pd.DataFrame({})
tab3_data2['score_diff'] = tab3_data['after'] - tab3_data['before']

tab3_data


long_form = tab3_data.reset_index().melt(id_vars='id', value_vars=['before','after'], var_name='group', value_name='score')
# var_name, value_name은 컬럼명이 되는 애들임.

pv1 = long_form.pivot_table( columns='group', values='score').reset_index()

pv2 = pv1.melt(id_vars='id', value_vars=['after', 'before'], var_name='group' ,value_name='score')


import seaborn as sns
tips = sns.load_dataset('tips')
tips = tips.reset_index()
pv3 = tips.pivot_table(index = 'index', columns='day', values='tip').reset_index()


?bernoulli.expect



# 대응표본 t 검정 (짝지을 수 있는 표본)
## 귀무가설 vs 대립가설
## H0: mu_before = mu_after vs. Ha: mu_after > mu_before
## H0: mu_d = 0 vs. Ha: mu_d > 0
## mu_d = mu_after - mu_before
## 유의수준 1%로 설정

# mu_d에 대응하는 표본으로 변환
tab3_data2
result = ttest_1samp(tab3_data2, 0, alternative='greater')
result
 




<<<<<<< HEAD
# pivot 연습 -------------------------------
# mu_d에 대응하는 표본으로 변환
tab3_data = tab3.pivot_table(index='id', 
                             columns='group',
                             values='score').reset_index()

tab3.pivot_table(index='id', columns='group',values='score')
tab3.pivot_table(columns='group',values='score')
tab3.pivot_table(index='id', columns='group',values='score')['after'].mean()
tab3.pivot_table(index='id', columns='group',values='score')['before'].mean()

tab3.pivot(index='id', columns='group', values='score')
tab3.pivot(index='id', columns='group', values='score').reset_index()
tab3.pivot( columns='group', values='score')


tab3_data['score_diff'] = tab3_data['after'] - tab3_data['before']
test3_data = tab3_data[['score_diff']]
test3_data

from scipy.stats import ttest_1samp

result = ttest_1samp(test3_data["score_diff"], 
                     popmean=0, alternative='greater')
t_value=result[0] # t 검정통계량
p_value=result[1] # 유의확률 (p-value)
t_value; p_value

# 
# long to wide: pivot_table()
tab3_data = tab3.pivot_table(
    index='id', 
    columns='group',
    values='score'
    ).reset_index()

# wide to long: melt()
long_form = tab3_data.melt(
    id_vars='id', 
    value_vars=['before', 'after'],
    var_name='group',
    value_name='score'
    )

# 연습 pivot&melt: long to wide, wide to long
df= pd.DataFrame({"id": [1, 2, 3],
                  "A": [10, 20, 30],
                  "B": [40, 50, 60]})

df_long=df.melt(id_vars="id", 
                value_vars=["A", "B"],
                var_name="group",
                value_name="score")

df_wide=df_long.pivot_table(
                index="id",
                columns="group",
                values="score"
                ).reset_index()
=======
>>>>>>> f6225cb6866db1dbc484cf29f80d50d2e032083a
