import pandas as pd 

import os
cwd = os.getcwd()
cwd = cwd+'/adp-dataset'
os.chdir(cwd)
os.getcwd()


admission = pd.read_csv('admission.csv')
admission.head()

# GPA : 학점
# GRE : 대학원 입학 시험 (영어, 수학)

# 합격을 한 사건 : admit
# admit의 확률 오즈(Odds)는?
# P(admit) = 합격 인원 수 / 전체 인원 수
# (모수 합격할 확률을 우리가 모르니까 조사한 데이터의 비율로 추정할 수 있음)
p_hat = admission['admit'].mean()
p_hat / (1- p_hat)   # 오즈비 : 0.465

# P(A) > 0.5 인 경우  -> 오즈비: 무한대에 가까워짐
# p(A) = 0.5 -> 오즈비 : 1
# p(A) < 0.5 -> 오즈비 : 0에 가까워짐
# 확률의 오즈비가 갖는 값의 범위 : 0 ~ 무한대

admission['admit'].unique()

# rank별로 오즈비 계산
grouped_data = admission.groupby('rank', as_index=False).agg(p_admit = ('admit','mean'))  # rank별로 합격률(추정) 구하기
grouped_data['odds'] = grouped_data['p_admit']/(1-grouped_data['p_admit'])
grouped_data['log_odds'] = np.log(grouped_data['odds'])
# 합격률이 높은 rank1일 수록 오즈비가 커짐. (확률이 0.5를 넘어서 더 커질 수록 오즈비가 무한에 가까워지니까)
# 어떤 사건의 확률이 커질 수록 오즈비가 커지니까, 확률을 직접적으로 사용하기 힘든 상황에서 확률과 비슷한 오즈비를 사용할 수 있음


# 오즈비가 3일 때, 사건의 확률을 구하여라
# p / (1-p) = 3 
# p = 3(1-p) , p = 3 - 3p , 4p = 3, p = 3/4

# x:gre, y:admit 산점도
import seaborn as sns 

sns.stripplot(data=admission, x='gre', y='admit', jitter=0.3, alpha=0.3)  # jitter : 약간 떨어뜨려주는 애

sns.scatterplot(data=admission, x='gre', y ='admit')

sns.stripplot(data=admission, x='rank', y='admit', jitter=0.3, alpha=0.3)  # jitter : 약간 떨어뜨려주는 애
sns.stripplot(data=grouped_data , x='rank', y='p_admit')

sns.regplot(data=grouped_data , x='rank', y='p_admit')  # 파란색 범위는 신뢰구간임
sns.regplot(data=grouped_data , x='rank', y='log_odds')  # 파란색 범위는 신뢰구간임

import statsmodels.api as sm
model = sm.formula.ols("log_adds ~ rank", data = groupded_data)
print(model.summary())
# rank가 한 단위 증가할 때마다 y(로그 오즈)가 0.5675만큼 준다 <- 근데 해석이 안됨. 로그 오즈가 뭔데
# 해석을 하려면 양쪽에 지수를 계산해줘야함
# 원래 모델 : 로그(p(x) / (1-p(x))) = 절편 + 기울기*x
# 지수 계산한 모델 : p(x)/(1-p(x)) = exp(절편 + 기울기*x)



admission['rank'] = admission['rank'].astype('category')
admission['gender'] = admission['gender'].astype('category')
model = sm.formula.logit("admit ~ gre + gpa + rank + gender", data=admission).fit() # 카테고리 처리를 한 후 돌리면, 카테고리인 변수들은 더미코딩 돼서 모델 추정을 하고, 수치를 가지는 범주형 컬럼을 카테고리 처리 안하면 수치변수로 보고 모델 추정함. 문자를 가지는 범주형 컬럼을 카테고리 처리안하고 돌려도 더미코딩해줌.



admission['gender'] = admission['gender'].astype('category')
model = sm.formula.logit("admit ~ gre + gpa + rank + gender", data=admission).fit() # 카테고리 처리를 한 후 돌리면, 카테고리인 변수들은 더미코딩 돼서 모델 추정을 하고, 수치를 가지는 범주형 컬럼을 카테고리 처리 안하면 수치변수로 보고 모델 추정함. 문자를 가지는 범주형 컬럼을 카테고리 처리안하고 돌려도 더미코딩해줌.
print(model.summary())

# 여학생, GPA 3.5, GRE 500, Rank 2 일 때 합격확률 예측하기
np.exp(-3.4075+0.0023*500+0.7753*3.5-0.5614*2) # 오즈 0.513
a = np.exp(-3.4075+0.0023*500+0.7753*3.5-0.5614*2)/(np.exp(-3.4075+0.0023*500+0.7753*3.5-0.5614*2)+1)
# 0.339
# 오즈 : 0.339 / (1-0.339)  # 0.513

# 이상태에서 GPA가 1 증가하면 합격확률이 어떻게 변하는지 구해보기
np.exp(-3.4075+0.0023*500+0.7753*4.5-0.5614*2)  # 오즈 1.115 
b = np.exp(-3.4075+0.0023*500+0.7753*4.5-0.5614*2)/(np.exp(-3.4075+0.0023*500+0.7753*4.5-0.5614*2)+1)
# 0.527
# 오즈 : 0.527 / (1-0.527)  # 1.114
# 오즈로 보면 gpa가 1단위 증가 전과 후가 2배하고도 조금 더 차이남


# 여학생, GPA 3, GRE 450, Rank 2, 합격 확률과 odds는?
np.exp(-3.4075+0.0023*450+0.7753*3-0.5614*2) # 오즈 0.3105
np.exp(-3.4075+0.0023*450+0.7753*3-0.5614*2)/(np.exp(-3.4075+0.0023*450+0.7753*3-0.5614*2)+1) # 오즈 0.237


print(model.summary()) 
# 위 결과를 보면
# gender[T.M] 을 보면 p_value 0.8 >0.05 임. 귀무가설 베타 =0, 대립가설 베타 0 아님. 귀무가설을 가각하지 못함. 그리면 베타가 0이라는 것임. 즉, 해당 변수는 없어짐.
# 성별 변수로는 해석할 수 없다는 것임. (성별 여성도)
# Z 는 표준정규분포를 따르는 검정통계량임
# 여기서 나오는 신뢰구간은 베타에 대한 신뢰구간임
# 근데 내가 알고 싶은 건 오즈비에 대한 신뢰구간임. 그럼 지수를 취하면 오즈비에 대한 신뢰구간을 구할 수 있음
# exp(신뢰구간 L) , exp(신뢰구간 U)

# 결과 보면 Log-Likelihood(l), LL-Null(l(0)) 있음
# -2*(l(0)) - l ~ 카이제곱(k-r)  <- 구한 통계량은 결과에 나온 LLR p-value임.
# LLR p-value : 모델이 유의한 지를 나타내는 지표
# 유의수준 0.05 보다 작으면 모델이 쓸만하다 (즉, 0에 가까울 수록 모델이 믿을 만 하다)


