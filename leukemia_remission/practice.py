import pandas as pd
import statsmodels.api as sm



# 문제1 : 데이터를 로드하고, 로지스틱 회귀모델을 적합하고, 회귀 표를 작성하세요.
df = pd.read_csv('./leukemia_remission/leukemia_remission.txt', delim_whitespace= True) # delim_whitespace : 공백 지우기
df.head()

train = df.drop(columns=('REMISS'))

model = sm.formula.logit("REMISS ~ CELL + SMEAR + INFIL + LI + BLAST + TEMP", data=df).fit()
print(model.summary())



# 문제2 : 해당 모델은 통계적으로 유의한가요? 그 이유를 검정통계량를 사용해서 설명하시오
# −2(ℓ(𝛽)̂ (0) − ℓ(𝛽)̂ )  =  -2*(-17.186+10.797)  = 12.779
1 - chi2.cdf(12.779, df=6)  # 0.0467

# LLR p-value: 0.0467 < 유의수준 0.05보다 작으니까 통계적으로 유의하다고 할 수 있다.




# 문제3 : 유의수준이 0.2를 기준으로 통계적으로 유의한 변수는 몇개이며, 어느 변수 인가요?
# P>|z|가 0.2보다 작은 LI, TEMP가 유의하다



# 문제4 : 다음 환자에 대한 오즈는 얼마인가요?

# CELL (골수의 세포성): 65%

# SMEAR (골수편의 백혈구 비율): 45%

# INFIL (골수의 백혈병 세포 침투 비율): 55%

# LI (골수 백혈병 세포의 라벨링 인덱스): 1.2

# BLAST (말초혈액의 백혈병 세포 수): 1.1세포/μL

# TEMP (치료 시작 전 최고 체온): 0.9


odds = np.exp( 64.2581 + 30.8301*0.65 + 24.6863 * 0.45  - 24.9745 * 0.55 + 4.3605 * 1.2 - 0.0115*1.1 - 100.1734 * 0.9)  # 0.03817459641135519



# 문제5 : 위 환자의 혈액에서 백혈병 세포가 관측되지 않은 확률은 얼마인가요?
odds / (1+odds)  # 0.03677088280074742



# 문제 6 : TEMP 변수의 계수는 얼마이며, 해당 계수를 사용해서 TEMP 변수가 백혈병 치료에 대한 영향을 설명하시오.
# TEMP 변수의 계수 : -100.1734
# TEMP 변수 1 단위 증가하면 로그오즈가 100.1734 감소하는데, 오즈비는 (e^-100.1734)배 감소한다. TEMP가 1단위 올라가면, 오즈비가 (e^-100.1734)배 감소. 
# 관측 불가 확률이 감소한다. 즉 관측될 확률이 증가한다. 따라서 백혈병 치료를 해야한다.
# np.exp(-100.1734)은 0에 가까운 값이다. 이는 체온이 1단위 상승할 때 백혈병 세포가 관측되지 않을 확률이 거의 없어지는 것을 의미(오즈비만큼 변동)한다. 따라서, 온도가 높아질수록 백혈병 세포가 관측될 확률 높아진다.

np.exp(-100.1734)  # 0

temp = 0 일 때, e^0 =1
temp =1 일 때 e^-100

temp = 0 일 때, 로그오즈(y) = 200   # y변화량 = -100.1734  -> 오즈 = e^(200) = 7.225973768125749e+86
temp = 1 일 때, 로그오즈(y) = 200-100.1734               # ->  오즈 = e^(99.8266) = e^200 * e^(-100.1734) = 2.2601721913757316e+43  # e^(-100.1734)배'로/만큼' 감소




temp = 0 일 때, 로그오즈(y) = 5 # y변화량 = 7-5 = 2 -> 오즈 = e^(5) = 148.4131591025766
temp = 1 일 때, 로그오즈(y) = 7                  # ->  오즈 = e^(5+2) = e^5 * e^2 = 1096.6331584284585  # e^(2) = e^(5+2) / e^(5) = 7.3890560989306495배로 증가 => 기존값의 **배로 증가

temp = 0 일 때, 로그오즈(y) = 5 # y변화량 = 5-3 = 2 -> 오즈 = e^(5) = 148.4131591025766
temp = 1 일 때, 로그오즈(y) = 3                  # ->  오즈 = e^(5-2) = e^5 * e^(-2) = 20.085536923187668  # e^(-2) = e^(5-2) / e^(5) = 0.1353352832366127배로 감소 => 기존값의 **배로 감소




# 문제 7 : CELL 변수의 99% 오즈비에 대한 신뢰구간을 구하시오.
# CELL 변수의 베타에 대한 99 : (베타_hat - z(0.005)*SE , 베타_hat + z(0.005)*SE)
z0005 = norm.ppf(0.995, loc=0, scale=1)
30.8301 - 52.135*z0005 , 30.8301 + 52.135*z0005
np.exp(30.8301 - 52.135*z0005) , np.exp(30.8301 + 52.135*z0005)  # (1.1683218982002717e-45, 5.141881884993857e+71)


# 문제 8 : 주어진 데이터에 대하여 로지스틱 회귀 모델의 예측 확률을 구한 후, 50% 이상인 경우 1로 처리하여, 혼동 행렬를 구하시오.
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay


y_pred = model.predict(train)
result = pd.DataFrame({'y_pred' : y_pred})
result['result'] = np.where(result['y_pred']>=0.5, 1,0)

conf_mat = confusion_matrix(y_true = df['REMISS'], y_pred = result['result'], labels=[1,0])
p = ConfusionMatrixDisplay(confusion_matrix = conf_mat, display_labels = ('관측불가_1', '관측가능_0'))
plt.rcParams['font.family'] = 'Malgun Gothic'
p.plot(cmap="Blues")



# 문제 9 : 해당 모델의 Accuracy는 얼마인가요?
(5+15)/(5+3+4+15)  # 0.7407407407407407

from sklearn.metrics import accuracy_score, f1_score
accuracy_score(df['REMISS'], result['result'])  # 0.7407407407407407



# 문제10 : 해당 모델의 F1 Score를 구하세요.
precision = 5/(5+3)
recall = 5/(5+4)

 2 / (1/precision + 1/recall)   # 0.5882352941176471


f1_score(df['REMISS'], result['result'])  # 0.5882352941176471




# 정리 ------------------------------------
import pandas as pd
import numpy as np
import statsmodels.api as sm


# 문제1 : 데이터를 로드하고, 로지스틱 회귀모델을 적합하고, 회귀 표를 작성하세요.
df = pd.read_csv('./leukemia_remission/leukemia_remission.txt', delim_whitespace= True) # delim_whitespace : 공백 지우기
df.head()
train = df.drop(columns=('REMISS'))

model = sm.formula.logit("REMISS ~ CELL + SMEAR + INFIL + LI + BLAST + TEMP", data=df).fit()
print(model.summary())



# 문제2 : 해당 모델은 통계적으로 유의한가요? 그 이유를 검정통계량를 사용해서 설명하시오
# 검정통계량 : −2(ℓ(𝛽)̂ (0) − ℓ(𝛽)̂ )  =  -2*(-17.186+10.797)  = 12.779
1 - chi2.cdf(12.779, df=6)  # 0.0467

# 결론 : LLR p-value: 0.0467 < 유의수준 0.05보다 작으니까 통계적으로 유의하다고 할 수 있다.




# 문제3 : 유의수준이 0.2를 기준으로 통계적으로 유의한 변수는 몇개이며, 어느 변수 인가요?
# P>|z|가 0.2보다 작은 LI, TEMP가 유의하다



# 문제4 : 다음 환자에 대한 오즈는 얼마인가요?
# CELL (골수의 세포성): 65%
# SMEAR (골수편의 백혈구 비율): 45%
# INFIL (골수의 백혈병 세포 침투 비율): 55%
# LI (골수 백혈병 세포의 라벨링 인덱스): 1.2
# BLAST (말초혈액의 백혈병 세포 수): 1.1세포/μL
# TEMP (치료 시작 전 최고 체온): 0.9

odds = np.exp( 64.2581 + 30.8301*0.65 + 24.6863 * 0.45  - 24.9745 * 0.55 + 4.3605 * 1.2 - 0.0115*1.1 - 100.1734 * 0.9)  
# 오즈 : 0.03817459641135519


# 문제5 : 위 환자의 혈액에서 백혈병 세포가 관측되지 않은 확률은 얼마인가요?
odds / (1+odds)  # 0.03677088280074742



# 문제 6 : TEMP 변수의 계수는 얼마이며, 해당 계수를 사용해서 TEMP 변수가 백혈병 치료에 대한 영향을 설명하시오.
# TEMP 변수의 계수 : -100.1734
# e^(-100.1734) = 3.13e-44 는 0에 가까운 값입니다. 이는 체온이 1단위 상승할 때 백혈병 세포가 관측되지 않을 확률이 (오즈비만큼 변동)거의 없어지는 것을 의미 ->  온도가 높아질수록 백혈병 세포가 관측될 확률 높아짐.
# np.exp(-100.1734)은 0에 가까운 값이다. 이는 체온이 1단위 상승할 때 백혈병 세포가 관측되지 않을 확률이 거의 없어지는 것을 의미(오즈비만큼 변동)한다. 따라서, 온도가 높아질수록 백혈병 세포가 관측될 확률 높아진다.


# 문제 7 : CELL 변수의 99% 오즈비에 대한 신뢰구간을 구하시오.
# CELL 변수의 베타에 대한 99% 신뢰구간 : (베타_hat - z(0.005)*SE , 베타_hat + z(0.005)*SE)
# CELL 변수의 오즈비에 대한 99% 신뢰구간 : (exp(베타_hat - z(0.005)*SE) , exp(베타_hat + z(0.005)*SE))

z0005 = norm.ppf(0.995, loc=0, scale=1)  # 2.5758293035489004

# CELL 변수의 베타에 대한 99% 신뢰구간 : (-103.4607607405219, 165.12096074052192
30.8301 - 52.135*z0005 , 30.8301 + 52.135*z0005 

# CELL 변수의 오즈비에 대한 99% 신뢰구간 : (1.1683218982002717e-45, 5.141881884993857e+71)
np.exp(30.8301 - 52.135*z0005) , np.exp(30.8301 + 52.135*z0005)  



# 문제 8 : 주어진 데이터에 대하여 로지스틱 회귀 모델의 예측 확률을 구한 후, 50% 이상인 경우 1로 처리하여, 혼동 행렬를 구하시오.
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

y_pred = model.predict(train)
result = pd.DataFrame({'y_pred' : y_pred})
result['result'] = np.where(result['y_pred']>=0.5, 1,0)

conf_mat = confusion_matrix(y_true = df['REMISS'], y_pred = result['result'], labels=[1,0])
p = ConfusionMatrixDisplay(confusion_matrix = conf_mat, display_labels = ('관측불가_1', '관측가능_0'))
plt.rcParams['font.family'] = 'Malgun Gothic'
p.plot(cmap="Blues")



# 문제 9 : 해당 모델의 Accuracy는 얼마인가요?
(5+15)/(5+3+4+15)  # 0.7407407407407407

from sklearn.metrics import accuracy_score, f1_score
accuracy_score(df['REMISS'], result['result'])  # 0.7407407407407407



# 문제10 : 해당 모델의 F1 Score를 구하세요.
precision = 5/(5+3)
recall = 5/(5+4)
 2 / (1/precision + 1/recall)   # 0.5882352941176471

f1_score(df['REMISS'], result['result'])  # 0.5882352941176471
