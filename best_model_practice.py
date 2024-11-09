# 펭귄 데이터 부리길이 예측 모형 만들기
# 엘리스틱 넷 & 디시전트리 회귀모델 사용
# 모든 변수 자유롭게 사용!
# 종속변수 : bill_length_mm


from palmerpenguins import load_penguins
import pandas as pd
import numpy as np
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import ElasticNet
from sklearn.model_selection import train_test_split


df = load_penguins()
penguins = df.dropna()

df.columns
# 'species', 'island', 'bill_length_mm', 'bill_depth_mm','flipper_length_mm', 'body_mass_g', 'sex', 'year'
df.info()  # 'species', 'island', 'sex' : 범주





penguins_dummies = pd.get_dummies(
                        penguins, 
                        columns=['species', 'island', 'sex'],
                        drop_first=True)

penguins_dummies.columns

x = penguins_dummies.drop(columns=('bill_length_mm'))
y = penguins_dummies['bill_length_mm']

len(penguins_dummies.columns)
len(x.columns)


train_x, test_x, train_y, test_y = train_test_split(x,y, test_size=0.2, random_state=42)




np.random.seed(20240905)

model = ElasticNet()  # alpha: 람다, l1_ratio : 알파

param_grid = {    # 하이퍼 파라미터 후보군들
    'alpha' : np.arange(0, 5 , 0.01),     
    'l1_ratio' : np.arange(0 , 1 , 0.01)
}


grid_search = GridSearchCV(
    estimator = model,
    param_grid = param_grid,
    scoring = 'neg_mean_squared_error',
    cv=5
)

grid_search.fit(train_x, train_y)


grid_search.best_params_ 
grid_search.cv_results_
result = pd.DataFrame(grid_search.cv_results_)
grid_search.best_score_    % 5.122
# alpha(0) -> 람다 패널티 0 일반 선형 회귀분석 , l1_ratio(0) -> alpha=0 릿지 회귀분석  => 결론 일반 선형 회귀분석




best_model = grid_search.best_estimator_ 
pred_y1 = best_model.predict(test_x)  # 바로 최적의 모델로 예측할 수 있음.




print("elasticnet test MSE :",np.mean((pred_y1 - test_y)**2))  # 5.642






# ------------------------------


from sklearn.tree import DecisionTreeRegressor



np.random.seed(20240905)
model2 = DecisionTreeRegressor()  

param_grid = {    # 하이퍼 파라미터 후보군들
    'max_depth' : np.arange(0, 7),     
    'min_samples_split' : np.arange(2, 5 )
}


grid_search = GridSearchCV(
    estimator = model2,
    param_grid = param_grid,
    scoring = 'neg_mean_squared_error',
    cv=5
)

grid_search.fit(train_x, train_y)


grid_search.best_params_ 
grid_search.cv_results_
result = pd.DataFrame(grid_search.cv_results_)
grid_search.best_score_  # 7.832

# 최적의 하이퍼파라미터 : max_depth 5, min_samples_split 2

best_model2 = grid_search.best_estimator_  
pred_y2 = best_model2.predict(test_x)  


print( "의사결정나무 test MSE :",np.mean((pred_y2 - test_y)**2))  # 7.293


from sklearn import tree
import matplotlib.pyplot as plt
plt.figure(figsize=(50,50))
tree.plot_tree(best_model2)


tree.plot_tree(clf, filled = True);








# --------------------------------------------------
import numpy as np
import pandas as pd
from palmerpenguins import load_penguins

penguins = load_penguins()
penguins.head()


df = penguins.dropna()
df = df[["bill_length_mm", "bill_depth_mm"]]
df

train_x = df['bill_length_mm']
train_x = train_x.values.reshape(-1, 1)
train_y = df['bill_depth_mm']
train_y = train_y.values.reshape(-1, 1)


from sklearn.linear_model import ElasticNet
from sklearn.model_selection import GridSearchCV
# 커스텀 함수 정의 (외부에서 파라미터 그리드를 받음)
def perform_grid_search(train_x, train_y, param_grid, cv=5, scoring='neg_mean_squared_error'):
    # ElasticNet 모델과 GridSearchCV 설정
    model = ElasticNet()
    grid_search = GridSearchCV(model, param_grid, scoring=scoring, cv=cv)
    grid_search.fit(train_x, train_y)
    
    # 최적의 파라미터, 최적의 모델, 점수 반환
    best_params = grid_search.best_params_
    best_model = grid_search.best_estimator_
    best_score = grid_search.best_score_
    

    print(f"Best Params: {best_params}")
    print(f"Best Score (neg MSE): {best_score}")
    
    return best_model, best_params, best_score

# 함수 밖에서 파라미터 그리드 설정   알파 : 1.0, l1 ratio : 0.2, neg MSE : -5.9871...
param_grid = {                         # 0.1             1.0
    'alpha': [0.01, 0.1, 1.0],
    'l1_ratio': [0.2, 0.5, 0.8]
}

# 함수 호출 시 파라미터 그리드 전달
best_model, best_params, best_score = perform_grid_search(train_x, train_y, param_grid)




-(3/4)*np.log2(3/4)-(1/4)*np.log2(1/4) # 0.8112781244591328
0.8112781244591328*4/5

from scipy.stats import poisson, expon, uniform
poisson.rvs(mu=4, size=10, random_state=42)
np.random.poisson(4, 5)
poisson.pmf(3, 4 )  # 0.19536681481316454
poisson.cdf(3,4)  # 0.43347012036670896
poisson.ppf(0.433, mu=4)


# 포아송 분포의 mu lambda값을 역수로 넣어줘야함. 그래야 lambda*exp(-lambda*x)로 계산됨.
expon.pdf(3,1/4) # 0.06392786120670757 
expon.cdf(55, 1/4)
expon.ppf(0.0639, 1/4)






# 분류 의사결정나무
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from palmerpenguins import load_penguins

penguins = load_penguins()
penguins.head()

df.columns
df=penguins.dropna()
df=df[["species","bill_length_mm", "bill_depth_mm"]]
df=df.rename(columns={'species' : 'y',
                    'bill_length_mm': 'x1',
                   'bill_depth_mm': 'x2'})
df

sns.scatterplot(data=df, x='x1', y='x2', hue='y')
plt.axvline(x=45)

# 나누기전 엔트로피와 나눈 후 엔트로피 비교
info_df = df[['y']].value_counts() / len(df)

befor_e = -(info_df[0]*np.log2(info_df[0]) + info_df[1]*np.log2(info_df[1]) + info_df[2]*np.log2(info_df[2]))
befor_e = -sum(info_df * np.log2(info_df))  

df_left = df[df['x1']<45]
df_right = df[df['x1']>= 45]
info_df_left = df_left[['y']].value_counts() / len(df_left)
info_df_right = df_right[['y']].value_counts() / len(df_right)

after_e_left = -(info_df_left[0]*np.log2(info_df_left[0]) + info_df_left[1]*np.log2(info_df_left[1]) +info_df_left[2]*np.log2(info_df_left[2]))
-sum(info_df_left*np.log2(info_df_left))
after_e_right = -(info_df_right[0]*np.log2(info_df_right[0]) + info_df_right[1]*np.log2(info_df_right[1]) +info_df_right[2]*np.log2(info_df_right[2]))
-sum(info_df_right*np.log2(info_df_right))

after_e_left * len(df_left)/len(df) + after_e_right * len(df_right)/len(df)





def entropy(df, col):
    entropy_i = []
    for i in range(len(df[col].unique())):
        df_left = df[df[col]< df[col].unique()[i]]
        df_right = df[df[col]>= df[col].unique()[i]]
        info_df_left = df_left[['y']].value_counts() / len(df_left)
        info_df_right = df_right[['y']].value_counts() / len(df_right)
        after_e_left = -sum(info_df_left*np.log2(info_df_left))
        after_e_right = -sum(info_df_right*np.log2(info_df_right))
        entropy_i.append(after_e_left * len(df_left)/len(df) + after_e_right * len(df_right)/len(df))
    return entropy_i


entropy_df = pd.DataFrame({ 'standard': df1['x2'].unique(),
                          'entropy' : entropy(df1, 'x2') })

entropy_df.iloc[np.argmin(entropy_df['entropy']),:]

# 기준 42.4, entropy 0.804


sns.scatterplot(data=df, x='x1', y='x2', hue='y')
plt.axvline(x=42.4)


df1 = df[df['x1']<42.4]





train_x = df[['x1', 'x2']]
train_y = df['y']
from sklearn.tree import DecisionTreeClassifier 
# from sklearn.model_selection import GridSearchCV

# 범주형인 y값을 예측할 때 씀. x중 범주형은 더미코딩 하기
# nan 결측치 제거 안 해도 분류가 되긴함. nan인 애들을 분류하고, nan이 아닌 애들을 분류하기 때문
model = DecisionTreeClassifier(random_state=2024, max_depth=2)
model.fit(train_x, train_y)
model.predict(train_x)


model = DecisionTreeClassifier(random_state=2024, criterion = 'entropy')
# criterion : gini, entropy, log_loss  (지표가 3가지 있음)


param_grid = {
    'max_depth' : np.arange(7,20,1),
    'min_samples_split' : np.arange(10, 30 , 1)
}

np.random.seed(20240906)
grid_search = GridSearchCV(
    estimator = model,
    param_grid = param_grid,
    scoring = 'accuracy',
    cv=5
)


grid_search.fit(train_x, train_y)



from sklearn import tree
tree.plot_tree(model)



# 성능 비교 행렬
from sklearn.metrics import confusion_matrix
# 아델리: 'A'
# 친스트랩(아델리가 아닌 것): 'C'

y_true = np.array(['A', 'A', 'C', 'A' , 'C', 'C', 'C'])
y_pred1 = np.array(['A', 'C', 'A', 'A', 'A', 'C', 'C'])
y_pred2 = np.array(['C', 'A', 'A', 'A', 'C', 'C', 'C'])

conf_mat = confusion_matrix(y_true = y_true,
                            y_pred = y_pred1,
                            labels=['A', 'C'])

conf_mat

conf_mat2 = confusion_matrix(y_true = y_true,
                            y_pred = y_pred2,
                            labels=['A', 'C'])

conf_mat2

from sklearn.metrics import ConfusionMatrixDisplay
p = ConfusionMatrixDisplay(confusion_matrix = conf_mat,  # 내가 계산한 conf_mat를 입력값으로 넣음
                           display_labels=('Adelie', 'Chinstrap'))
p.plot(cmap="Blues")


p = ConfusionMatrixDisplay(confusion_matrix = conf_mat2,  # 내가 계산한 conf_mat를 입력값으로 넣음
                           display_labels=('Adelie', 'Chinstrap'))
p.plot(cmap="Blues")




# 
from sklearn.ensemble import BaggingClassifier
from sklearn.tree import DecisionTreeClassifier

bagging_model = BaggingClassifier(DecisionTreeClassifier(), n_estimator=50, max_sample=100, n_jobs=-1, random_state=42)
  # for문 처럼 decisiontreeclassifier 모델을 50번 fit한다.(bagging에 사용할 모델 개수 50개) max_sample : 하나의 subdata를 행 100으로 한다. , n_jobs=-1 : 계산 메모리 4개를 동시에 사용해서 한 번에 모델 4개를 fit할 수 있다. ,random_state : 나중에 bootstrap으로 할 때 동일한 데이터로 하기 위해서
  # 이건 랜덤포레스트 아이디어임. (랜덤포레스트 옵션과, 이렇게 할 때 하는 옵션이 조금 다르긴 함)
  # baggingclassifier은 bootstrap도 되고, 열도 랜덤하게 선택할 수 있음(random sub space : sub space마다 열 달라짐) <- 랜덤포레스트의 더 general한 경우
  # 랜덤포레스트는 행 랜덤 추출만(bootstrap)

bagging_model.fit(train_x, train_y)


#
from sklearn.ensamble import RandomForestClassifier
rf_model