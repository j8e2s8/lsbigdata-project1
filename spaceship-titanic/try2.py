import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report



## 필요한 데이터 불러오기
train=pd.read_csv("./spaceship-titanic/train.csv")
test=pd.read_csv("./spaceship-titanic/test.csv")
sub_df=pd.read_csv("./spaceship-titanic/sample_submission.csv")

train.columns
train.head()
train.info()
train['Destination'].unique()


## 필요없는 칼럼 제거
all_data=pd.concat([train, test])

#cabin split => deck/num/side
all_data[['Cabin_Deck', 'Cabin_Number', 'Cabin_Side']] = all_data['Cabin'].str.split('/', expand=True)
all_data = all_data.drop(['Cabin','Cabin_Number'], axis=1)

all_data = all_data.drop(['PassengerId', 'Name'], axis=1)

# 범주형 칼럼
c = all_data.columns[all_data.dtypes == object].drop('Transported')
all_data.info()

# 정수형 전처리 : 범주형을 숫자로 바꿈
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()

for i in c:
    all_data[i] = le.fit_transform(all_data[i])


# 결측치 처리
all_data.fillna(-1, inplace=True)

# train / test 데이터셋
train_n=len(train)
train=all_data.iloc[:train_n,]
test=all_data.iloc[train_n:,]

# 타겟 변수 분리
y = train['Transported'].astype("bool")
train.info()
train.head()
c

# OneHotEncoder 설정
preprocessor = ColumnTransformer(
    transformers=[
        ('cat', OneHotEncoder(drop='first',  # 더미코딩이 됨. 근데 onehotencoding이 더 성능이 좋은 것음.
                              handle_unknown='ignore'), c)
    ], remainder='passthrough')

# 학습 데이터와 테스트 데이터에 전처리 적용
X_train = preprocessor.fit_transform(train.drop(['Transported'], axis=1))
X_test = preprocessor.transform(test)




# 랜포
param_grid1 = {
    'n_estimators': [50, 100, 200],  # 트리의 개수
    'max_depth': [None, 10, 20, 30], # 최대 깊이
    'min_samples_split': [2, 5, 10], # 분할 시 필요한 최소 샘플 수
    'min_samples_leaf': [1, 2, 4],   # 리프 노드의 최소 샘플 수
    'max_features': ['auto', 'sqrt', 'log2'], # 각 트리가 사용할 최대 특성 수
    'bootstrap': [True, False]       # 부트스트래핑 사용 여부
}
rf_model = RandomForestClassifier(random_state=42)
grid_search1 = GridSearchCV(estimator=rf_model, 
                            param_grid=param_grid1, 
                            cv=5, 
                            n_jobs=-1, 
                            verbose=2)

grid_search1.fit(X_train, y)

grid_search1.best_params_   
# {'bootstrap': True,
#  'max_depth': 10,
#  'max_features': 'sqrt',
#  'min_samples_leaf': 1,
#  'min_samples_split': 2,
#  'n_estimators': 50}


# {'bootstrap': False,
#  'max_depth': None,
#  'max_features': 'sqrt',
#  'min_samples_leaf': 1,
#  'min_samples_split': 10,
#  'n_estimators': 100}

grid_search1.best_score_   # 0.8024873096782604   # 0.7942051004803496

best_rf_model=grid_search1.best_estimator_ 



# 로지스틱 회귀분석
param_grid2 = {
    'C': [0.1, 1, 10, 100],
    'penalty': ['l1', 'l2'],
    'solver': ['liblinear', 'lbfgs']
}

logistic_model = LogisticRegression(max_iter=1000, random_state=42)

grid_search2 = GridSearchCV(estimator=logistic_model, 
                            param_grid=param_grid2, 
                            cv=5,   # 데이터 개수가 8693개라서 cv값 늘리면 validset이 너무 작아져서 알 될 것 같음
                            n_jobs=-1, 
                            scoring='accuracy')

grid_search2.fit(X_train, y)

grid_search2.best_params_  # {'C': 0.1, 'penalty': 'l2', 'solver': 'lbfgs'}   # {'C': 10, 'penalty': 'l1', 'solver': 'liblinear'}
grid_search2.best_score_  # 0.7885656412723474  # 0.78856663386693

best_lr_model=grid_search2.best_estimator_ 


# SVC
from sklearn.svm import SVC

standard_scaler = StandardScaler()
standardized_data = standard_scaler.fit_transform(X_train)
# standard_train_x = pd.DataFrame(standardized_data, columns = X_train.columns)

param_grid3 = {
    'C': [0.1, 1, 10, 100],                # 규제 매개변수
    'kernel': ['linear', 'poly', 'rbf', 'sigmoid'],  # 커널 유형
    'gamma': [0.001, 0.01, 0.1, 1, 10],    # 커널 계수
    'degree': [2, 3, 4, 5]                  # 'poly' 커널의 경우 다항식 차수
}

svc_model = SVC(probability=True)  # SVC는 random_state를 지원하지 않는다 함


grid_search3 = GridSearchCV(estimator=svc_model
                    , param_grid=param_grid3
                    , cv=5
                    , n_jobs=-1
                    , verbose=2)


grid_search3.fit(standardized_data, y)

grid_search3.best_params_  
grid_search3.best_score_  

best_svc_model=grid_search3.best_estimator_ 


# xgboost 분류
from xgboost import XGBClassifier
from sklearn.model_selection import RandomizedSearchCV

# XGBoost 모델 초기화
xgb_model = XGBClassifier(objective='binary:logistic',  # 이진 분류를 위한 목적 함수
                          eval_metric='logloss',  # 로지스틱 손실 함수
                          use_label_encoder=False,
                          random_state=42)

# 하이퍼파라미터 분포
param_dist = {
    'n_estimators': [100, 200, 300],
    'learning_rate': [0.01, 0.1, 0.2],
    'max_depth': [3, 4, 5],
    'min_child_weight': [1, 2, 3],
    'subsample': [0.8, 0.9, 1.0],
    'colsample_bytree': [0.8, 0.9, 1.0],
    'gamma': [0, 0.1, 0.2],
}

# RandomizedSearchCV 설정
random_search = RandomizedSearchCV(estimator=xgb_model, 
                                    param_distributions=param_dist, 
                                    n_iter=50,  # 시도할 조합 수
                                    cv=5, 
                                    scoring='accuracy', 
                                    verbose=2, 
                                    n_jobs=-1, 
                                    random_state=42)

# 모델 학습
random_search.fit(X_train, y)

random_search.best_params_  
# {'subsample': 0.8,
#  'n_estimators': 200,
#  'min_child_weight': 1,
#  'max_depth': 3,
#  'learning_rate': 0.1,
#  'gamma': 0.2,
#  'colsample_bytree': 0.9}
random_search.best_score_  # 0.805131846338418

best_xg_model=random_search.best_estimator_ 




# 새로운 데이터 셋 만들기
train_pred_y_rf = best_rf_model.predict_proba(X_train)[:,1] 
train_pred_y_lr = best_lr_model.predict_proba(X_train)[:,1] 
# train_pred_y_svc = best_svc_model.predict_proba(X_train)[:,1] 
train_pred_y_xg = best_xg_model.predict_proba(X_train)[:,1] 

stacking_train_x = pd.DataFrame({
    'y1' : train_pred_y_rf,
    'y2' : train_pred_y_lr,
 #   'y3' : train_pred_y_svc,
    'y4' : train_pred_y_xg,
})


test_pred_y_rf = best_rf_model.predict_proba(X_test)[:,1] 
test_pred_y_lr = best_lr_model.predict_proba(X_test)[:,1]
# test_pred_y_svc = best_svc_model.predict_proba(X_test)[:,1] 
test_pred_y_xg = best_xg_model.predict_proba(X_test)[:,1]

stacking_test_x=pd.DataFrame({
    'y1' : test_pred_y_rf,
    'y2' : test_pred_y_lr,
 #   'y3' : test_pred_y_svc,
    'y4' : test_pred_y_xg
})



# 최종 확률 예측을 위한 로지스틱 회귀 모델 / xg 분류 모델

xgb_model = XGBClassifier(objective='binary:logistic',  # 이진 분류를 위한 목적 함수
                          eval_metric='logloss',  # 로지스틱 손실 함수
                          use_label_encoder=False)

# 하이퍼파라미터 분포
param_dist2 = {
    'n_estimators': [100, 200, 300],
    'learning_rate': [0.01, 0.1, 0.2],
    'max_depth': [3, 4, 5],
    'min_child_weight': [1, 2, 3],
    'subsample': [0.8, 0.9, 1.0],
    'colsample_bytree': [0.8, 0.9, 1.0],
    'gamma': [0, 0.1, 0.2],
}
# RandomizedSearchCV 설정
random_search2 = RandomizedSearchCV(estimator=xgb_model, 
                                    param_distributions=param_dist2, 
                                    n_iter=50,  # 시도할 조합 수
                                    cv=5, 
                                    scoring='accuracy', 
                                    verbose=2, 
                                    n_jobs=-1, 
                                    random_state=42)

# 모델 학습
random_search2.fit(stacking_train_x, y)

random_search2.best_params_  
# {'subsample': 1.0,
#  'n_estimators': 300,
#  'min_child_weight': 1,
#  'max_depth': 4,
#  'learning_rate': 0.2,
#  'gamma': 0.2,
#  'colsample_bytree': 1.0}
random_search2.best_score_   # 0.862651842156286

blander_model = random_search2.best_estimator_ 



# 로지스틱 이었을 때 씀
# grid_search4.fit(stacking_train_x, y) 

# grid_search4.best_params_  

# {'C': 100,
#  'class_weight': 'balanced',
#  'max_iter': 100,
#  'penalty': 'l1',
#  'solver': 'saga'}

# {'C': 100,
#  'class_weight': None,
#  'max_iter': 100,
#  'penalty': 'l2',
#  'solver': 'newton-cg'}

# grid_search4.best_score_   # 0.8517234419739133  # 0.9986196979733204

# blander_model = grid_search4.best_estimator_

# blander_model.coef_ # array([0.3583818 , 0.70122774]) <- 랜포가 더 좋다고 생각해서 비중을 더 키운 것임
# blander_model.intercept_  # -10825.640652675705  <- 두 모델 오버슈팅 된 것 같아서 줄여줌


pred_y=blander_model.predict(stacking_test_x)
pred_y = pred_y.astype(bool)

# Transported 바꾸기
sub_df["Transported"] = pred_y
sub_df

# csv 파일로 내보내기
sub_df.to_csv("./spaceship-titanic/rf_lr_xg_stacking(xg).csv", index=False)