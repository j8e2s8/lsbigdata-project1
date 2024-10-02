import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.stats import norm, uniform
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.linear_model import Lasso, Ridge, LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import cross_val_score, KFold
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
from sklearn.metrics import make_scorer, mean_squared_error
import statsmodels.api as sm


train_df = pd.read_csv('./blueberry/train.csv')
train_df = train_df.drop(columns=('id'))
test_df = pd.read_csv('./blueberry/test.csv')
test_df = test_df.drop(columns= ('id'))
submission = pd.read_csv('./blueberry/sample_submission.csv')

train_df.head()
train_df.info()
train_df.columns
# 'clonesize', 'honeybee', 'bumbles', 'andrena', 'osmia','MaxOfUpperTRange',
# 'MinOfUpperTRange', 'AverageOfUpperTRange','MaxOfLowerTRange', 
# 'MinOfLowerTRange', 'AverageOfLowerTRange', 'RainingDays', 
# 'AverageRainingDays', 'fruitset', 'fruitmass', 'seeds', 'yield'

# 각 변수별 히스토그램 확인
# 'clonesize', 'honeybee', 'bumbles', 'andrena', 'osmia','MaxOfUpperTRange',
# 'MinOfUpperTRange', 'AverageOfUpperTRange','MaxOfLowerTRange', 
# 'MinOfLowerTRange', 'AverageOfLowerTRange', 'RainingDays', 
# 'AverageRainingDays'  <- 수치에 의미가 있는 범주형 컬럼으로 만들기

# 'fruitset', 'fruitmass', 'seeds' <- 수치 컬럼


df = pd.concat([train_df, test_df])
len(df)

cat_col = ['clonesize', 'honeybee', 'bumbles', 'andrena', 'osmia','MaxOfUpperTRange','MinOfUpperTRange', 'AverageOfUpperTRange','MaxOfLowerTRange', 'MinOfLowerTRange', 'AverageOfLowerTRange', 'RainingDays', 'AverageRainingDays']
for i in cat_col:
    df[i] = pd.Categorical(df[i], categories = df[i].unique(), ordered = True)


df.info()


df_dummies = pd.get_dummies(df, columns=cat_col, drop_first=True)
df_dummies.columns
len(df_dummies.columns)  # 103



len(train_df) , len(test_df)

train_x = df_dummies.iloc[:15289,:].drop(columns=('yield'))
train_y = df_dummies.loc[:,'yield'][:15289]
test_x = df_dummies.iloc[15289:,:].drop(columns=('yield'))
len(train_x) , len(train_y) , len(test_x)
len(train_x.columns) , len(test_x.columns)




# 표준화
col = ['fruitset', 'fruitmass', 'seeds']
standard_scaler = StandardScaler()
standardized_data = standard_scaler.fit_transform(train_x[[i for i in col]])
train_x[[i + '_standard' for i in col]] = standardized_data

train_x.columns
len(train_x.columns) # 105
train_s_x = train_x.drop(columns=col)
len(train_s_x.columns)  # 102



standardized_data = standard_scaler.fit_transform(test_x[[i for i in col]])
test_x[[i + '_standard' for i in col]] = standardized_data

test_x.columns
len(test_x.columns) # 105
test_s_x = test_x.drop(columns=col)
len(test_s_x.columns)  # 102



# 다항 변수 늘리기
polynomial_transformer=PolynomialFeatures(3) 

polynomial_features = polynomial_transformer.fit_transform(train_s_x[[i+'_standard' for i in col]].values)
polynomial_features = polynomial_transformer.fit_transform([train_s_x[i+'_standard'].values for i in col])
features = polynomial_transformer.get_feature_names_out(train_s_x.columns)
train_bigs_x = pd.DataFrame(polynomial_features, columns=features)

polynomial_features = polynomial_transformer.fit_transform(test_s_x.values)
features = polynomial_transformer.get_feature_names_out(test_s_x.columns)
test_bigs_x = pd.DataFrame(polynomial_features, columns=features)






















from statsmodels.formula.api import ols
res = ols('yield~fruitset', data=train_df).fit()
fitted = res.predict(train_x['fruitset'])
residual = train_df['yield'] - fitted

plt.plot()

