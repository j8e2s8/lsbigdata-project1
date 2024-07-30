import pandas as pd
# 데이터 불러오기
train_df = pd.read_csv('train.csv')
test_df = pd.read_csv('test.csv')
df.head()

price_mean = df['SalePrice'].mean()

sample_submission = pd.read_csv('sample_submission.csv')
#sample_submission['SalePrice'] = price_mean
#sample_submission.head()

#sample_submission.to_csv('house price/sample_submission.csv' , index = False)



# groupby
group_train = train_df.groupby('YearBuilt',as_index=False).agg(price_mean=('SalePrice','mean'))
test = test_df[['Id','YearBuilt']]
test = pd.merge(test, group_train, how='left', on='YearBuilt')

null_id = test[test['price_mean'].isna()==True]['Id']
group_train['YearBuilt'].value_counts().sort_index()

for i in null_id:
    test.loc[test['Id']==i,'price_mean'] = int(test[test['YearBuilt']>int(test[test['Id']==i]['YearBuilt'])].sort_values('YearBuilt').iloc[0,:]['price_mean'])

# test['SalePrice'] = test['SalePrice'].fillna()  # 결측치 대체하기
 
test[test['price_mean'].isna()==True]  # 결측치 없다고 나옴


submission_merge = pd.merge(sample_submission, test, how='left', on='Id')[['Id','price_mean']]
submission_merge = submission_merge.rename(columns = {'price_mean' : 'SalePrice'})
submission_merge.info()


submission_merge.to_csv('sample_submission.csv', index=False)


# groupby2
import numpy as np
train_df['MSSubClass'].unique() != test_df['MSSubClass'].unique()


train_df.columns
train_df.info()
train_df['MSSubClass']=train_df['MSSubClass'].astype('object')

group_train = train_df.groupby('MSSubClass').agg(max_price = ('SalePrice', 'max')
                                 ,min_price = ('SalePrice', 'min')
                                 , mean_price = ('SalePrice', 'mean')).sort_values('max_price')
                                 
                                 
                                              
group_train = train_df.groupby(['YearBuilt','LandContour'],as_index=False).agg(price_mean=('SalePrice','mean'))
test = test_df[['Id','YearBuilt','LandContour']]
test = pd.merge(test, group_train, how='left', on=['YearBuilt','LandContour'])


test[test['price_mean'].isna()==True]
null_id = test[test['price_mean'].isna()==True]['Id']
group_train['YearBuilt'].value_counts().sort_index()

group_train2 = train_df.groupby(['LandContour'],as_index=False).agg(price_mean=('SalePrice','mean'))

for i in null_id:
    test.loc[test['Id']==i,'price_mean'] = int(test[test['YearBuilt']>int(test[test['Id']==i]['YearBuilt'])].sort_values('YearBuilt').iloc[0,:]['price_mean'])

test2 = pd.merge(test, group_train2, how='left', on='LandContour')
test2[test2['price_mean_y'].isna()==True]


submission_merge = pd.merge(sample_submission, test2, how='left', on='Id')[['Id','price_mean_y']]
submission_merge = submission_merge.rename(columns = {'price_mean_y' : 'SalePrice'})
submission_merge.info()


submission_merge.to_csv('sample_submission.csv', index=False)





# groupby3
                        
group_train = train_df.groupby(['YearBuilt','OverallQual'],as_index=False).agg(price_mean=('SalePrice','mean'))
test = test_df[['Id','YearBuilt','OverallQual']]
test = pd.merge(test, group_train, how='left', on=['YearBuilt','OverallQual'])


len(test[test['price_mean'].isna()==True])
test[test['price_mean'].isna()==True]['OverallQual'].value_counts()
test[test['price_mean'].isna()==True].groupby(['OverallQual', 'YearBuilt']).agg(value_count = ('YearBuilt', 'count'))

 #----------------------------------
test.groupby('OverallQual','Y).agg(year=('YearBuilt','min'))


null_id = test[test['price_mean'].isna()==True]['Id']
group_train['YearBuilt'].value_counts().sort_index()

group_train2 = train_df.groupby(['LotConfig'],as_index=False).agg(price_mean=('SalePrice','mean'))

for i in null_id:
    test.loc[test['Id']==i,'price_mean'] = int(test[test['YearBuilt']>int(test[test['Id']==i]['YearBuilt'])].sort_values('YearBuilt').iloc[0,:]['price_mean'])

test2 = pd.merge(test, group_train2, how='left', on='LandContour')
test2[test2['price_mean_y'].isna()==True]


submission_merge = pd.merge(sample_submission, test2, how='left', on='Id')[['Id','price_mean_y']]
submission_merge = submission_merge.rename(columns = {'price_mean_y' : 'SalePrice'})
submission_merge.info()


submission_merge.to_csv('sample_submission.csv', index=False)

