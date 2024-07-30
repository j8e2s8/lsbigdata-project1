# 9-7 교재
import pandas as pd
welfare = pd.read_spss('data/Koweps_hpwc14_2019_beta2.sav' )
age_div = welfare.query('marriage != "etc"').groupby('ageg', as_index = False)

mpg = pd.read_csv('data/mpg.csv')
mpg.head()
a = mpg.groupby('category', as_index=False)['manufacturer'].value_counts(normalize = True)

sum(a['proportion'][6:13])
