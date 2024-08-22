# house price
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

# 데이터 불러오기
train_df = pd.read_csv('./house price/train.csv')
test_df = pd.read_csv('./house price/test.csv')
submission = pd.read_csv('./house price/sample_submission.csv')

import os
os.getcwd()


