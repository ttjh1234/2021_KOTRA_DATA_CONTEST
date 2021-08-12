# -*- coding: utf-8 -*-
"""
Created on Fri Jul 30 01:32:11 2021

@author: 82104
"""

import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns

from scipy.stats import uniform, randint

import sklearn
from sklearn.preprocessing import StandardScaler
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error, mean_absolute_error

import xgboost as xgb
from xgboost import XGBRegressor
from xgboost import plot_importance

from lightgbm import LGBMRegressor
from sklearn.linear_model import LinearRegression

pd.set_option('display.max_columns',30)

os.chdir("C:/Users/82104/Desktop/공모전1")

data2 = pd.read_csv("./빅데이터분석_최성수.csv")
data2.info()

data.info()

data.UNC_YEAR

data3=data[['HSCD','COUNTRYNM','HS','KR_TRADE_HSCD_COUNTRYCD','KR_2017']]
data3.columns=['HSCD','COUNTRYNM','HS','KR_2018','KR_2017']

data3.info()
data2.HSCD[0]
data3.HSCD[0]
data3.[0]
dash=pd.merge(data2,data3,how='left',left_on=['HSCD','COUNTRYNM'],right_on=['HSCD','COUNTRYNM'])

dash.info()
dash=dash.drop('Unnamed: 0',axis=1)
dash.to_csv('./dash.csv')

dash2017=dash[['UNC_YEAR','HSCD','HS','COUNTRYNM','KR_2017']]
dash2018=dash[['UNC_YEAR','HSCD','HS','COUNTRYNM','KR_2018']]
dash2019=dash[['UNC_YEAR','HSCD','HS','COUNTRYNM','KR_TRADE_HSCD_COUNTRYCD']]
dash2017.UNC_YEAR=2017
dash2019.UNC_YEAR=2019
dash2017.columns=['UNC_YEAR','HSCD','HS','COUNTRYNM','KR_TRADE_HSCD_COUNTRYCD']
dash2018.columns=['UNC_YEAR','HSCD','HS','COUNTRYNM','KR_TRADE_HSCD_COUNTRYCD']
dash2019.columns=['UNC_YEAR','HSCD','HS','COUNTRYNM','KR_TRADE_HSCD_COUNTRYCD']
dash2018
dash2019
dash_data=pd.concat([dash2017,dash2018,dash2019])
dash_data.to_excel('./dash_data.xlsx')
