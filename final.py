# -*- coding: utf-8 -*-
"""
Created on Sat Jul 17 00:04:49 2021

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

data = pd.read_csv("./공모전데이터_분석용_KOTRA_0525.csv")

# 모든 결측치 처리 및 추가변수
def missing_data(df):
    #명목형 변수 형변환 
    df["HSCD"] = df["HSCD"].map(lambda x: str(x))
    df["COUNTRYCD"] = df["COUNTRYCD"].map(lambda x: str(x))
    
    # PA_NUS_FCRF 결측치 처리 (분석용 기준임)
    df.loc[df["PA_NUS_FCRF"].isnull() == True,"PA_NUS_FCRF"] = 1.13
    
    #SNDIST 결측치 처리
    missing_country = df.loc[df.SNDIST.isnull()==True,'COUNTRYNM'].unique()

    for i in missing_country:
        df.loc[(df.SNDIST.isnull()==True)&(df.COUNTRYNM==i),'SNDIST']=df.loc[df.COUNTRYNM==i,'SNDIST'].mean()
        
        
# TARIFF_AVG 처리
def TARIFF_AVG_missing(df) : 
    hs_2 = [] 
    for i in range(df.shape[0]):
        hs_2.append(df.iloc[i,1][:2])
    df["HS"] = hs_2
    
    ##품목분류 안된 999999코드는 관세 평균을 0으로 대치 
    df.loc[df.HSCD == "999999" ,"TARIFF_AVG"] = 0
    
    ##외부데이터 이용 
    ad_data1 = pd.read_csv("./WtoData_20210629174857.csv")
    ad_data1 = ad_data1[['Reporting Economy','Product/Sector Code','Year','Value']]
    ad_data1.columns = ['COUNTRYNM',"HS","Year","tariff"]
    ad_data1["HS"] = ad_data1["HS"].map(lambda x: str(x))
    
    ## 추가 데이터 국가명 다른 부분 해결 
    ad_data1.loc[ad_data1.COUNTRYNM=='Hong Kong, China','COUNTRYNM']='China, Hong Kong SAR'
    ad_data1.loc[ad_data1.COUNTRYNM=='Saudi Arabia, Kingdom of','COUNTRYNM']='Saudi Arabia'
    ad_data1.loc[ad_data1.COUNTRYNM=='United States of America','COUNTRYNM']='USA'
    ad_data1=ad_data1[ad_data1.Year==2017] #2018의 경우 이부분 수정할 것

    ad_data1.loc[ad_data1.tariff.isnull()==True,'tariff'] = 0
    
    data2 = df[df.TARIFF_AVG.isnull()==True]
    merge = pd.merge(data2,ad_data1, how="left", left_on=['HS','COUNTRYNM'], right_on=['HS','COUNTRYNM'])
    merge.loc[:,'TARIFF_AVG']=merge.tariff
    
    
    EU = ['Austria', 'Belgium', 'France', 'Germany', 'Italy', 'Netherlands','Spain']
    tariff_missing=merge[merge.TARIFF_AVG.isnull()==True]
    test=tariff_missing.COUNTRYNM.isin(EU)
    
    HS= [] 
    for i in test.index:
        if test[i]==True:
            HS.append(tariff_missing.HS[i])
            
    HS1=np.array(HS)
    HS2=pd.DataFrame(HS1)
    Euro_HS=HS2[0].unique()
    
    # 유로국가이면서 평균관세가 결측치인 품목들을 기존 데이터들의 자료를 인용하여 대치한다.
    for i in Euro_HS:
        masking= (merge.TARIFF_AVG.isnull()==True) & (merge.COUNTRYNM.isin(EU)) & (merge.HS==i)
        merge.loc[masking,'TARIFF_AVG'] = ad_data1.loc[(ad_data1.COUNTRYNM=='European Union') & (ad_data1.HS==i),'tariff'].values[0]
    
    merge[merge.TARIFF_AVG.isnull()==True]
    merge=merge[['HSCD','COUNTRYNM','TARIFF_AVG']]
    merge.columns=['HSCD','COUNTRYNM','Imputate']
    
    data2=pd.merge(df,merge,how='left',left_on=['HSCD','COUNTRYNM'],right_on=['HSCD','COUNTRYNM'])
    data2.loc[data2.TARIFF_AVG.isnull()==True,'TARIFF_AVG']=data2.loc[data2.TARIFF_AVG.isnull()==True,'Imputate']
    data2 = data2.drop('Imputate',axis=1)

    tt=data2.loc[data2.TARIFF_AVG.isnull()==True,['COUNTRYNM','HS']]
    maskout=(data2.TARIFF_AVG.isnull()==True)
    

    for i,j in zip(tt.COUNTRYNM,tt.HS):
        maskfor=(data2.COUNTRYNM==i)&(data2.HS==j)
        data2.loc[maskout&(maskfor),'TARIFF_AVG'] = data2.loc[maskfor,'TARIFF_AVG'].mean()
    
    return data2

## 2017년 자료
def read_2017_i(df):
    file_=os.listdir("./2017_import")
    file_path_1="./2017_import"

    i_res=pd.DataFrame()
    for i in range(1,len(file_)+1):
        filename = "comtrade (" + str(i) + ").csv"
        filepath = file_path_1 + '/' + filename
        tem = pd.read_csv(filepath)
        tem = tem[["Reporter","Commodity Code","Trade Value (US$)","Mode of Transport","Customs","2nd Partner"]]
        tem = tem[(tem["Mode of Transport"]=="All MOTs")]
        tem = tem[tem["Customs"]=="All CPCs"]
        tem = tem[tem["2nd Partner"] == "World"]
        tem.columns = ["COUNTRYNM","HSCD","i_KR_TRADE_HSCD_COUNTRYCD","type","type2","type3"]
    
        tem["HSCD"] = tem["HSCD"].map(lambda x: str(x))
    
        i_res = pd.concat([i_res,tem])

    i_res = i_res.reset_index().drop(["index","type","type2","type3"],axis=1)
    res=pd.merge(df,i_res,how='left', left_on=["COUNTRYNM","HSCD"], right_on=["COUNTRYNM","HSCD"])
    return res

def read_2017_e(df):
    file_=os.listdir("./2017_export")
    file_path_1="./2017_export"

    e_res=pd.DataFrame()
    for i in range(1,len(file_)+1):
        filename = "comtrade (" + str(i) + ").csv"
        filepath = file_path_1 + '/' + filename
        tem = pd.read_csv(filepath)
        tem = tem[["Partner","Commodity Code","Trade Value (US$)","Mode of Transport","Customs","2nd Partner"]]
        tem = tem[(tem["Mode of Transport"]=="All MOTs")]
        tem = tem[tem["Customs"]=="All CPCs"]
        tem = tem[tem["2nd Partner"] == "World"]
        tem.columns = ["COUNTRYNM","HSCD","e_KR_TRADE_HSCD_COUNTRYCD","type","type2","type3"]
    
        tem["HSCD"] = tem["HSCD"].map(lambda x: str(x))
    
        e_res = pd.concat([e_res,tem])

    e_res = e_res.reset_index().drop(["index","type","type2","type3"],axis=1)
    res=pd.merge(df,e_res,how='left', left_on=["COUNTRYNM","HSCD"], right_on=["COUNTRYNM","HSCD"])
    return res

data = pd.read_csv("./공모전데이터_분석용_KOTRA_0525.csv")

data.info()

missing_data(data)

data.info()

data =TARIFF_AVG_missing(data)
data = read_2017_i(data)
data = read_2017_e(data)

def imputate_KR(a,b,c):
    if (c not in except_country):
        if a!=-99:
            return a
        elif (a==-99)&(b!=-99):
            return b
        else:
            return 0
    else:
        if b!=-99:
            return b
        else : 
            return 0
        
except_country=['Guatemala','Viet Nam','Iran','Egypt']

data.loc[data["i_KR_TRADE_HSCD_COUNTRYCD"].isnull() == True,"i_KR_TRADE_HSCD_COUNTRYCD"] = -99
data.loc[data["e_KR_TRADE_HSCD_COUNTRYCD"].isnull() == True,"e_KR_TRADE_HSCD_COUNTRYCD"] = -99

data["KR_2017"] = data.apply(lambda x: imputate_KR(x["i_KR_TRADE_HSCD_COUNTRYCD"],x["e_KR_TRADE_HSCD_COUNTRYCD"],x["COUNTRYNM"]),axis=1)

data = data.drop(["i_KR_TRADE_HSCD_COUNTRYCD","e_KR_TRADE_HSCD_COUNTRYCD"],axis=1)

data.info()

def feature_engineering (df):
    # 작년도와 이번년도 GDP 비
    df["GDP_RATIO"]=df["NY_GDP_MKTP_CD"]/df["NY_GDP_MKTP_CD_1Y"]
    # 한국까지의 평균 거리와 수입국가간 평균 거리의 비
    df["RDIST"]=df['SNDIST']/df['KMDIST']
    
    # 해당 국가가 자료수집 간 예외 국가인지 식별 변수
    df["Except_country"]=0
    except_country=['Guatemala','Viet Nam','Iran','Egypt']
    df.loc[df['COUNTRYNM'].isin(except_country),'Except_country']=1
    
    # 품목 2자리 변수가 84 그룹과 85 그룹인지 확인하는 변수 
    df['HS_84_85']=0
    df.loc[((df['HS']==85)|(df['HS']==84)),'HS_84_85']=1
    
    df['HS_84_85']=df['HS_84_85'].astype('object')
    df['Except_country']=df['Except_country'].astype('object')
    
    # 해당 연도 해당 품목 전세계 수입 금액과 해당 연도 모든 품목 전세계 수입 금액의 비
    df['HSCD_RATIO']=df['TRADE_HSCD']/df['TRADE_HSCD'].unique().sum()
    # 해당 연도 해당 국가 전체 수입 금액과 해당 연도 모든 국가 전체 수입 금액의 비
    df['COUNTRY_RATIO']=df['TRADE_COUNTRYCD']/df['TRADE_COUNTRYCD'].unique().sum()
    # 해당 연도 해당 국가의 전체 수입 금액 중에 해당 품목의 전체 수입 금액의 비
    df['TRADE_RATIO']=df['TRADE_HSCD_COUNTRYCD']/df['TRADE_HSCD']
    # 품목별 해당연도 한국에서 수출한 금액과 해당 연도 품목별 전세계 수입금액의 비
    for i in df['HSCD'].unique():
        temp=df.loc[df['HSCD']==i,['KR_2017','TRADE_HSCD']]
        temp['temp']=temp['KR_2017'].sum()/temp['TRADE_HSCD'].unique()[0]
        df.loc[df['HSCD']==i,'KR_HSCD_RATIO']=temp['temp']
    
    # 나라별 해당연도 한국으로부터 수입한 금액과 해당연도 전세계에서 수입한 금액의 비 
    for i in df['COUNTRYCD'].unique():
        temp=df.loc[df['COUNTRYCD']==i,['KR_2017','TRADE_COUNTRYCD']]
        temp['temp']=temp['KR_2017'].sum()/temp['TRADE_COUNTRYCD'].unique()[0]
        df.loc[df['COUNTRYCD']==i,'KR_COUNTRY_RATIO']=temp['temp']
    
    # 해당 연도 해당 국가의 해당 품목 한국 수입 금액과 해당 연도 해당 국가의 해당 품목 수입금액의 비
    df['KR_HSCD_COUNTRY_RATIO']=df['KR_2017']/df['TRADE_HSCD_COUNTRYCD']
    
    # 나라별 해당 품목을 한국으로부터 수입한 금액과 한국으로부터 수입한 전체 금액의 비 
    for i in df['COUNTRYCD'].unique():
        temp=df.loc[df['COUNTRYCD']==i,['KR_2017']]
        temp['temp']=temp['KR_2017']/temp['KR_2017'].sum()
        df.loc[df['COUNTRYCD']==i,'KR_RATIO']=temp['temp']
    
    # 나라별 해당 연도 해당 국가의 해당 품목 수입 금액과 해당 연도 해당 품목의 전세계 총 수입금액의 비
    for i in df['COUNTRYCD'].unique():
        temp=df.loc[df['COUNTRYCD']==i,['TRADE_HSCD_COUNTRYCD','TRADE_COUNTRYCD']]
        temp['temp']=temp['TRADE_HSCD_COUNTRYCD']/temp['TRADE_COUNTRYCD'].unique()[0]
        df.loc[df['COUNTRYCD']==i,'TRADE_HSCD_RATIO']=temp['temp']
        
def add_FTA(df):
    FTA = pd.read_excel("./국가별FTA체결현황.xlsx")
    df=pd.merge(df,FTA,how='left',left_on='COUNTRYNM',right_on='country')
    df["FTA"] = df["FTA"].astype(object)
    df=df.drop(['country'],axis=1)
    return df

def add_group(df):
    ex = pd.read_excel("./2019년_HS_신성질별 성질별 연계(홈페이지 게재용) .xlsx",sheet_name="Sheet4")
    ex = ex[["6자리","현행수출1단위분류"]]
    ex = ex[ex["6자리"] > 99999]

    ex.columns = ["hscd","group"]
    for i in ex.index:
        if ex["group"][i] == "1. 식료 및 직접소비재":
            ex["group"][i] = "1"
        elif ex["group"][i] == "2. 원료 및 연료":
            ex["group"][i] = "2"
        elif ex["group"][i] == "3. 경공업품":
            ex["group"][i] = "3"
        elif ex["group"][i] == "4. 중화학 공업품":
            ex["group"][i] = "4"

    ex1 = ex.drop_duplicates(["hscd"])
    df['HSCD']=df['HSCD'].astype('int64')
    df = pd.merge(df,ex1,how="left",left_on='HSCD',right_on="hscd")
    df = df.drop("hscd",axis=1)
    df.loc[df.group.isnull()==True,'group']='5'
    df['HSCD']=df['HSCD'].astype(object)
    return df

def income_group(df):
    income=pd.read_csv("./Metadata_Country_API_PA.NUS.FCRF_DS2_en_csv_v2_2445345.csv")
    income=income[['IncomeGroup','TableName']]
    
    income.loc[income.TableName=='Czech Republic','TableName']='Czechia'
    income.loc[income.TableName=='Hong Kong SAR, China','TableName']='China, Hong Kong SAR'
    income.loc[income.TableName=='Iran, Islamic Rep.','TableName']='Iran'
    income.loc[income.TableName=='Vietnam','TableName']='Viet Nam'
    income.loc[income.TableName=='Egypt, Arab Rep.','TableName']='Egypt'
    income.loc[income.TableName=='United States','TableName']='USA'

    df=pd.merge(df,income,how='left',left_on='COUNTRYNM',right_on='TableName')
    df=df.drop("TableName",axis=1)
    return df

def variable_to_group(df):
    df['dist']=pd.qcut(df['SNDIST'],3,labels=[1,3,5])

    df['TARIFF']=0
    df.loc[df['TARIFF_AVG']>0,'TARIFF']=1

    df['PA']=pd.qcut(df['PA_NUS_FCRF'],2,labels=[0,1])

    df['BE']=pd.qcut(df['IC_BUS_EASE_DFRN_DB'],3,labels=[1,3,5])
    df['TARIFF']=df['TARIFF'].astype('category')


feature_engineering(data)
data=add_FTA(data)
data=add_group(data)
data=income_group(data)
variable_to_group(data)

data.info()

data.describe()

data

data=data.reset_index().drop('index',axis=1)

new=data[['TRADE_COUNTRYCD', 'TRADE_HSCD', 'TARIFF_AVG', 'SNDIST','NY_GDP_MKTP_CD', 'SP_POP_TOTL', 'PA_NUS_FCRF', 'IC_BUS_EASE_DFRN_DB','KMDIST', 'KR_2017', 'GDP_RATIO', 'RDIST', 'HSCD_RATIO','COUNTRY_RATIO','TRADE_HSCD_COUNTRYCD','KR_TRADE_HSCD_COUNTRYCD']]

mask = ((new['KR_TRADE_HSCD_COUNTRYCD']==0)&(new['KR_2017']!=0))|((new['KR_TRADE_HSCD_COUNTRYCD']!=0)&(new['KR_2017']==0)) 

# 41 
# KMDIST GDP_RATIO TRADE_HSCD COUNTRY_RATIO
model_c=new[~mask]
x_pred_41=model_c[model_c['TRADE_HSCD_COUNTRYCD'].isnull()==True]
index=x_pred_41.index
model_c=model_c[model_c['TRADE_HSCD_COUNTRYCD'].isnull()==False]
model_c=model_c.reset_index().drop('index',axis=1)
y=model_c["TRADE_HSCD_COUNTRYCD"]
x=model_c.drop("TRADE_HSCD_COUNTRYCD",axis=1)
x2=x[['TRADE_COUNTRYCD', 'TARIFF_AVG', 'SNDIST','NY_GDP_MKTP_CD', 'SP_POP_TOTL', 'PA_NUS_FCRF', 'IC_BUS_EASE_DFRN_DB', 'KR_2017', 'RDIST', 'HSCD_RATIO']]
x_pred_41=x_pred_41[['TRADE_COUNTRYCD', 'TARIFF_AVG', 'SNDIST','NY_GDP_MKTP_CD', 'SP_POP_TOTL', 'PA_NUS_FCRF', 'IC_BUS_EASE_DFRN_DB', 'KR_2017', 'RDIST', 'HSCD_RATIO']]
name=x2.columns
x2=np.log1p(x2)
x_pred_41=np.log1p(x_pred_41)
y=np.log1p(y)
ss=StandardScaler()
ss.fit(x2)
x_num=ss.transform(x2)
x_pred_41=ss.transform(x_pred_41)
x=pd.DataFrame(x_num)
x.columns=list(name)
x_pred_41=pd.DataFrame(x_pred_41,index=index)
x_pred_41.columns=list(name)


x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=41)


xgbr41=XGBRegressor(random_state=42)
xgbr41.fit(x_train,y_train)

model_compare=new[['TRADE_COUNTRYCD','TARIFF_AVG', 'SNDIST','NY_GDP_MKTP_CD', 'SP_POP_TOTL', 'PA_NUS_FCRF', 'IC_BUS_EASE_DFRN_DB','KR_2017','RDIST', 'HSCD_RATIO','TRADE_HSCD_COUNTRYCD']]
model_compare=model_compare[model_compare['TRADE_HSCD_COUNTRYCD'].isnull()==False]
model_compare=model_compare.reset_index().drop('index',axis=1)
y_c=model_compare["TRADE_HSCD_COUNTRYCD"]
x_c=model_compare.drop("TRADE_HSCD_COUNTRYCD",axis=1)
x2_c=x_c[['TRADE_COUNTRYCD','TARIFF_AVG', 'SNDIST','NY_GDP_MKTP_CD', 'SP_POP_TOTL', 'PA_NUS_FCRF', 'IC_BUS_EASE_DFRN_DB','KR_2017','RDIST', 'HSCD_RATIO']]
name=x2.columns
x2_c=np.log1p(x2_c)
y_c=np.log1p(y_c)
x_num_c=ss.transform(x2_c)
x_c=pd.DataFrame(x_num_c)
x_c.columns=list(name)
    
pred_41=xgbr41.predict(x_c)
rmse_41=np.sqrt((((np.expm1(y_c)-np.expm1(pred_41))**2).sum())/len(pred_41))

# 42 
# NY_GDP_MKTP_CD SP_POP_TOTL HSCD_RATIO COUNTRY_RATIO
model_c=new[~mask]
x_pred_42=model_c[model_c['TRADE_HSCD_COUNTRYCD'].isnull()==True]
index=x_pred_42.index
model_c=model_c[model_c['TRADE_HSCD_COUNTRYCD'].isnull()==False]
model_c=model_c.reset_index().drop('index',axis=1)
y=model_c["TRADE_HSCD_COUNTRYCD"]
x=model_c.drop("TRADE_HSCD_COUNTRYCD",axis=1)
x2=x[['TRADE_COUNTRYCD', 'TRADE_HSCD', 'TARIFF_AVG', 'SNDIST', 'PA_NUS_FCRF', 'IC_BUS_EASE_DFRN_DB','KMDIST', 'KR_2017', 'GDP_RATIO', 'RDIST']]
x_pred_42=x_pred_42[['TRADE_COUNTRYCD', 'TRADE_HSCD', 'TARIFF_AVG', 'SNDIST', 'PA_NUS_FCRF', 'IC_BUS_EASE_DFRN_DB','KMDIST', 'KR_2017', 'GDP_RATIO', 'RDIST']]
name=x2.columns
x2=np.log1p(x2)
x_pred_42=np.log1p(x_pred_42)
y=np.log1p(y)
ss=StandardScaler()
ss.fit(x2)
x_num=ss.transform(x2)
x_pred_42=ss.transform(x_pred_42)
x=pd.DataFrame(x_num)
x.columns=list(name)
x_pred_42=pd.DataFrame(x_pred_42,index=index)
x_pred_42.columns=list(name)
    
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=42)


xgbr42=XGBRegressor(random_state=42)
xgbr42.fit(x_train,y_train)

model_compare=new[['TRADE_COUNTRYCD', 'TRADE_HSCD', 'TARIFF_AVG', 'SNDIST','PA_NUS_FCRF', 'IC_BUS_EASE_DFRN_DB','KMDIST', 'KR_2017', 'GDP_RATIO', 'RDIST','TRADE_HSCD_COUNTRYCD']]
model_compare=model_compare[model_compare['TRADE_HSCD_COUNTRYCD'].isnull()==False]
model_compare=model_compare.reset_index().drop('index',axis=1)
y_c=model_compare["TRADE_HSCD_COUNTRYCD"]
x_c=model_compare.drop("TRADE_HSCD_COUNTRYCD",axis=1)
x2_c=x_c[['TRADE_COUNTRYCD', 'TRADE_HSCD', 'TARIFF_AVG', 'SNDIST','PA_NUS_FCRF', 'IC_BUS_EASE_DFRN_DB','KMDIST', 'KR_2017', 'GDP_RATIO', 'RDIST']]
name=x2.columns
x2_c=np.log1p(x2_c)
y_c=np.log1p(y_c)
x_num_c=ss.transform(x2_c)
x_c=pd.DataFrame(x_num_c)
x_c.columns=list(name)
    
pred_42=xgbr42.predict(x_c)
rmse_42=np.sqrt((((np.expm1(y_c)-np.expm1(pred_42))**2).sum())/len(pred_42))

# 43
# PA_NUS_FCRF TRADE_HSCD TRADE_COUNTRYCD
model_c=new[~mask]
x_pred_43=model_c[model_c['TRADE_HSCD_COUNTRYCD'].isnull()==True]
index=x_pred_43.index
model_c=model_c[model_c['TRADE_HSCD_COUNTRYCD'].isnull()==False]
model_c=model_c.reset_index().drop('index',axis=1)
y=model_c["TRADE_HSCD_COUNTRYCD"]
x=model_c.drop("TRADE_HSCD_COUNTRYCD",axis=1)
x2=x[['TARIFF_AVG', 'SNDIST','NY_GDP_MKTP_CD', 'SP_POP_TOTL', 'IC_BUS_EASE_DFRN_DB','KMDIST', 'KR_2017', 'GDP_RATIO', 'RDIST', 'HSCD_RATIO','COUNTRY_RATIO']]
x_pred_43=x_pred_43[['TARIFF_AVG', 'SNDIST','NY_GDP_MKTP_CD', 'SP_POP_TOTL', 'IC_BUS_EASE_DFRN_DB','KMDIST', 'KR_2017', 'GDP_RATIO', 'RDIST', 'HSCD_RATIO','COUNTRY_RATIO']]
name=x2.columns
x2=np.log1p(x2)
x_pred_43=np.log1p(x_pred_43)
y=np.log1p(y)
ss=StandardScaler()
ss.fit(x2)
x_num=ss.transform(x2)
x_pred_43=ss.transform(x_pred_43)
x=pd.DataFrame(x_num)
x.columns=list(name)
x_pred_43=pd.DataFrame(x_pred_43,index=index)
x_pred_43.columns=list(name)
    
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=43)


xgbr43=XGBRegressor(random_state=42)
xgbr43.fit(x_train,y_train)

model_compare=new[['TARIFF_AVG', 'SNDIST','NY_GDP_MKTP_CD', 'SP_POP_TOTL','IC_BUS_EASE_DFRN_DB','KMDIST', 'KR_2017', 'GDP_RATIO', 'RDIST', 'HSCD_RATIO','COUNTRY_RATIO','TRADE_HSCD_COUNTRYCD']]
model_compare=model_compare[model_compare['TRADE_HSCD_COUNTRYCD'].isnull()==False]
model_compare=model_compare.reset_index().drop('index',axis=1)
y_c=model_compare["TRADE_HSCD_COUNTRYCD"]
x_c=model_compare.drop("TRADE_HSCD_COUNTRYCD",axis=1)
x2_c=x_c[['TARIFF_AVG', 'SNDIST','NY_GDP_MKTP_CD', 'SP_POP_TOTL','IC_BUS_EASE_DFRN_DB','KMDIST', 'KR_2017', 'GDP_RATIO', 'RDIST', 'HSCD_RATIO','COUNTRY_RATIO']]
name=x2.columns
x2_c=np.log1p(x2_c)
y_c=np.log1p(y_c)
x_num_c=ss.transform(x2_c)
x_c=pd.DataFrame(x_num_c)
x_c.columns=list(name)
    
pred_43=xgbr43.predict(x_c)
rmse_43=np.sqrt((((np.expm1(y_c)-np.expm1(pred_43))**2).sum())/len(pred_43))

# 44
# RDIST SP_POP_TOTL PA_NUS_FCRF TRADE_COUNTRYCD HSCD_RATIO
model_c=new[~mask]
x_pred_44=model_c[model_c['TRADE_HSCD_COUNTRYCD'].isnull()==True]
index=x_pred_44.index
model_c=model_c[model_c['TRADE_HSCD_COUNTRYCD'].isnull()==False]
model_c=model_c.reset_index().drop('index',axis=1)
y=model_c["TRADE_HSCD_COUNTRYCD"]
x=model_c.drop("TRADE_HSCD_COUNTRYCD",axis=1)
x2=x[['TRADE_HSCD', 'TARIFF_AVG', 'SNDIST','NY_GDP_MKTP_CD','IC_BUS_EASE_DFRN_DB','KMDIST', 'KR_2017', 'GDP_RATIO','COUNTRY_RATIO']]
x_pred_44=x_pred_44[['TRADE_HSCD', 'TARIFF_AVG', 'SNDIST','NY_GDP_MKTP_CD','IC_BUS_EASE_DFRN_DB','KMDIST', 'KR_2017', 'GDP_RATIO','COUNTRY_RATIO']]
name=x2.columns
x2=np.log1p(x2)
x_pred_44=np.log1p(x_pred_44)
y=np.log1p(y)
ss=StandardScaler()
ss.fit(x2)
x_num=ss.transform(x2)
x_pred_44=ss.transform(x_pred_44)
x=pd.DataFrame(x_num)
x.columns=list(name)
x_pred_44=pd.DataFrame(x_pred_44,index=index)
x_pred_44.columns=list(name)

x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=44)


xgbr44=XGBRegressor(random_state=42)
xgbr44.fit(x_train,y_train)

model_compare=new[['TRADE_HSCD', 'TARIFF_AVG', 'SNDIST','NY_GDP_MKTP_CD','IC_BUS_EASE_DFRN_DB','KMDIST', 'KR_2017', 'GDP_RATIO','COUNTRY_RATIO','TRADE_HSCD_COUNTRYCD']]
model_compare=model_compare[model_compare['TRADE_HSCD_COUNTRYCD'].isnull()==False]
model_compare=model_compare.reset_index().drop('index',axis=1)
y_c=model_compare["TRADE_HSCD_COUNTRYCD"]
x_c=model_compare.drop("TRADE_HSCD_COUNTRYCD",axis=1)
x2_c=x_c[['TRADE_HSCD', 'TARIFF_AVG', 'SNDIST','NY_GDP_MKTP_CD','IC_BUS_EASE_DFRN_DB','KMDIST', 'KR_2017', 'GDP_RATIO','COUNTRY_RATIO']]
name=x2.columns
x2_c=np.log1p(x2_c)
y_c=np.log1p(y_c)
x_num_c=ss.transform(x2_c)
x_c=pd.DataFrame(x_num_c)
x_c.columns=list(name)
    
pred_44=xgbr44.predict(x_c)
rmse_44=np.sqrt((((np.expm1(y_c)-np.expm1(pred_44))**2).sum())//len(pred_44))

# 45
# TARIFF_AVG IC_BUS TRADE_COUNTRYCD HSCD_RATIO
model_c=new[~mask]
x_pred_45=model_c[model_c['TRADE_HSCD_COUNTRYCD'].isnull()==True]
index=x_pred_45.index
model_c=model_c[model_c['TRADE_HSCD_COUNTRYCD'].isnull()==False]
model_c=model_c.reset_index().drop('index',axis=1)
y=model_c["TRADE_HSCD_COUNTRYCD"]
x=model_c.drop("TRADE_HSCD_COUNTRYCD",axis=1)
x2=x[['TRADE_HSCD','SNDIST','NY_GDP_MKTP_CD', 'SP_POP_TOTL', 'PA_NUS_FCRF','KMDIST', 'KR_2017', 'GDP_RATIO', 'RDIST','COUNTRY_RATIO']]
x_pred_45=x_pred_45[['TRADE_HSCD','SNDIST','NY_GDP_MKTP_CD', 'SP_POP_TOTL', 'PA_NUS_FCRF','KMDIST', 'KR_2017', 'GDP_RATIO', 'RDIST','COUNTRY_RATIO']]
name=x2.columns
x2=np.log1p(x2)
x_pred_45=np.log1p(x_pred_45)
y=np.log1p(y)
ss=StandardScaler()
ss.fit(x2)
x_num=ss.transform(x2)
x_pred_45=ss.transform(x_pred_45)
x=pd.DataFrame(x_num)
x.columns=list(name)
x_pred_45=pd.DataFrame(x_pred_45,index=index)
x_pred_45.columns=list(name)
    
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=45)


xgbr45=XGBRegressor(random_state=42)
xgbr45.fit(x_train,y_train)

model_compare=new[['TRADE_HSCD','SNDIST','NY_GDP_MKTP_CD', 'SP_POP_TOTL', 'PA_NUS_FCRF','KMDIST', 'KR_2017', 'GDP_RATIO', 'RDIST','COUNTRY_RATIO','TRADE_HSCD_COUNTRYCD']]
model_compare=model_compare[model_compare['TRADE_HSCD_COUNTRYCD'].isnull()==False]
model_compare=model_compare.reset_index().drop('index',axis=1)
y_c=model_compare["TRADE_HSCD_COUNTRYCD"]
x_c=model_compare.drop("TRADE_HSCD_COUNTRYCD",axis=1)
x2_c=x_c[['TRADE_HSCD','SNDIST','NY_GDP_MKTP_CD', 'SP_POP_TOTL', 'PA_NUS_FCRF','KMDIST', 'KR_2017', 'GDP_RATIO', 'RDIST','COUNTRY_RATIO']]
name=x2.columns
x2_c=np.log1p(x2_c)
y_c=np.log1p(y_c)
x_num_c=ss.transform(x2_c)
x_c=pd.DataFrame(x_num_c)
x_c.columns=list(name)
    
pred_45=xgbr45.predict(x_c)
rmse_45=np.sqrt((((np.expm1(y_c)-np.expm1(pred_45))**2).sum())//len(pred_45))

# 46
# IC_BUS SNDIST TRADE_COUNTRYCD TRADE_HSCD
model_c=new[~mask]
x_pred_46=model_c[model_c['TRADE_HSCD_COUNTRYCD'].isnull()==True]
index=x_pred_46.index
model_c=model_c[model_c['TRADE_HSCD_COUNTRYCD'].isnull()==False]
model_c=model_c.reset_index().drop('index',axis=1)
y=model_c["TRADE_HSCD_COUNTRYCD"]
x=model_c.drop("TRADE_HSCD_COUNTRYCD",axis=1)
x2=x[['TARIFF_AVG','NY_GDP_MKTP_CD', 'SP_POP_TOTL', 'PA_NUS_FCRF','KMDIST', 'KR_2017', 'GDP_RATIO', 'RDIST', 'HSCD_RATIO','COUNTRY_RATIO']]
x_pred_46=x_pred_46[['TARIFF_AVG','NY_GDP_MKTP_CD', 'SP_POP_TOTL', 'PA_NUS_FCRF','KMDIST', 'KR_2017', 'GDP_RATIO', 'RDIST', 'HSCD_RATIO','COUNTRY_RATIO']]
name=x2.columns
x2=np.log1p(x2)
x_pred_46=np.log1p(x_pred_46)
y=np.log1p(y)
ss=StandardScaler()
ss.fit(x2)
x_num=ss.transform(x2)
x_pred_46=ss.transform(x_pred_46)
x=pd.DataFrame(x_num)
x.columns=list(name)
x_pred_46=pd.DataFrame(x_pred_46,index=index)
x_pred_46.columns=list(name)
    
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=46)


xgbr46=XGBRegressor(random_state=42)
xgbr46.fit(x_train,y_train)

model_compare=new[['TARIFF_AVG','NY_GDP_MKTP_CD', 'SP_POP_TOTL', 'PA_NUS_FCRF','KMDIST', 'KR_2017', 'GDP_RATIO', 'RDIST', 'HSCD_RATIO','COUNTRY_RATIO','TRADE_HSCD_COUNTRYCD']]
model_compare=model_compare[model_compare['TRADE_HSCD_COUNTRYCD'].isnull()==False]
model_compare=model_compare.reset_index().drop('index',axis=1)
y_c=model_compare["TRADE_HSCD_COUNTRYCD"]
x_c=model_compare.drop("TRADE_HSCD_COUNTRYCD",axis=1)
x2_c=x_c[['TARIFF_AVG','NY_GDP_MKTP_CD', 'SP_POP_TOTL', 'PA_NUS_FCRF','KMDIST', 'KR_2017', 'GDP_RATIO', 'RDIST', 'HSCD_RATIO','COUNTRY_RATIO']]
name=x2.columns
x2_c=np.log1p(x2_c)
y_c=np.log1p(y_c)
x_num_c=ss.transform(x2_c)
x_c=pd.DataFrame(x_num_c)
x_c.columns=list(name)
    
pred_46=xgbr46.predict(x_c)
rmse_46=np.sqrt((((np.expm1(y_c)-np.expm1(pred_46))**2).sum())/len(pred_46))

# 47
# SP_POP TRADE_COUNTRYCD HSCD_RATIO
model_c=new[~mask]
x_pred_47=model_c[model_c['TRADE_HSCD_COUNTRYCD'].isnull()==True]
index=x_pred_47.index
model_c=model_c[model_c['TRADE_HSCD_COUNTRYCD'].isnull()==False]
model_c=model_c.reset_index().drop('index',axis=1)
y=model_c["TRADE_HSCD_COUNTRYCD"]
x=model_c.drop("TRADE_HSCD_COUNTRYCD",axis=1)
x2=x[['TRADE_HSCD', 'TARIFF_AVG', 'SNDIST','NY_GDP_MKTP_CD','PA_NUS_FCRF', 'IC_BUS_EASE_DFRN_DB','KMDIST', 'KR_2017', 'GDP_RATIO', 'RDIST','COUNTRY_RATIO']]
x_pred_47=x_pred_47[['TRADE_HSCD', 'TARIFF_AVG', 'SNDIST','NY_GDP_MKTP_CD','PA_NUS_FCRF', 'IC_BUS_EASE_DFRN_DB','KMDIST', 'KR_2017', 'GDP_RATIO', 'RDIST','COUNTRY_RATIO']]
name=x2.columns
x2=np.log1p(x2)
x_pred_47=np.log1p(x_pred_47)
y=np.log1p(y)
ss=StandardScaler()
ss.fit(x2)
x_num=ss.transform(x2)
x_pred_47=ss.transform(x_pred_47)
x=pd.DataFrame(x_num)
x.columns=list(name)
x_pred_47=pd.DataFrame(x_pred_47,index=index)
x_pred_47.columns=list(name)
    
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=47)


xgbr47=XGBRegressor(random_state=42)
xgbr47.fit(x_train,y_train)

model_compare=new[['TRADE_HSCD', 'TARIFF_AVG', 'SNDIST','NY_GDP_MKTP_CD','PA_NUS_FCRF', 'IC_BUS_EASE_DFRN_DB','KMDIST', 'KR_2017', 'GDP_RATIO', 'RDIST','COUNTRY_RATIO','TRADE_HSCD_COUNTRYCD']]
model_compare=model_compare[model_compare['TRADE_HSCD_COUNTRYCD'].isnull()==False]
model_compare=model_compare.reset_index().drop('index',axis=1)
y_c=model_compare["TRADE_HSCD_COUNTRYCD"]
x_c=model_compare.drop("TRADE_HSCD_COUNTRYCD",axis=1)
x2_c=x_c[['TRADE_HSCD', 'TARIFF_AVG', 'SNDIST','NY_GDP_MKTP_CD','PA_NUS_FCRF', 'IC_BUS_EASE_DFRN_DB','KMDIST', 'KR_2017', 'GDP_RATIO', 'RDIST','COUNTRY_RATIO']]
name=x2.columns
x2_c=np.log1p(x2_c)
y_c=np.log1p(y_c)
x_num_c=ss.transform(x2_c)
x_c=pd.DataFrame(x_num_c)
x_c.columns=list(name)
    
pred_47=xgbr47.predict(x_c)
rmse_47=np.sqrt((((np.expm1(y_c)-np.expm1(pred_47))**2).sum())/len(pred_47))

# 48
# TRADE_HSCD COUNTRY_RATIO
model_c=new[~mask]
x_pred_48=model_c[model_c['TRADE_HSCD_COUNTRYCD'].isnull()==True]
index=x_pred_48.index
model_c=model_c[model_c['TRADE_HSCD_COUNTRYCD'].isnull()==False]
model_c=model_c.reset_index().drop('index',axis=1)
y=model_c["TRADE_HSCD_COUNTRYCD"]
x=model_c.drop("TRADE_HSCD_COUNTRYCD",axis=1)
x2=x[['TRADE_COUNTRYCD','TARIFF_AVG', 'SNDIST','NY_GDP_MKTP_CD', 'SP_POP_TOTL', 'PA_NUS_FCRF', 'IC_BUS_EASE_DFRN_DB','KMDIST', 'KR_2017', 'GDP_RATIO', 'RDIST', 'HSCD_RATIO']]
x_pred_48=x_pred_48[['TRADE_COUNTRYCD','TARIFF_AVG', 'SNDIST','NY_GDP_MKTP_CD', 'SP_POP_TOTL', 'PA_NUS_FCRF', 'IC_BUS_EASE_DFRN_DB','KMDIST', 'KR_2017', 'GDP_RATIO', 'RDIST', 'HSCD_RATIO']]
name=x2.columns
x2=np.log1p(x2)
x_pred_48=np.log1p(x_pred_48)
y=np.log1p(y)
ss=StandardScaler()
ss.fit(x2)
x_num=ss.transform(x2)
x_pred_48=ss.transform(x_pred_48)
x=pd.DataFrame(x_num)
x.columns=list(name)
x_pred_48=pd.DataFrame(x_pred_48,index=index)
x_pred_48.columns=list(name)
   
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=48)


xgbr48=XGBRegressor(random_state=42)
xgbr48.fit(x_train,y_train)

model_compare=new[['TRADE_COUNTRYCD','TARIFF_AVG', 'SNDIST','NY_GDP_MKTP_CD', 'SP_POP_TOTL', 'PA_NUS_FCRF', 'IC_BUS_EASE_DFRN_DB','KMDIST', 'KR_2017', 'GDP_RATIO', 'RDIST', 'HSCD_RATIO','TRADE_HSCD_COUNTRYCD']]
model_compare=model_compare[model_compare['TRADE_HSCD_COUNTRYCD'].isnull()==False]
model_compare=model_compare.reset_index().drop('index',axis=1)
y_c=model_compare["TRADE_HSCD_COUNTRYCD"]
x_c=model_compare.drop("TRADE_HSCD_COUNTRYCD",axis=1)
x2_c=x_c[['TRADE_COUNTRYCD','TARIFF_AVG', 'SNDIST','NY_GDP_MKTP_CD', 'SP_POP_TOTL', 'PA_NUS_FCRF', 'IC_BUS_EASE_DFRN_DB','KMDIST', 'KR_2017', 'GDP_RATIO', 'RDIST', 'HSCD_RATIO']]
name=x2.columns
x2_c=np.log1p(x2_c)
y_c=np.log1p(y_c)
x_num_c=ss.transform(x2_c)
x_c=pd.DataFrame(x_num_c)
x_c.columns=list(name)
    
pred_48=xgbr48.predict(x_c)
rmse_48=np.sqrt((((np.expm1(y_c)-np.expm1(pred_48))**2).sum())/len(pred_48))

# 49
# RDIST HSCD_RATIO COUNTRY_RATIO
model_c=new[~mask]
x_pred_49=model_c[model_c['TRADE_HSCD_COUNTRYCD'].isnull()==True]
index=x_pred_49.index
model_c=model_c[model_c['TRADE_HSCD_COUNTRYCD'].isnull()==False]
model_c=model_c.reset_index().drop('index',axis=1)
y=model_c["TRADE_HSCD_COUNTRYCD"]
x=model_c.drop("TRADE_HSCD_COUNTRYCD",axis=1)
x2=x[['TRADE_COUNTRYCD', 'TRADE_HSCD', 'TARIFF_AVG', 'SNDIST','NY_GDP_MKTP_CD', 'SP_POP_TOTL', 'PA_NUS_FCRF', 'IC_BUS_EASE_DFRN_DB','KMDIST', 'KR_2017', 'GDP_RATIO']]
x_pred_49=x_pred_49[['TRADE_COUNTRYCD', 'TRADE_HSCD', 'TARIFF_AVG', 'SNDIST','NY_GDP_MKTP_CD', 'SP_POP_TOTL', 'PA_NUS_FCRF', 'IC_BUS_EASE_DFRN_DB','KMDIST', 'KR_2017', 'GDP_RATIO']]
name=x2.columns
x2=np.log1p(x2)
x_pred_49=np.log1p(x_pred_49)
y=np.log1p(y)
ss=StandardScaler()
ss.fit(x2)
x_num=ss.transform(x2)
x_pred_49=ss.transform(x_pred_49)
x=pd.DataFrame(x_num)
x.columns=list(name)
x_pred_49=pd.DataFrame(x_pred_49,index=index)
x_pred_49.columns=list(name)
 
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=49)


xgbr49=XGBRegressor(random_state=42)
xgbr49.fit(x_train,y_train)

model_compare=new[['TRADE_COUNTRYCD', 'TRADE_HSCD', 'TARIFF_AVG', 'SNDIST','NY_GDP_MKTP_CD', 'SP_POP_TOTL', 'PA_NUS_FCRF', 'IC_BUS_EASE_DFRN_DB','KMDIST', 'KR_2017', 'GDP_RATIO','TRADE_HSCD_COUNTRYCD']]
model_compare=model_compare[model_compare['TRADE_HSCD_COUNTRYCD'].isnull()==False]
model_compare=model_compare.reset_index().drop('index',axis=1)
y_c=model_compare["TRADE_HSCD_COUNTRYCD"]
x_c=model_compare.drop("TRADE_HSCD_COUNTRYCD",axis=1)
x2_c=x_c[['TRADE_COUNTRYCD', 'TRADE_HSCD', 'TARIFF_AVG', 'SNDIST','NY_GDP_MKTP_CD', 'SP_POP_TOTL', 'PA_NUS_FCRF', 'IC_BUS_EASE_DFRN_DB','KMDIST', 'KR_2017', 'GDP_RATIO']]
name=x2.columns
x2_c=np.log1p(x2_c)
y_c=np.log1p(y_c)
x_num_c=ss.transform(x2_c)
x_c=pd.DataFrame(x_num_c)
x_c.columns=list(name)
    
pred_49=xgbr49.predict(x_c)
rmse_49=np.sqrt((((np.expm1(y_c)-np.expm1(pred_49))**2).sum())/len(pred_49))

# 50
# TARIFF_AVG SP_POP TRADE_COUNTRYCD HSCD_RATIO
model_c=new[~mask]
x_pred_50=model_c[model_c['TRADE_HSCD_COUNTRYCD'].isnull()==True]
index=x_pred_50.index
model_c=model_c[model_c['TRADE_HSCD_COUNTRYCD'].isnull()==False]
model_c=model_c.reset_index().drop('index',axis=1)
y=model_c["TRADE_HSCD_COUNTRYCD"]
x=model_c.drop("TRADE_HSCD_COUNTRYCD",axis=1)
x2=x[['TRADE_HSCD','SNDIST','NY_GDP_MKTP_CD','PA_NUS_FCRF', 'IC_BUS_EASE_DFRN_DB','KMDIST', 'KR_2017', 'GDP_RATIO', 'RDIST','COUNTRY_RATIO']]
x_pred_50=x_pred_50[['TRADE_HSCD','SNDIST','NY_GDP_MKTP_CD','PA_NUS_FCRF', 'IC_BUS_EASE_DFRN_DB','KMDIST', 'KR_2017', 'GDP_RATIO', 'RDIST','COUNTRY_RATIO']]
name=x2.columns
x2=np.log1p(x2)
x_pred_50=np.log1p(x_pred_50)
y=np.log1p(y)
ss=StandardScaler()
ss.fit(x2)
x_num=ss.transform(x2)
x_pred_50=ss.transform(x_pred_50)
x=pd.DataFrame(x_num)
x.columns=list(name)
x_pred_50=pd.DataFrame(x_pred_50,index=index)
x_pred_50.columns=list(name)

    
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=50)


xgbr50=XGBRegressor(random_state=42)
xgbr50.fit(x_train,y_train)

model_compare=new[['TRADE_HSCD','SNDIST','NY_GDP_MKTP_CD','PA_NUS_FCRF', 'IC_BUS_EASE_DFRN_DB','KMDIST', 'KR_2017', 'GDP_RATIO', 'RDIST','COUNTRY_RATIO','TRADE_HSCD_COUNTRYCD']]
model_compare=model_compare[model_compare['TRADE_HSCD_COUNTRYCD'].isnull()==False]
model_compare=model_compare.reset_index().drop('index',axis=1)
y_c=model_compare["TRADE_HSCD_COUNTRYCD"]
x_c=model_compare.drop("TRADE_HSCD_COUNTRYCD",axis=1)
x2_c=x_c[['TRADE_HSCD', 'SNDIST','NY_GDP_MKTP_CD','PA_NUS_FCRF', 'IC_BUS_EASE_DFRN_DB','KMDIST', 'KR_2017', 'GDP_RATIO', 'RDIST','COUNTRY_RATIO']]
name=x2.columns
x2_c=np.log1p(x2_c)
y_c=np.log1p(y_c)
x_num_c=ss.transform(x2_c)
x_c=pd.DataFrame(x_num_c)
x_c.columns=list(name)
    
pred_50=xgbr50.predict(x_c)
rmse_50=np.sqrt((((np.expm1(y_c)-np.expm1(pred_50))**2).sum())/len(pred_50))


# 역수 변환
reverse_rmse_sum=1/rmse_41+1/rmse_42+1/rmse_43+1/rmse_44+1/rmse_45+1/rmse_46+1/rmse_47+1/rmse_48+1/rmse_49+1/rmse_50
# 가중치
weight=[(1/rmse_41)/reverse_rmse_sum,(1/rmse_42)/reverse_rmse_sum,
        (1/rmse_43)/reverse_rmse_sum,(1/rmse_44)/reverse_rmse_sum,
        (1/rmse_45)/reverse_rmse_sum,(1/rmse_46)/reverse_rmse_sum,
        (1/rmse_47)/reverse_rmse_sum,(1/rmse_48)/reverse_rmse_sum,
        (1/rmse_49)/reverse_rmse_sum,(1/rmse_50)/reverse_rmse_sum]
np.sum(weight)
# 1

#최종 모형과 비교
pred_final=weight[0]*pred_41+weight[1]*pred_42+weight[2]*pred_43+weight[3]*pred_44+weight[4]*pred_45+weight[5]*pred_46+weight[6]*pred_47+weight[7]*pred_48+weight[8]*pred_49+weight[9]*pred_50


np.sqrt((((np.expm1(y_c)-np.expm1(pred_final))**2).sum()))

# 양호한 결과
# 93424620574.83125

# 이제 누락치 처리.
pred_41_m=np.expm1(xgbr41.predict(x_pred_41))
pred_42_m=np.expm1(xgbr42.predict(x_pred_42))
pred_43_m=np.expm1(xgbr43.predict(x_pred_43))
pred_44_m=np.expm1(xgbr44.predict(x_pred_44))
pred_45_m=np.expm1(xgbr45.predict(x_pred_45))
pred_46_m=np.expm1(xgbr46.predict(x_pred_46))
pred_47_m=np.expm1(xgbr47.predict(x_pred_47))
pred_48_m=np.expm1(xgbr48.predict(x_pred_48))
pred_49_m=np.expm1(xgbr49.predict(x_pred_49))
pred_50_m=np.expm1(xgbr50.predict(x_pred_50))


pred_final=weight[0]*pred_41_m+weight[1]*pred_42_m+weight[2]*pred_43_m+weight[3]*pred_44_m+weight[4]*pred_45_m+weight[5]*pred_46_m+weight[6]*pred_47_m+weight[7]*pred_48_m+weight[8]*pred_49_m+weight[9]*pred_50_m

pred_trade=list(pred_final)

data.info()

for i,j in zip(list(index),pred_trade):
    data.loc[data.index==i,'TRADE_HSCD_COUNTRYCD']=j

feature_engineering(data)


##################################################################################



data.info()

model_data=data.drop(['UNC_YEAR','COUNTRYCD'],axis=1)

model_data.info()

model_cat=model_data[['dist','TARIFF','PA','BE']]
model_data=model_data.drop(['dist','TARIFF','PA','BE'],axis=1)

model_target=model_data['KR_TRADE_HSCD_COUNTRYCD']
model_data=model_data.drop('KR_TRADE_HSCD_COUNTRYCD',axis=1)

model_data2=pd.get_dummies(model_data,drop_first=True)
columns=model_data2.columns
model_data2.info()
model_num=model_data2.iloc[:,:22]
model_num.shape
model_num.head()
model_obj=model_data2.iloc[:,22:]
model_obj.shape


model_num=np.log1p(model_num)
model_target=np.log1p(model_target)

x=np.concatenate([model_num,model_obj,model_cat],axis=1)

d10=pd.DataFrame(x)
d10.columns=list(columns)+['dist','TARIFF','PA','BE']
d10['y']=model_target

d10.info()


mask=((d10['y']==0)&(d10['KR_2017']!=0))|((d10['y']!=0)&(d10['KR_2017']==0))
d100=d10[~mask]

model_f=d100
model_f=model_f.reset_index().drop('index',axis=1)
y=model_f['y']
x=model_f.drop('y',axis=1)
x=x[['TRADE_COUNTRYCD','KR_2017','SNDIST','TRADE_HSCD_COUNTRYCD','IC_BUS_EASE_DFRN_DB','KR_HSCD_COUNTRY_RATIO','KR_RATIO','GDP_RATIO','PA_NUS_FCRF','TARIFF_AVG','TRADE_HSCD_RATIO','SP_POP_TOTL','RDIST','KR_COUNTRY_RATIO','KMDIST','TRADE_HSCD','KR_HSCD_RATIO','NY_GDP_MKTP_CD','TRADE_RATIO']]
name=x.columns
ss=StandardScaler()
ss.fit(x)
x=ss.transform(x)
x=pd.DataFrame(x)
x.columns=list(name)

x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=41)

final_xgbr_41=XGBRegressor(loss_function='MAE',random_state=42)
final_xgbr_41.fit(x_train,y_train)
final_xgbr_41.score(x_train,y_train)
final_xgbr_41.score(x_test,y_test)

model_compare=d10[['TRADE_COUNTRYCD','KR_2017','SNDIST','TRADE_HSCD_COUNTRYCD','IC_BUS_EASE_DFRN_DB','KR_HSCD_COUNTRY_RATIO','KR_RATIO','GDP_RATIO','PA_NUS_FCRF','TARIFF_AVG','TRADE_HSCD_RATIO','SP_POP_TOTL','RDIST','KR_COUNTRY_RATIO','KMDIST','TRADE_HSCD','KR_HSCD_RATIO','NY_GDP_MKTP_CD','TRADE_RATIO','y']]
y_c=model_compare['y']
x_c=model_compare.drop('y',axis=1)
x2_c=x_c[['TRADE_COUNTRYCD','KR_2017','SNDIST','TRADE_HSCD_COUNTRYCD','IC_BUS_EASE_DFRN_DB','KR_HSCD_COUNTRY_RATIO','KR_RATIO','GDP_RATIO','PA_NUS_FCRF','TARIFF_AVG','TRADE_HSCD_RATIO','SP_POP_TOTL','RDIST','KR_COUNTRY_RATIO','KMDIST','TRADE_HSCD','KR_HSCD_RATIO','NY_GDP_MKTP_CD','TRADE_RATIO']]
name=x2_c.columns
x_num_c=ss.transform(x2_c)
x_c=pd.DataFrame(x_num_c)
x_c.columns=name
x_c.info()

pred=final_xgbr_41.predict(x_c)


print("변경 전\n")
print("train",final_xgbr_41.score(x_train,y_train),"\ntest",final_xgbr_41.score(x_test,y_test))
print("all",final_xgbr_41.score(x_c,y_c))
print("all MAE",mean_absolute_error(np.expm1(y_c),np.expm1(pred)))

#변경 전

#train 0.9462950848652445 
#test 0.8353098397563183
#all 0.8054919827882898
#all MAE 4886058.9520838205

residual=y_c-pred

residual.hist(figsize=(10,10))

out_index=residual[(((residual>5)|(residual<-5)))].index
out_index=list(out_index)

len(out_index)

model_f=d10
model_f=model_f.drop(out_index,axis=0)
mask2=((model_f['y']==0)&(model_f['KR_2017']!=0))|((model_f['y']!=0)&(model_f['KR_2017']==0))
model_f=model_f[~mask2]
model_f=model_f.reset_index().drop('index',axis=1)
y=model_f['y']
x=model_f.drop('y',axis=1)
x=x[['TRADE_COUNTRYCD','KR_2017','SNDIST','TRADE_HSCD_COUNTRYCD','IC_BUS_EASE_DFRN_DB','KR_HSCD_COUNTRY_RATIO','KR_RATIO','GDP_RATIO','PA_NUS_FCRF','TARIFF_AVG','TRADE_HSCD_RATIO','SP_POP_TOTL','RDIST','KR_COUNTRY_RATIO','KMDIST','TRADE_HSCD','KR_HSCD_RATIO','NY_GDP_MKTP_CD','TRADE_RATIO']]
name=x.columns
ss=StandardScaler()
ss.fit(x)
x=ss.transform(x)
x=pd.DataFrame(x)
x.columns=list(name)

x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=41)

final_xgbr_41=XGBRegressor(loss_function='MAE',random_state=42)
final_xgbr_41.fit(x_train,y_train)
final_xgbr_41.score(x_train,y_train)
final_xgbr_41.score(x_test,y_test)

model_compare=d10[['TRADE_COUNTRYCD','KR_2017','SNDIST','TRADE_HSCD_COUNTRYCD','IC_BUS_EASE_DFRN_DB','KR_HSCD_COUNTRY_RATIO','KR_RATIO','GDP_RATIO','PA_NUS_FCRF','TARIFF_AVG','TRADE_HSCD_RATIO','SP_POP_TOTL','RDIST','KR_COUNTRY_RATIO','KMDIST','TRADE_HSCD','KR_HSCD_RATIO','NY_GDP_MKTP_CD','TRADE_RATIO','y']]
y_c=model_compare['y']
x_c=model_compare.drop('y',axis=1)
x2_c=x_c[['TRADE_COUNTRYCD','KR_2017','SNDIST','TRADE_HSCD_COUNTRYCD','IC_BUS_EASE_DFRN_DB','KR_HSCD_COUNTRY_RATIO','KR_RATIO','GDP_RATIO','PA_NUS_FCRF','TARIFF_AVG','TRADE_HSCD_RATIO','SP_POP_TOTL','RDIST','KR_COUNTRY_RATIO','KMDIST','TRADE_HSCD','KR_HSCD_RATIO','NY_GDP_MKTP_CD','TRADE_RATIO']]
name=x2_c.columns
x_num_c=ss.transform(x2_c)
x_c=pd.DataFrame(x_num_c)
x_c.columns=name
x_c.info()

pred=final_xgbr_41.predict(x_c)

print("변경 후\n")
print("train",final_xgbr_41.score(x_train,y_train),"\ntest",final_xgbr_41.score(x_test,y_test))
print("all",final_xgbr_41.score(x_c,y_c))
print("all MAE",mean_absolute_error(np.expm1(y_c),np.expm1(pred)))

#변경 후

#train 0.9461892415280116 
#test 0.8535017319825502
#all 0.8000635405073062
#all MAE 5449709.797588638

residual2=y_c-pred

residual2.hist(figsize=(10,10))

out_index2=residual2[(((residual2>5)|(residual2<-5)))].index
out_index2=list(out_index2)


set1=set(out_index)
set2=set(out_index2)
out_index3=list(set1|set2)

len(out_index3)
### 두 번째 잔차 제거
model_f=d10
model_f=model_f.drop(out_index3,axis=0)
mask2=((model_f['y']==0)&(model_f['KR_2017']!=0))|((model_f['y']!=0)&(model_f['KR_2017']==0))
model_f=model_f[~mask2]
model_f=model_f.reset_index().drop('index',axis=1)
y=model_f['y']
x=model_f.drop('y',axis=1)
x=x[['TRADE_COUNTRYCD','KR_2017','SNDIST','TRADE_HSCD_COUNTRYCD','IC_BUS_EASE_DFRN_DB','KR_HSCD_COUNTRY_RATIO','KR_RATIO','GDP_RATIO','PA_NUS_FCRF','TARIFF_AVG','TRADE_HSCD_RATIO','SP_POP_TOTL','RDIST','KR_COUNTRY_RATIO','KMDIST','TRADE_HSCD','KR_HSCD_RATIO','NY_GDP_MKTP_CD','TRADE_RATIO']]
name=x.columns
ss=StandardScaler()
ss.fit(x)
x=ss.transform(x)
x=pd.DataFrame(x)
x.columns=list(name)

x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=41)

final_xgbr_41=XGBRegressor(loss_function='MAE',random_state=42)
final_xgbr_41.fit(x_train,y_train)
final_xgbr_41.score(x_train,y_train)
final_xgbr_41.score(x_test,y_test)

model_compare=d10[['TRADE_COUNTRYCD','KR_2017','SNDIST','TRADE_HSCD_COUNTRYCD','IC_BUS_EASE_DFRN_DB','KR_HSCD_COUNTRY_RATIO','KR_RATIO','GDP_RATIO','PA_NUS_FCRF','TARIFF_AVG','TRADE_HSCD_RATIO','SP_POP_TOTL','RDIST','KR_COUNTRY_RATIO','KMDIST','TRADE_HSCD','KR_HSCD_RATIO','NY_GDP_MKTP_CD','TRADE_RATIO','y']]
y_c=model_compare['y']
x_c=model_compare.drop('y',axis=1)
x2_c=x_c[['TRADE_COUNTRYCD','KR_2017','SNDIST','TRADE_HSCD_COUNTRYCD','IC_BUS_EASE_DFRN_DB','KR_HSCD_COUNTRY_RATIO','KR_RATIO','GDP_RATIO','PA_NUS_FCRF','TARIFF_AVG','TRADE_HSCD_RATIO','SP_POP_TOTL','RDIST','KR_COUNTRY_RATIO','KMDIST','TRADE_HSCD','KR_HSCD_RATIO','NY_GDP_MKTP_CD','TRADE_RATIO']]
name=x2_c.columns
x_num_c=ss.transform(x2_c)
x_c=pd.DataFrame(x_num_c)
x_c.columns=name
x_c.info()

pred=final_xgbr_41.predict(x_c)

print("변경 후\n")
print("train",final_xgbr_41.score(x_train,y_train),"\ntest",final_xgbr_41.score(x_test,y_test))
print("all",final_xgbr_41.score(x_c,y_c))
print("all MAE",mean_absolute_error(np.expm1(y_c),np.expm1(pred)))

residual3=y_c-pred

#train 0.9492506319592191 
#test 0.8563848422220977
#all 0.8013832248751789
#all MAE 4504031.479932943

out_index4=residual3[(((residual3>5)|(residual3<-5)))].index
out_index4=list(out_index4)

len(out_index4)

set4=set(out_index4)
set3=set(out_index3)
out_index5=list(set3|set4)

len(out_index5)

### 세 번째 잔차 제거
model_f=d10
model_f=model_f.drop(out_index5,axis=0)
mask2=((model_f['y']==0)&(model_f['KR_2017']!=0))|((model_f['y']!=0)&(model_f['KR_2017']==0))
model_f=model_f[~mask2]
model_f=model_f.reset_index().drop('index',axis=1)
y=model_f['y']
x=model_f.drop('y',axis=1)
x=x[['TRADE_COUNTRYCD','KR_2017','SNDIST','TRADE_HSCD_COUNTRYCD','IC_BUS_EASE_DFRN_DB','KR_HSCD_COUNTRY_RATIO','KR_RATIO','GDP_RATIO','PA_NUS_FCRF','TARIFF_AVG','TRADE_HSCD_RATIO','SP_POP_TOTL','RDIST','KR_COUNTRY_RATIO','KMDIST','TRADE_HSCD','KR_HSCD_RATIO','NY_GDP_MKTP_CD','TRADE_RATIO']]
name=x.columns
ss=StandardScaler()
ss.fit(x)
x=ss.transform(x)
x=pd.DataFrame(x)
x.columns=list(name)

x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=41)

final_xgbr_41=XGBRegressor(loss_function='MAE',random_state=42)
final_xgbr_41.fit(x_train,y_train)
final_xgbr_41.score(x_train,y_train)
final_xgbr_41.score(x_test,y_test)

model_compare=d10[['TRADE_COUNTRYCD','KR_2017','SNDIST','TRADE_HSCD_COUNTRYCD','IC_BUS_EASE_DFRN_DB','KR_HSCD_COUNTRY_RATIO','KR_RATIO','GDP_RATIO','PA_NUS_FCRF','TARIFF_AVG','TRADE_HSCD_RATIO','SP_POP_TOTL','RDIST','KR_COUNTRY_RATIO','KMDIST','TRADE_HSCD','KR_HSCD_RATIO','NY_GDP_MKTP_CD','TRADE_RATIO','y']]
y_c=model_compare['y']
x_c=model_compare.drop('y',axis=1)
x2_c=x_c[['TRADE_COUNTRYCD','KR_2017','SNDIST','TRADE_HSCD_COUNTRYCD','IC_BUS_EASE_DFRN_DB','KR_HSCD_COUNTRY_RATIO','KR_RATIO','GDP_RATIO','PA_NUS_FCRF','TARIFF_AVG','TRADE_HSCD_RATIO','SP_POP_TOTL','RDIST','KR_COUNTRY_RATIO','KMDIST','TRADE_HSCD','KR_HSCD_RATIO','NY_GDP_MKTP_CD','TRADE_RATIO']]
name=x2_c.columns
x_num_c=ss.transform(x2_c)
x_c=pd.DataFrame(x_num_c)
x_c.columns=name
x_c.info()

pred=final_xgbr_41.predict(x_c)

print("변경 후\n")
print("train",final_xgbr_41.score(x_train,y_train),"\ntest",final_xgbr_41.score(x_test,y_test))
print("all",final_xgbr_41.score(x_c,y_c))
print("all MAE",mean_absolute_error(np.expm1(y_c),np.expm1(pred)))

### 두 번째 잔차 제거 모형이 제일 좋음. 이걸로 변수선택 진행 
mask2=((model_f['y']==0)&(model_f['KR_2017']!=0))|((model_f['y']!=0)&(model_f['KR_2017']==0))
d10.info()
model_f=d10
model_f=model_f.drop(out_index3,axis=0)
model_f=model_f[~mask2]
model_f=model_f.reset_index().drop('index',axis=1)
f10=model_f

y=f10['y']
x=f10.drop('y',axis=1)
x=x[['TRADE_COUNTRYCD','KR_2017','SNDIST','TRADE_HSCD_COUNTRYCD','IC_BUS_EASE_DFRN_DB','KR_HSCD_COUNTRY_RATIO','KR_RATIO','GDP_RATIO','PA_NUS_FCRF','TARIFF_AVG','TRADE_HSCD_RATIO','SP_POP_TOTL','RDIST','KR_COUNTRY_RATIO','KMDIST','TRADE_HSCD','KR_HSCD_RATIO','NY_GDP_MKTP_CD','TRADE_RATIO']]
name=x.columns
ss=StandardScaler()
ss.fit(x)
x=ss.transform(x)
x=pd.DataFrame(x)
x.columns=list(name)

x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=41)

final_xgbr_41=XGBRegressor(loss_function='MAE',random_state=42)
final_xgbr_41.fit(x_train,y_train)
final_xgbr_41.score(x_train,y_train)
final_xgbr_41.score(x_test,y_test)

model_compare=d10[['TRADE_COUNTRYCD','KR_2017','SNDIST','TRADE_HSCD_COUNTRYCD','IC_BUS_EASE_DFRN_DB','KR_HSCD_COUNTRY_RATIO','KR_RATIO','GDP_RATIO','PA_NUS_FCRF','TARIFF_AVG','TRADE_HSCD_RATIO','SP_POP_TOTL','RDIST','KR_COUNTRY_RATIO','KMDIST','TRADE_HSCD','KR_HSCD_RATIO','NY_GDP_MKTP_CD','TRADE_RATIO','y']]
y_c=model_compare['y']
x_c=model_compare.drop('y',axis=1)
x2_c=x_c[['TRADE_COUNTRYCD','KR_2017','SNDIST','TRADE_HSCD_COUNTRYCD','IC_BUS_EASE_DFRN_DB','KR_HSCD_COUNTRY_RATIO','KR_RATIO','GDP_RATIO','PA_NUS_FCRF','TARIFF_AVG','TRADE_HSCD_RATIO','SP_POP_TOTL','RDIST','KR_COUNTRY_RATIO','KMDIST','TRADE_HSCD','KR_HSCD_RATIO','NY_GDP_MKTP_CD','TRADE_RATIO']]
name=x2_c.columns
x_num_c=ss.transform(x2_c)
x_c=pd.DataFrame(x_num_c)
x_c.columns=name
x_c.info()

pred=final_xgbr_41.predict(x_c)

print("변경 후\n")



## 변수 선택 전 
#train 0.9492506319592191 
#test 0.8563848422220977
#all 0.8013832248751789
#all MAE 4504031.479932943

for i in ['TRADE_COUNTRYCD','KR_2017','SNDIST','TRADE_HSCD_COUNTRYCD','IC_BUS_EASE_DFRN_DB','KR_HSCD_COUNTRY_RATIO','KR_RATIO','GDP_RATIO','PA_NUS_FCRF','TARIFF_AVG','TRADE_HSCD_RATIO','SP_POP_TOTL','RDIST','KR_COUNTRY_RATIO','KMDIST','TRADE_HSCD','KR_HSCD_RATIO','NY_GDP_MKTP_CD','TRADE_RATIO']:
    model_f=f10
    model_f=model_f.reset_index().drop('index',axis=1)
    y=model_f['y']
    x=model_f.drop('y',axis=1)
    x=x[['TRADE_COUNTRYCD','KR_2017','SNDIST','TRADE_HSCD_COUNTRYCD','IC_BUS_EASE_DFRN_DB','KR_HSCD_COUNTRY_RATIO','KR_RATIO','GDP_RATIO','PA_NUS_FCRF','TARIFF_AVG','TRADE_HSCD_RATIO','SP_POP_TOTL','RDIST','KR_COUNTRY_RATIO','KMDIST','TRADE_HSCD','KR_HSCD_RATIO','NY_GDP_MKTP_CD','TRADE_RATIO']]
    x=x.drop(i,axis=1)
    name=x.columns
    ss=StandardScaler()
    ss.fit(x)
    x=ss.transform(x)
    x=pd.DataFrame(x)
    x.columns=list(name)
    
    x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=41)


    final_xgbr_41=XGBRegressor(random_state=42)
    final_xgbr_41.fit(x_train,y_train)

    model_compare=d10[['TRADE_COUNTRYCD','KR_2017','SNDIST','TRADE_HSCD_COUNTRYCD','IC_BUS_EASE_DFRN_DB','KR_HSCD_COUNTRY_RATIO','KR_RATIO','GDP_RATIO','PA_NUS_FCRF','TARIFF_AVG','TRADE_HSCD_RATIO','SP_POP_TOTL','RDIST','KR_COUNTRY_RATIO','KMDIST','TRADE_HSCD','KR_HSCD_RATIO','NY_GDP_MKTP_CD','TRADE_RATIO','y']]
    y_c=model_compare['y']
    x_c=model_compare.drop('y',axis=1)
    x2_c=x_c[['TRADE_COUNTRYCD','KR_2017','SNDIST','TRADE_HSCD_COUNTRYCD','IC_BUS_EASE_DFRN_DB','KR_HSCD_COUNTRY_RATIO','KR_RATIO','GDP_RATIO','PA_NUS_FCRF','TARIFF_AVG','TRADE_HSCD_RATIO','SP_POP_TOTL','RDIST','KR_COUNTRY_RATIO','KMDIST','TRADE_HSCD','KR_HSCD_RATIO','NY_GDP_MKTP_CD','TRADE_RATIO']]
    x2_c=x2_c.drop(i,axis=1)
    name=x2_c.columns
    x_num_c=ss.transform(x2_c)
    x_c=pd.DataFrame(x_num_c)
    x_c.columns=list(name)
    
    pred=final_xgbr_41.predict(x_c)
    print(i);
    print("train",final_xgbr_41.score(x_train,y_train),"\ntest",final_xgbr_41.score(x_test,y_test))
    print("all",final_xgbr_41.score(x_c,y_c))
    print("all MAE",mean_absolute_error(np.expm1(y_c),np.expm1(pred)))

## 첫 번째 변수 선택
#PA_NUS_FCRF
#train 0.948106878966641 
#test 0.8561118831809057
#all 0.8002010002513746
#all MAE 4092389.1831766414

for i in ['TRADE_COUNTRYCD','KR_2017','SNDIST','TRADE_HSCD_COUNTRYCD','IC_BUS_EASE_DFRN_DB','KR_HSCD_COUNTRY_RATIO','KR_RATIO','GDP_RATIO','TARIFF_AVG','TRADE_HSCD_RATIO','SP_POP_TOTL','RDIST','KR_COUNTRY_RATIO','KMDIST','TRADE_HSCD','KR_HSCD_RATIO','NY_GDP_MKTP_CD','TRADE_RATIO']:
    model_f=f10
    model_f=model_f.reset_index().drop('index',axis=1)
    y=model_f['y']
    x=model_f.drop('y',axis=1)
    x=x[['TRADE_COUNTRYCD','KR_2017','SNDIST','TRADE_HSCD_COUNTRYCD','IC_BUS_EASE_DFRN_DB','KR_HSCD_COUNTRY_RATIO','KR_RATIO','GDP_RATIO','TARIFF_AVG','TRADE_HSCD_RATIO','SP_POP_TOTL','RDIST','KR_COUNTRY_RATIO','KMDIST','TRADE_HSCD','KR_HSCD_RATIO','NY_GDP_MKTP_CD','TRADE_RATIO']]
    x=x.drop(i,axis=1)
    name=x.columns
    ss=StandardScaler()
    ss.fit(x)
    x=ss.transform(x)
    x=pd.DataFrame(x)
    x.columns=list(name)
    
    x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=41)


    final_xgbr_41=XGBRegressor(random_state=42)
    final_xgbr_41.fit(x_train,y_train)

    model_compare=d10[['TRADE_COUNTRYCD','KR_2017','SNDIST','TRADE_HSCD_COUNTRYCD','IC_BUS_EASE_DFRN_DB','KR_HSCD_COUNTRY_RATIO','KR_RATIO','GDP_RATIO','TARIFF_AVG','TRADE_HSCD_RATIO','SP_POP_TOTL','RDIST','KR_COUNTRY_RATIO','KMDIST','TRADE_HSCD','KR_HSCD_RATIO','NY_GDP_MKTP_CD','TRADE_RATIO','y']]
    y_c=model_compare['y']
    x_c=model_compare.drop('y',axis=1)
    x2_c=x_c[['TRADE_COUNTRYCD','KR_2017','SNDIST','TRADE_HSCD_COUNTRYCD','IC_BUS_EASE_DFRN_DB','KR_HSCD_COUNTRY_RATIO','KR_RATIO','GDP_RATIO','TARIFF_AVG','TRADE_HSCD_RATIO','SP_POP_TOTL','RDIST','KR_COUNTRY_RATIO','KMDIST','TRADE_HSCD','KR_HSCD_RATIO','NY_GDP_MKTP_CD','TRADE_RATIO']]
    x2_c=x2_c.drop(i,axis=1)
    name=x2_c.columns
    x_num_c=ss.transform(x2_c)
    x_c=pd.DataFrame(x_num_c)
    x_c.columns=list(name)
    
    pred=final_xgbr_41.predict(x_c)
    print(i);
    print("train",final_xgbr_41.score(x_train,y_train),"\ntest",final_xgbr_41.score(x_test,y_test))
    print("all",final_xgbr_41.score(x_c,y_c))
    print("all MAE",mean_absolute_error(np.expm1(y_c),np.expm1(pred)))
    print("\n")

## MAE 측면에서 좋아진게 없음 변수 종료. 
## 현재 고려한 모형에서 잔차 및 이상치 확인 및 처리
model_f=f10
model_f=model_f.reset_index().drop('index',axis=1)
y=model_f['y']
x=model_f.drop('y',axis=1)
x=x[['TRADE_COUNTRYCD','KR_2017','SNDIST','TRADE_HSCD_COUNTRYCD','IC_BUS_EASE_DFRN_DB','KR_HSCD_COUNTRY_RATIO','KR_RATIO','GDP_RATIO','TARIFF_AVG','TRADE_HSCD_RATIO','SP_POP_TOTL','RDIST','KR_COUNTRY_RATIO','KMDIST','TRADE_HSCD','KR_HSCD_RATIO','NY_GDP_MKTP_CD','TRADE_RATIO']]
name=x.columns
ss=StandardScaler()
ss.fit(x)
x=ss.transform(x)
x=pd.DataFrame(x)
x.columns=list(name)

x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=41)


final_xgbr_41=XGBRegressor(random_state=42)
final_xgbr_41.fit(x_train,y_train)

model_compare=d10[['TRADE_COUNTRYCD','KR_2017','SNDIST','TRADE_HSCD_COUNTRYCD','IC_BUS_EASE_DFRN_DB','KR_HSCD_COUNTRY_RATIO','KR_RATIO','GDP_RATIO','TARIFF_AVG','TRADE_HSCD_RATIO','SP_POP_TOTL','RDIST','KR_COUNTRY_RATIO','KMDIST','TRADE_HSCD','KR_HSCD_RATIO','NY_GDP_MKTP_CD','TRADE_RATIO','y']]
y_c=model_compare['y']
x_c=model_compare.drop('y',axis=1)
x2_c=x_c[['TRADE_COUNTRYCD','KR_2017','SNDIST','TRADE_HSCD_COUNTRYCD','IC_BUS_EASE_DFRN_DB','KR_HSCD_COUNTRY_RATIO','KR_RATIO','GDP_RATIO','TARIFF_AVG','TRADE_HSCD_RATIO','SP_POP_TOTL','RDIST','KR_COUNTRY_RATIO','KMDIST','TRADE_HSCD','KR_HSCD_RATIO','NY_GDP_MKTP_CD','TRADE_RATIO']]
name=x2_c.columns
x_num_c=ss.transform(x2_c)
x_c=pd.DataFrame(x_num_c)
x_c.columns=list(name)

pred=final_xgbr_41.predict(x_c)
print(i);
print("train",final_xgbr_41.score(x_train,y_train),"\ntest",final_xgbr_41.score(x_test,y_test))
print("all",final_xgbr_41.score(x_c,y_c))
print("all MAE",mean_absolute_error(np.expm1(y_c),np.expm1(pred)))
print("\n")

res_41=y_c-pred
res_41.hist(figsize=(10,10))

out_ind_41=res_41[(((res_41>5)|(res_41<-5)))].index
out_ind_41=list(out_ind_41)

len(out_index4)

set4=set(out_ind_41)
set3=set(out_index3)
out_index5=list(set3|set4)

len(out_index5)
model_f=d10
mask2=((model_f['y']==0)&(model_f['KR_2017']!=0))|((model_f['y']!=0)&(model_f['KR_2017']==0))
model_f=model_f.drop(out_index5,axis=0)
model_f=model_f[~mask2]
model_f=model_f.reset_index().drop('index',axis=1)
y=model_f['y']
x=model_f.drop('y',axis=1)
x=x[['TRADE_COUNTRYCD','KR_2017','SNDIST','TRADE_HSCD_COUNTRYCD','IC_BUS_EASE_DFRN_DB','KR_HSCD_COUNTRY_RATIO','KR_RATIO','GDP_RATIO','PA_NUS_FCRF','TARIFF_AVG','TRADE_HSCD_RATIO','SP_POP_TOTL','RDIST','KR_COUNTRY_RATIO','KMDIST','TRADE_HSCD','KR_HSCD_RATIO','NY_GDP_MKTP_CD','TRADE_RATIO']]
name=x.columns
ss=StandardScaler()
ss.fit(x)
x=ss.transform(x)
x=pd.DataFrame(x)
x.columns=list(name)

x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=41)

final_xgbr_41=XGBRegressor(loss_function='MAE',random_state=42)
final_xgbr_41.fit(x_train,y_train)
final_xgbr_41.score(x_train,y_train)
final_xgbr_41.score(x_test,y_test)

model_compare=d10[['TRADE_COUNTRYCD','KR_2017','SNDIST','TRADE_HSCD_COUNTRYCD','IC_BUS_EASE_DFRN_DB','KR_HSCD_COUNTRY_RATIO','KR_RATIO','GDP_RATIO','PA_NUS_FCRF','TARIFF_AVG','TRADE_HSCD_RATIO','SP_POP_TOTL','RDIST','KR_COUNTRY_RATIO','KMDIST','TRADE_HSCD','KR_HSCD_RATIO','NY_GDP_MKTP_CD','TRADE_RATIO','y']]
y_c=model_compare['y']
x_c=model_compare.drop('y',axis=1)
x2_c=x_c[['TRADE_COUNTRYCD','KR_2017','SNDIST','TRADE_HSCD_COUNTRYCD','IC_BUS_EASE_DFRN_DB','KR_HSCD_COUNTRY_RATIO','KR_RATIO','GDP_RATIO','PA_NUS_FCRF','TARIFF_AVG','TRADE_HSCD_RATIO','SP_POP_TOTL','RDIST','KR_COUNTRY_RATIO','KMDIST','TRADE_HSCD','KR_HSCD_RATIO','NY_GDP_MKTP_CD','TRADE_RATIO']]
name=x2_c.columns
x_num_c=ss.transform(x2_c)
x_c=pd.DataFrame(x_num_c)
x_c.columns=name
x_c.info()

pred=final_xgbr_41.predict(x_c)

print("변경 후\n")
print("train",final_xgbr_41.score(x_train,y_train),"\ntest",final_xgbr_41.score(x_test,y_test))
print("all",final_xgbr_41.score(x_c,y_c))
print("all MAE",mean_absolute_error(np.expm1(y_c),np.expm1(pred)))

res_41_2=y_c-pred

#변경 후

#train 0.9470878833097496 
#test 0.8608957684483639
#all 0.7969443743989976
#all MAE 5198948.495297681


out_ind_41_2=res_41_2[(((res_41_2>5)|(res_41_2<-5)))].index
out_ind_41_2=list(out_ind_41_2)

len(out_index4)

set4=set(out_ind_41_2)
set3=set(out_index5)
out_index5=list(set3|set4)

len(out_index5)
model_f=d10
mask2=((model_f['y']==0)&(model_f['KR_2017']!=0))|((model_f['y']!=0)&(model_f['KR_2017']==0))
model_f=model_f.drop(out_index5,axis=0)
model_f=model_f[~mask2]
model_f=model_f.reset_index().drop('index',axis=1)
y=model_f['y']
x=model_f.drop('y',axis=1)
x=x[['TRADE_COUNTRYCD','KR_2017','SNDIST','TRADE_HSCD_COUNTRYCD','IC_BUS_EASE_DFRN_DB','KR_HSCD_COUNTRY_RATIO','KR_RATIO','GDP_RATIO','PA_NUS_FCRF','TARIFF_AVG','TRADE_HSCD_RATIO','SP_POP_TOTL','RDIST','KR_COUNTRY_RATIO','KMDIST','TRADE_HSCD','KR_HSCD_RATIO','NY_GDP_MKTP_CD','TRADE_RATIO']]
name=x.columns
ss=StandardScaler()
ss.fit(x)
x=ss.transform(x)
x=pd.DataFrame(x)
x.columns=list(name)

x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=41)

final_xgbr_41=XGBRegressor(loss_function='MAE',random_state=42)
final_xgbr_41.fit(x_train,y_train)
final_xgbr_41.score(x_train,y_train)
final_xgbr_41.score(x_test,y_test)

model_compare=d10[['TRADE_COUNTRYCD','KR_2017','SNDIST','TRADE_HSCD_COUNTRYCD','IC_BUS_EASE_DFRN_DB','KR_HSCD_COUNTRY_RATIO','KR_RATIO','GDP_RATIO','PA_NUS_FCRF','TARIFF_AVG','TRADE_HSCD_RATIO','SP_POP_TOTL','RDIST','KR_COUNTRY_RATIO','KMDIST','TRADE_HSCD','KR_HSCD_RATIO','NY_GDP_MKTP_CD','TRADE_RATIO','y']]
y_c=model_compare['y']
x_c=model_compare.drop('y',axis=1)
x2_c=x_c[['TRADE_COUNTRYCD','KR_2017','SNDIST','TRADE_HSCD_COUNTRYCD','IC_BUS_EASE_DFRN_DB','KR_HSCD_COUNTRY_RATIO','KR_RATIO','GDP_RATIO','PA_NUS_FCRF','TARIFF_AVG','TRADE_HSCD_RATIO','SP_POP_TOTL','RDIST','KR_COUNTRY_RATIO','KMDIST','TRADE_HSCD','KR_HSCD_RATIO','NY_GDP_MKTP_CD','TRADE_RATIO']]
name=x2_c.columns
x_num_c=ss.transform(x2_c)
x_c=pd.DataFrame(x_num_c)
x_c.columns=name
x_c.info()

pred=final_xgbr_41.predict(x_c)

print("변경 후\n")
print("train",final_xgbr_41.score(x_train,y_train),"\ntest",final_xgbr_41.score(x_test,y_test))
print("all",final_xgbr_41.score(x_c,y_c))
print("all MAE",mean_absolute_error(np.expm1(y_c),np.expm1(pred)))

res_41_3=y_c-pred
#train 0.9478365164267478 
#test 0.8644741762088848
#all 0.793694550112963
#all MAE 5305500.206481213

out_ind_41_3=res_41_3[(((res_41_3>5)|(res_41_3<-5)))].index
out_ind_41_3=list(out_ind_41_3)


set4=set(out_ind_41_3)
set3=set(out_index5)
out_index5=list(set3|set4)

len(out_index5)
model_f=d10
mask2=((model_f['y']==0)&(model_f['KR_2017']!=0))|((model_f['y']!=0)&(model_f['KR_2017']==0))
model_f=model_f.drop(out_index5,axis=0)
model_f=model_f[~mask2]
model_f=model_f.reset_index().drop('index',axis=1)
y=model_f['y']
x=model_f.drop('y',axis=1)
x=x[['TRADE_COUNTRYCD','KR_2017','SNDIST','TRADE_HSCD_COUNTRYCD','IC_BUS_EASE_DFRN_DB','KR_HSCD_COUNTRY_RATIO','KR_RATIO','GDP_RATIO','PA_NUS_FCRF','TARIFF_AVG','TRADE_HSCD_RATIO','SP_POP_TOTL','RDIST','KR_COUNTRY_RATIO','KMDIST','TRADE_HSCD','KR_HSCD_RATIO','NY_GDP_MKTP_CD','TRADE_RATIO']]
name=x.columns
ss=StandardScaler()
ss.fit(x)
x=ss.transform(x)
x=pd.DataFrame(x)
x.columns=list(name)

x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=41)

final_xgbr_41=XGBRegressor(random_state=42)
final_xgbr_41.fit(x_train,y_train)
final_xgbr_41.score(x_train,y_train)
final_xgbr_41.score(x_test,y_test)

model_compare=d10[['TRADE_COUNTRYCD','KR_2017','SNDIST','TRADE_HSCD_COUNTRYCD','IC_BUS_EASE_DFRN_DB','KR_HSCD_COUNTRY_RATIO','KR_RATIO','GDP_RATIO','PA_NUS_FCRF','TARIFF_AVG','TRADE_HSCD_RATIO','SP_POP_TOTL','RDIST','KR_COUNTRY_RATIO','KMDIST','TRADE_HSCD','KR_HSCD_RATIO','NY_GDP_MKTP_CD','TRADE_RATIO','y']]
y_c=model_compare['y']
x_c=model_compare.drop('y',axis=1)
x2_c=x_c[['TRADE_COUNTRYCD','KR_2017','SNDIST','TRADE_HSCD_COUNTRYCD','IC_BUS_EASE_DFRN_DB','KR_HSCD_COUNTRY_RATIO','KR_RATIO','GDP_RATIO','PA_NUS_FCRF','TARIFF_AVG','TRADE_HSCD_RATIO','SP_POP_TOTL','RDIST','KR_COUNTRY_RATIO','KMDIST','TRADE_HSCD','KR_HSCD_RATIO','NY_GDP_MKTP_CD','TRADE_RATIO']]
name=x2_c.columns
x_num_c=ss.transform(x2_c)
x_c=pd.DataFrame(x_num_c)
x_c.columns=name
x_c.info()

pred=final_xgbr_41.predict(x_c)

print("변경 후\n")
print("train",final_xgbr_41.score(x_train,y_train),"\ntest",final_xgbr_41.score(x_test,y_test))
print("all",final_xgbr_41.score(x_c,y_c))
print("all MAE",mean_absolute_error(np.expm1(y_c),np.expm1(pred)))

res_41_4=y_c-pred

#변경 후

#train 0.9475344233686508 
#test 0.8703059652019135
#all 0.7943544968768544
#all MAE 6033029.423601016

out_ind_41_4=res_41_4[(((res_41_4>5)|(res_41_4<-5)))].index
out_ind_41_4=list(out_ind_41_4)


set4=set(out_ind_41_4)
set3=set(out_index5)
out_index5=list(set3|set4)

len(out_index5)
model_f=d10
mask2=((model_f['y']==0)&(model_f['KR_2017']!=0))|((model_f['y']!=0)&(model_f['KR_2017']==0))
model_f=model_f.drop(out_index5,axis=0)
model_f=model_f[~mask2]
model_f=model_f.reset_index().drop('index',axis=1)
y=model_f['y']
x=model_f.drop('y',axis=1)
x=x[['TRADE_COUNTRYCD','KR_2017','SNDIST','TRADE_HSCD_COUNTRYCD','IC_BUS_EASE_DFRN_DB','KR_HSCD_COUNTRY_RATIO','KR_RATIO','GDP_RATIO','PA_NUS_FCRF','TARIFF_AVG','TRADE_HSCD_RATIO','SP_POP_TOTL','RDIST','KR_COUNTRY_RATIO','KMDIST','TRADE_HSCD','KR_HSCD_RATIO','NY_GDP_MKTP_CD','TRADE_RATIO']]
name=x.columns
ss=StandardScaler()
ss.fit(x)
x=ss.transform(x)
x=pd.DataFrame(x)
x.columns=list(name)

x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=41)

final_xgbr_41=XGBRegressor(random_state=42)
final_xgbr_41.fit(x_train,y_train)
final_xgbr_41.score(x_train,y_train)
final_xgbr_41.score(x_test,y_test)

model_compare=d10[['TRADE_COUNTRYCD','KR_2017','SNDIST','TRADE_HSCD_COUNTRYCD','IC_BUS_EASE_DFRN_DB','KR_HSCD_COUNTRY_RATIO','KR_RATIO','GDP_RATIO','PA_NUS_FCRF','TARIFF_AVG','TRADE_HSCD_RATIO','SP_POP_TOTL','RDIST','KR_COUNTRY_RATIO','KMDIST','TRADE_HSCD','KR_HSCD_RATIO','NY_GDP_MKTP_CD','TRADE_RATIO','y']]
y_c=model_compare['y']
x_c=model_compare.drop('y',axis=1)
x2_c=x_c[['TRADE_COUNTRYCD','KR_2017','SNDIST','TRADE_HSCD_COUNTRYCD','IC_BUS_EASE_DFRN_DB','KR_HSCD_COUNTRY_RATIO','KR_RATIO','GDP_RATIO','PA_NUS_FCRF','TARIFF_AVG','TRADE_HSCD_RATIO','SP_POP_TOTL','RDIST','KR_COUNTRY_RATIO','KMDIST','TRADE_HSCD','KR_HSCD_RATIO','NY_GDP_MKTP_CD','TRADE_RATIO']]
name=x2_c.columns
x_num_c=ss.transform(x2_c)
x_c=pd.DataFrame(x_num_c)
x_c.columns=name
x_c.info()

pred=final_xgbr_41.predict(x_c)

print("변경 후\n")
print("train",final_xgbr_41.score(x_train,y_train),"\ntest",final_xgbr_41.score(x_test,y_test))
print("all",final_xgbr_41.score(x_c,y_c))
print("all MAE",mean_absolute_error(np.expm1(y_c),np.expm1(pred)))

res_41_5=y_c-pred

#train 0.9491460125044863 
#test 0.8741070861537267
#all 0.7967998317985859
#all MAE 4744369.916223232

out_ind_41_5=res_41_5[(((res_41_5>5)|(res_41_5<-5)))].index
out_ind_41_5=list(out_ind_41_5)


set4=set(out_ind_41_5)
set3=set(out_index5)
out_index5=list(set3|set4)

len(out_index5)
model_f=d10
mask2=((model_f['y']==0)&(model_f['KR_2017']!=0))|((model_f['y']!=0)&(model_f['KR_2017']==0))
model_f=model_f.drop(out_index5,axis=0)
model_f=model_f[~mask2]
model_f=model_f.reset_index().drop('index',axis=1)
y=model_f['y']
x=model_f.drop('y',axis=1)
x=x[['TRADE_COUNTRYCD','KR_2017','SNDIST','TRADE_HSCD_COUNTRYCD','IC_BUS_EASE_DFRN_DB','KR_HSCD_COUNTRY_RATIO','KR_RATIO','GDP_RATIO','PA_NUS_FCRF','TARIFF_AVG','TRADE_HSCD_RATIO','SP_POP_TOTL','RDIST','KR_COUNTRY_RATIO','KMDIST','TRADE_HSCD','KR_HSCD_RATIO','NY_GDP_MKTP_CD','TRADE_RATIO']]
name=x.columns
ss=StandardScaler()
ss.fit(x)
x=ss.transform(x)
x=pd.DataFrame(x)
x.columns=list(name)

x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=41)

final_xgbr_41=XGBRegressor(random_state=42)
final_xgbr_41.fit(x_train,y_train)
final_xgbr_41.score(x_train,y_train)
final_xgbr_41.score(x_test,y_test)

model_compare=d10[['TRADE_COUNTRYCD','KR_2017','SNDIST','TRADE_HSCD_COUNTRYCD','IC_BUS_EASE_DFRN_DB','KR_HSCD_COUNTRY_RATIO','KR_RATIO','GDP_RATIO','PA_NUS_FCRF','TARIFF_AVG','TRADE_HSCD_RATIO','SP_POP_TOTL','RDIST','KR_COUNTRY_RATIO','KMDIST','TRADE_HSCD','KR_HSCD_RATIO','NY_GDP_MKTP_CD','TRADE_RATIO','y']]
y_c=model_compare['y']
x_c=model_compare.drop('y',axis=1)
x2_c=x_c[['TRADE_COUNTRYCD','KR_2017','SNDIST','TRADE_HSCD_COUNTRYCD','IC_BUS_EASE_DFRN_DB','KR_HSCD_COUNTRY_RATIO','KR_RATIO','GDP_RATIO','PA_NUS_FCRF','TARIFF_AVG','TRADE_HSCD_RATIO','SP_POP_TOTL','RDIST','KR_COUNTRY_RATIO','KMDIST','TRADE_HSCD','KR_HSCD_RATIO','NY_GDP_MKTP_CD','TRADE_RATIO']]
name=x2_c.columns
x_num_c=ss.transform(x2_c)
x_c=pd.DataFrame(x_num_c)
x_c.columns=name
x_c.info()

pred=final_xgbr_41.predict(x_c)

print("변경 후\n")
print("train",final_xgbr_41.score(x_train,y_train),"\ntest",final_xgbr_41.score(x_test,y_test))
print("all",final_xgbr_41.score(x_c,y_c))
print("all MAE",mean_absolute_error(np.expm1(y_c),np.expm1(pred)))

res_41_6=y_c-pred

#train 0.9487274767518037 
#test 0.8645407814848932
#all 0.7917018126297304
#all MAE 4267349.691017757

out_ind_41_6=res_41_6[(((res_41_6>5)|(res_41_5<-5)))].index
out_ind_41_6=list(out_ind_41_6)


set4=set(out_ind_41_6)
set3=set(out_index5)
out_index5=list(set3|set4)

len(out_index5)
model_f=d10
mask2=((model_f['y']==0)&(model_f['KR_2017']!=0))|((model_f['y']!=0)&(model_f['KR_2017']==0))
model_f=model_f.drop(out_index5,axis=0)
model_f=model_f[~mask2]
model_f=model_f.reset_index().drop('index',axis=1)
y=model_f['y']
x=model_f.drop('y',axis=1)
x=x[['TRADE_COUNTRYCD','KR_2017','SNDIST','TRADE_HSCD_COUNTRYCD','IC_BUS_EASE_DFRN_DB','KR_HSCD_COUNTRY_RATIO','KR_RATIO','GDP_RATIO','PA_NUS_FCRF','TARIFF_AVG','TRADE_HSCD_RATIO','SP_POP_TOTL','RDIST','KR_COUNTRY_RATIO','KMDIST','TRADE_HSCD','KR_HSCD_RATIO','NY_GDP_MKTP_CD','TRADE_RATIO']]
name=x.columns
ss=StandardScaler()
ss.fit(x)
x=ss.transform(x)
x=pd.DataFrame(x)
x.columns=list(name)

x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=41)

final_xgbr_41=XGBRegressor(random_state=42)
final_xgbr_41.fit(x_train,y_train)
final_xgbr_41.score(x_train,y_train)
final_xgbr_41.score(x_test,y_test)

model_compare=d10[['TRADE_COUNTRYCD','KR_2017','SNDIST','TRADE_HSCD_COUNTRYCD','IC_BUS_EASE_DFRN_DB','KR_HSCD_COUNTRY_RATIO','KR_RATIO','GDP_RATIO','PA_NUS_FCRF','TARIFF_AVG','TRADE_HSCD_RATIO','SP_POP_TOTL','RDIST','KR_COUNTRY_RATIO','KMDIST','TRADE_HSCD','KR_HSCD_RATIO','NY_GDP_MKTP_CD','TRADE_RATIO','y']]
y_c=model_compare['y']
x_c=model_compare.drop('y',axis=1)
x2_c=x_c[['TRADE_COUNTRYCD','KR_2017','SNDIST','TRADE_HSCD_COUNTRYCD','IC_BUS_EASE_DFRN_DB','KR_HSCD_COUNTRY_RATIO','KR_RATIO','GDP_RATIO','PA_NUS_FCRF','TARIFF_AVG','TRADE_HSCD_RATIO','SP_POP_TOTL','RDIST','KR_COUNTRY_RATIO','KMDIST','TRADE_HSCD','KR_HSCD_RATIO','NY_GDP_MKTP_CD','TRADE_RATIO']]
name=x2_c.columns
x_num_c=ss.transform(x2_c)
x_c=pd.DataFrame(x_num_c)
x_c.columns=name
x_c.info()

pred=final_xgbr_41.predict(x_c)

print("변경 후\n")
print("train",final_xgbr_41.score(x_train,y_train),"\ntest",final_xgbr_41.score(x_test,y_test))
print("all",final_xgbr_41.score(x_c,y_c))
print("all MAE",mean_absolute_error(np.expm1(y_c),np.expm1(pred)))

res_41_7=y_c-pred
#변경 후

#train 0.9499205089170668 
#test 0.877340331660095
#all 0.7950160635572254
#all MAE 4325832.809369137

out_ind_41_7=res_41_7[(((res_41_7>5)|(res_41_7<-5)))].index
out_ind_41_7=list(out_ind_41_7)


set4=set(out_ind_41_7)
set3=set(out_index5)
out_index5=list(set3|set4)

len(out_index5)
model_f=d10
mask2=((model_f['y']==0)&(model_f['KR_2017']!=0))|((model_f['y']!=0)&(model_f['KR_2017']==0))
model_f=model_f.drop(out_index5,axis=0)
model_f=model_f[~mask2]
model_f=model_f.reset_index().drop('index',axis=1)
y=model_f['y']
x=model_f.drop('y',axis=1)
x=x[['TRADE_COUNTRYCD','KR_2017','SNDIST','TRADE_HSCD_COUNTRYCD','IC_BUS_EASE_DFRN_DB','KR_HSCD_COUNTRY_RATIO','KR_RATIO','GDP_RATIO','PA_NUS_FCRF','TARIFF_AVG','TRADE_HSCD_RATIO','SP_POP_TOTL','RDIST','KR_COUNTRY_RATIO','KMDIST','TRADE_HSCD','KR_HSCD_RATIO','NY_GDP_MKTP_CD','TRADE_RATIO']]
name=x.columns
ss=StandardScaler()
ss.fit(x)
x=ss.transform(x)
x=pd.DataFrame(x)
x.columns=list(name)

x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=41)

final_xgbr_41=XGBRegressor(random_state=42)
final_xgbr_41.fit(x_train,y_train)
final_xgbr_41.score(x_train,y_train)
final_xgbr_41.score(x_test,y_test)

model_compare=d10[['TRADE_COUNTRYCD','KR_2017','SNDIST','TRADE_HSCD_COUNTRYCD','IC_BUS_EASE_DFRN_DB','KR_HSCD_COUNTRY_RATIO','KR_RATIO','GDP_RATIO','PA_NUS_FCRF','TARIFF_AVG','TRADE_HSCD_RATIO','SP_POP_TOTL','RDIST','KR_COUNTRY_RATIO','KMDIST','TRADE_HSCD','KR_HSCD_RATIO','NY_GDP_MKTP_CD','TRADE_RATIO','y']]
y_c=model_compare['y']
x_c=model_compare.drop('y',axis=1)
x2_c=x_c[['TRADE_COUNTRYCD','KR_2017','SNDIST','TRADE_HSCD_COUNTRYCD','IC_BUS_EASE_DFRN_DB','KR_HSCD_COUNTRY_RATIO','KR_RATIO','GDP_RATIO','PA_NUS_FCRF','TARIFF_AVG','TRADE_HSCD_RATIO','SP_POP_TOTL','RDIST','KR_COUNTRY_RATIO','KMDIST','TRADE_HSCD','KR_HSCD_RATIO','NY_GDP_MKTP_CD','TRADE_RATIO']]
name=x2_c.columns
x_num_c=ss.transform(x2_c)
x_c=pd.DataFrame(x_num_c)
x_c.columns=name
x_c.info()

pred=final_xgbr_41.predict(x_c)

print("변경 후\n")
print("train",final_xgbr_41.score(x_train,y_train),"\ntest",final_xgbr_41.score(x_test,y_test))
print("all",final_xgbr_41.score(x_c,y_c))
print("all MAE",mean_absolute_error(np.expm1(y_c),np.expm1(pred)))

#변경 후

#train 0.9499845243333898 
#test 0.8698915295090041
#all 0.7922663690093542
#all MAE 4803396.753473907

res_41_8=y_c-pred

out_ind_41_8=res_41_8[(((res_41_8>5)|(res_41_8<-5)))].index
out_ind_41_8=list(out_ind_41_8)


set4=set(out_ind_41_8)
set3=set(out_index5)
out_index5=list(set3|set4)

len(out_index5)
model_f=d10
mask2=((model_f['y']==0)&(model_f['KR_2017']!=0))|((model_f['y']!=0)&(model_f['KR_2017']==0))
model_f=model_f.drop(out_index5,axis=0)
model_f=model_f[~mask2]
model_f=model_f.reset_index().drop('index',axis=1)
y=model_f['y']
x=model_f.drop('y',axis=1)
x=x[['TRADE_COUNTRYCD','KR_2017','SNDIST','TRADE_HSCD_COUNTRYCD','IC_BUS_EASE_DFRN_DB','KR_HSCD_COUNTRY_RATIO','KR_RATIO','GDP_RATIO','PA_NUS_FCRF','TARIFF_AVG','TRADE_HSCD_RATIO','SP_POP_TOTL','RDIST','KR_COUNTRY_RATIO','KMDIST','TRADE_HSCD','KR_HSCD_RATIO','NY_GDP_MKTP_CD','TRADE_RATIO']]
name=x.columns
ss=StandardScaler()
ss.fit(x)
x=ss.transform(x)
x=pd.DataFrame(x)
x.columns=list(name)

x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=41)

final_xgbr_41=XGBRegressor(random_state=42)
final_xgbr_41.fit(x_train,y_train)
final_xgbr_41.score(x_train,y_train)
final_xgbr_41.score(x_test,y_test)

model_compare=d10[['TRADE_COUNTRYCD','KR_2017','SNDIST','TRADE_HSCD_COUNTRYCD','IC_BUS_EASE_DFRN_DB','KR_HSCD_COUNTRY_RATIO','KR_RATIO','GDP_RATIO','PA_NUS_FCRF','TARIFF_AVG','TRADE_HSCD_RATIO','SP_POP_TOTL','RDIST','KR_COUNTRY_RATIO','KMDIST','TRADE_HSCD','KR_HSCD_RATIO','NY_GDP_MKTP_CD','TRADE_RATIO','y']]
y_c=model_compare['y']
x_c=model_compare.drop('y',axis=1)
x2_c=x_c[['TRADE_COUNTRYCD','KR_2017','SNDIST','TRADE_HSCD_COUNTRYCD','IC_BUS_EASE_DFRN_DB','KR_HSCD_COUNTRY_RATIO','KR_RATIO','GDP_RATIO','PA_NUS_FCRF','TARIFF_AVG','TRADE_HSCD_RATIO','SP_POP_TOTL','RDIST','KR_COUNTRY_RATIO','KMDIST','TRADE_HSCD','KR_HSCD_RATIO','NY_GDP_MKTP_CD','TRADE_RATIO']]
name=x2_c.columns
x_num_c=ss.transform(x2_c)
x_c=pd.DataFrame(x_num_c)
x_c.columns=name
x_c.info()

pred=final_xgbr_41.predict(x_c)

print("변경 후\n")
print("train",final_xgbr_41.score(x_train,y_train),"\ntest",final_xgbr_41.score(x_test,y_test))
print("all",final_xgbr_41.score(x_c,y_c))
print("all MAE",mean_absolute_error(np.expm1(y_c),np.expm1(pred)))


#변경 후

#train 0.9487679313331744 
#test 0.8694867762841038
#all 0.7915104890099929
#all MAE 4687724.935362783


res_41_9=y_c-pred

out_ind_41_9=res_41_9[(((res_41_9>5)|(res_41_9<-5)))].index
out_ind_41_9=list(out_ind_41_9)


set4=set(out_ind_41_9)
set3=set(out_index5)
out_index5=list(set3|set4)

len(out_index5)
model_f=d10
mask2=((model_f['y']==0)&(model_f['KR_2017']!=0))|((model_f['y']!=0)&(model_f['KR_2017']==0))
model_f=model_f.drop(out_index5,axis=0)
model_f=model_f[~mask2]
model_f=model_f.reset_index().drop('index',axis=1)
y=model_f['y']
x=model_f.drop('y',axis=1)
x=x[['TRADE_COUNTRYCD','KR_2017','SNDIST','TRADE_HSCD_COUNTRYCD','IC_BUS_EASE_DFRN_DB','KR_HSCD_COUNTRY_RATIO','KR_RATIO','GDP_RATIO','PA_NUS_FCRF','TARIFF_AVG','TRADE_HSCD_RATIO','SP_POP_TOTL','RDIST','KR_COUNTRY_RATIO','KMDIST','TRADE_HSCD','KR_HSCD_RATIO','NY_GDP_MKTP_CD','TRADE_RATIO']]
name=x.columns
ss=StandardScaler()
ss.fit(x)
x=ss.transform(x)
x=pd.DataFrame(x)
x.columns=list(name)

x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=41)

final_xgbr_41=XGBRegressor(random_state=42)
final_xgbr_41.fit(x_train,y_train)
final_xgbr_41.score(x_train,y_train)
final_xgbr_41.score(x_test,y_test)

model_compare=d10[['TRADE_COUNTRYCD','KR_2017','SNDIST','TRADE_HSCD_COUNTRYCD','IC_BUS_EASE_DFRN_DB','KR_HSCD_COUNTRY_RATIO','KR_RATIO','GDP_RATIO','PA_NUS_FCRF','TARIFF_AVG','TRADE_HSCD_RATIO','SP_POP_TOTL','RDIST','KR_COUNTRY_RATIO','KMDIST','TRADE_HSCD','KR_HSCD_RATIO','NY_GDP_MKTP_CD','TRADE_RATIO','y']]
y_c=model_compare['y']
x_c=model_compare.drop('y',axis=1)
x2_c=x_c[['TRADE_COUNTRYCD','KR_2017','SNDIST','TRADE_HSCD_COUNTRYCD','IC_BUS_EASE_DFRN_DB','KR_HSCD_COUNTRY_RATIO','KR_RATIO','GDP_RATIO','PA_NUS_FCRF','TARIFF_AVG','TRADE_HSCD_RATIO','SP_POP_TOTL','RDIST','KR_COUNTRY_RATIO','KMDIST','TRADE_HSCD','KR_HSCD_RATIO','NY_GDP_MKTP_CD','TRADE_RATIO']]
name=x2_c.columns
x_num_c=ss.transform(x2_c)
x_c=pd.DataFrame(x_num_c)
x_c.columns=name
x_c.info()

pred=final_xgbr_41.predict(x_c)

print("변경 후\n")
print("train",final_xgbr_41.score(x_train,y_train),"\ntest",final_xgbr_41.score(x_test,y_test))
print("all",final_xgbr_41.score(x_c,y_c))
print("all MAE",mean_absolute_error(np.expm1(y_c),np.expm1(pred)))

#변경 후

# train 0.94978699671295 
# test 0.873544218355575
# all 0.7938898091659463
# all MAE 4680051.27857136

res_41_10=y_c-pred

out_ind_41_10=res_41_10[(((res_41_10>5)|(res_41_10<-5)))].index
out_ind_41_10=list(out_ind_41_10)


set4=set(out_ind_41_10)
set3=set(out_index5)
out_index5=list(set3|set4)

len(out_index5)
model_f=d10
mask2=((model_f['y']==0)&(model_f['KR_2017']!=0))|((model_f['y']!=0)&(model_f['KR_2017']==0))
model_f=model_f.drop(out_index5,axis=0)
model_f=model_f[~mask2]
model_f=model_f.reset_index().drop('index',axis=1)
y=model_f['y']
x=model_f.drop('y',axis=1)
x=x[['TRADE_COUNTRYCD','KR_2017','SNDIST','TRADE_HSCD_COUNTRYCD','IC_BUS_EASE_DFRN_DB','KR_HSCD_COUNTRY_RATIO','KR_RATIO','GDP_RATIO','PA_NUS_FCRF','TARIFF_AVG','TRADE_HSCD_RATIO','SP_POP_TOTL','RDIST','KR_COUNTRY_RATIO','KMDIST','TRADE_HSCD','KR_HSCD_RATIO','NY_GDP_MKTP_CD','TRADE_RATIO']]
name=x.columns
ss=StandardScaler()
ss.fit(x)
x=ss.transform(x)
x=pd.DataFrame(x)
x.columns=list(name)

x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=41)

final_xgbr_41=XGBRegressor(random_state=42)
final_xgbr_41.fit(x_train,y_train)
final_xgbr_41.score(x_train,y_train)
final_xgbr_41.score(x_test,y_test)

model_compare=d10[['TRADE_COUNTRYCD','KR_2017','SNDIST','TRADE_HSCD_COUNTRYCD','IC_BUS_EASE_DFRN_DB','KR_HSCD_COUNTRY_RATIO','KR_RATIO','GDP_RATIO','PA_NUS_FCRF','TARIFF_AVG','TRADE_HSCD_RATIO','SP_POP_TOTL','RDIST','KR_COUNTRY_RATIO','KMDIST','TRADE_HSCD','KR_HSCD_RATIO','NY_GDP_MKTP_CD','TRADE_RATIO','y']]
y_c=model_compare['y']
x_c=model_compare.drop('y',axis=1)
x2_c=x_c[['TRADE_COUNTRYCD','KR_2017','SNDIST','TRADE_HSCD_COUNTRYCD','IC_BUS_EASE_DFRN_DB','KR_HSCD_COUNTRY_RATIO','KR_RATIO','GDP_RATIO','PA_NUS_FCRF','TARIFF_AVG','TRADE_HSCD_RATIO','SP_POP_TOTL','RDIST','KR_COUNTRY_RATIO','KMDIST','TRADE_HSCD','KR_HSCD_RATIO','NY_GDP_MKTP_CD','TRADE_RATIO']]
name=x2_c.columns
x_num_c=ss.transform(x2_c)
x_c=pd.DataFrame(x_num_c)
x_c.columns=name
x_c.info()

pred=final_xgbr_41.predict(x_c)

print("변경 후\n")
print("train",final_xgbr_41.score(x_train,y_train),"\ntest",final_xgbr_41.score(x_test,y_test))
print("all",final_xgbr_41.score(x_c,y_c))
print("all MAE",mean_absolute_error(np.expm1(y_c),np.expm1(pred)))

#변경 후

#train 0.9495107891964865 
#test 0.8683261853320233
#all 0.7893779340560207
#all MAE 5081393.457673153

## 한계가있음 ㅇㅇ. 아 좋은 방법없나.


data.SNDIST.hist()

tt2=data.loc[data.SNDIST.isnull()==True,'COUNTRYNM'].unique()

for i in tt2:
    data.loc[data.COUNTRYNM==i,'SNDIST'].hist(alpha=0.5)
    print(data.loc[data.COUNTRYNM==i,'SNDIST'].mean())
    print(data.loc[data.COUNTRYNM==i,'SNDIST'].median())
    
for i in tt2:
    data7.loc[(data7.SNDIST.isnull()==True)&(data7.COUNTRYNM==i),'SNDIST']=data7.loc[data7.COUNTRYNM==i,'SNDIST'].mean()

data["HSCD"] = data["HSCD"].map(lambda x: str(x))
hs_2 = [] 
for i in range(data.shape[0]):
   hs_2.append(data.iloc[i,1][:2])
data["HS"] = hs_2

data.info()

tt3=data.loc[data.TARIFF_AVG.isnull()==True,'HS'].unique()

data.TARIFF_AVG.hist()

data=TARIFF_AVG_missing(data,2017)
tt3
tt4
tt3=data.loc[data.TARIFF_AVG.isnull()==True,'HS'].unique()
tt4=data.loc[data.TARIFF_AVG.isnull()==True,'COUNTRYNM'].unique()
i=tt3[0]
for i in tt3:
    for j in tt4:
        data.loc[(data.HS==i)&(data.COUNTRYNM==j),'TARIFF_AVG'].hist(alpha=0.5)
        data.loc[(data.HS==i)&(data.COUNTRYNM==j),'TARIFF_AVG'].mean()
        data.loc[(data.HS==i)&(data.COUNTRYNM==j),'TARIFF_AVG'].median()
    plt.show()
    plt.close()