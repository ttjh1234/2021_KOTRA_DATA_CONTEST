# -*- coding: utf-8 -*-
"""
Created on Tue Jul 13 20:50:33 2021

@author: 82104
"""

import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
import sklearn
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import GridSearchCV
from scipy.stats import uniform, randint
from sklearn.metrics import mean_squared_error, mean_absolute_error
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor

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
    df["RDIST"]=df['KMDIST']/df['SNDIST']
    
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
data_compare.info()
data.describe()

data
data_compare=data_compare.drop(['2017_KR_IMPORT','2017_KR_EXPORT'],axis=1)
data_compare=data_compare.drop('Have_zero',axis=1)

(data['TARIFF_AVG']-data_compare['TARIFF_AVG']).sum()
(data['SNDIST']-data_compare['SNDIST']).sum()
(data['PA_NUS_FCRF']-data_compare['PA_NUS_FCRF']).sum()
(data['RDIST']-data_compare['RDIST']).sum()


mask = ((data['KR_TRADE_HSCD_COUNTRYCD']==0)&(data['KR_2017']!=0))|((data['KR_TRADE_HSCD_COUNTRYCD']!=0)&(data['KR_2017']==0))

from sklearn.linear_model import LinearRegression
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures

from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import GridSearchCV
from scipy.stats import uniform, randint

data.info()
data=data.reset_index().drop('index',axis=1)

data2=data[['COUNTRYNM','TRADE_COUNTRYCD','TRADE_HSCD','TARIFF_AVG','SNDIST','NY_GDP_MKTP_CD','SP_POP_TOTL','PA_NUS_FCRF','IC_BUS_EASE_DFRN_DB','TRADE_HSCD_COUNTRYCD','IncomeGroup','HS','group']]
data2=pd.get_dummies(data2,drop_first=True)
data2.info()
pred=data2[data2.TRADE_HSCD_COUNTRYCD.isnull()==True]
play=data2[data2.TRADE_HSCD_COUNTRYCD.isnull()==False]
y_pred=pred['TRADE_HSCD_COUNTRYCD']
y=play['TRADE_HSCD_COUNTRYCD']

play=play.drop('TRADE_HSCD_COUNTRYCD',axis=1)
pred=pred.drop('TRADE_HSCD_COUNTRYCD',axis=1)

y=np.log1p(y)

y_pred=np.log1p(y_pred)
play.info()
play=play.reset_index().drop('index',axis=1)

play.info()
x_num=play.loc[:,:'IC_BUS_EASE_DFRN_DB']
x_num.info()
x_cat=play.loc[:,'COUNTRYNM_Australia':]

pred=pred.reset_index().drop('index',axis=1)

x_num_p=pred.loc[:,:'IC_BUS_EASE_DFRN_DB']
x_cat_p=pred.loc[:,'COUNTRYNM_Australia':]

x_num=np.log1p(x_num)
x_num_p=np.log1p(x_num_p)
ss=StandardScaler()
ss.fit(x_num)
x_num2=ss.transform(x_num)
x_cat2=np.array(x_cat)

x_num_p2=ss.transform(x_num_p)
x_cat_p2=np.array(x_cat_p)

x=np.concatenate([x_num2,x_cat2],axis=1)
x_pred=np.concatenate([x_num_p2,x_cat_p2],axis=1)

x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=42)

from xgboost import XGBRegressor

xgb=XGBRegressor(random_state=42)
xgb.fit(x_train,y_train)
xgb.score(x_train,y_train)
xgb.score(x_test,y_test)
y_pv=xgb.predict(x_pred)
y_pv=np.exp(y_pv)-1
y_pv.shape
pred['pred']=y_pv

pred_value=list(pred['pred'])

data.loc[data.TRADE_HSCD_COUNTRYCD.isnull()==True,'TRADE_HSCD_COUNTRYCD']=pred_value

data.info()

feature_engineering(data)

model_data=data.drop(['UNC_YEAR','COUNTRYCD'],axis=1)
model_data.HSCD=model_data.HSCD.astype('object')
model_data=model_data.drop(['2017_KR_IMPORT','2017_KR_EXPORT'],axis=1)

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
d10.head()

mask=((d10['y']==0)&(d10['KR_2017']!=0))|((d10['y']!=0)&(d10['KR_2017']==0))
d100=d10[~mask]

model_c=d100
model_c=model_c.reset_index().drop('index',axis=1)
model_c.head()
model_c.info()
y=model_c['y']
x=model_c.drop('y',axis=1)
x2=x[['TRADE_COUNTRYCD','SNDIST','TRADE_HSCD_COUNTRYCD','IC_BUS_EASE_DFRN_DB','KR_HSCD_COUNTRY_RATIO','TRADE_HSCD_RATIO','SP_POP_TOTL','RDIST','TARIFF_AVG','PA_NUS_FCRF','GDP_RATIO','KMDIST','TRADE_RATIO','KR_HSCD_RATIO','KR_COUNTRY_RATIO','NY_GDP_MKTP_CD']]
x2.info()

y.head()

x2.head()
name=x2.columns
ss=StandardScaler()
ss.fit(x2)
x_num=ss.transform(x2)
x=pd.DataFrame(x_num)


x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=42)


xgbr2=XGBRegressor(random_state=42)
xgbr2.fit(x_train,y_train)

xgbr2.score(x_train,y_train)
xgbr2.score(x_test,y_test)

d10.info()
model_compare=d10[['TRADE_COUNTRYCD','SNDIST','TRADE_HSCD_COUNTRYCD','IC_BUS_EASE_DFRN_DB','KR_HSCD_COUNTRY_RATIO','TRADE_HSCD_RATIO','SP_POP_TOTL','RDIST','TARIFF_AVG','PA_NUS_FCRF','GDP_RATIO','KMDIST','TRADE_RATIO','KR_HSCD_RATIO','KR_COUNTRY_RATIO','NY_GDP_MKTP_CD','y']]
y_c=model_compare['y']
x_c=model_compare.drop('y',axis=1)
name=x_c.columns
ss=StandardScaler()
ss.fit(x_c)
x_num_c=ss.transform(x_c)
x_c=pd.DataFrame(x_num_c)
x_c.columns=list(name)
    

y_c
pred=xgbr2.predict(x_c)

np.sqrt(((np.expm1(y_c)-np.expm1(pred))**2).sum())

model_c=d100
model_c=model_c.reset_index().drop('index',axis=1)
y=model_c['y']
x=model_c.drop('y',axis=1)
x2=x[['TRADE_COUNTRYCD','SNDIST','KR_RATIO','TRADE_HSCD_COUNTRYCD','IC_BUS_EASE_DFRN_DB','KR_HSCD_COUNTRY_RATIO','TRADE_HSCD_RATIO','RDIST','TARIFF_AVG','PA_NUS_FCRF','GDP_RATIO','KMDIST','KR_HSCD_RATIO','TRADE_HSCD','KR_COUNTRY_RATIO','NY_GDP_MKTP_CD']]

name=x2.columns
ss=StandardScaler()
ss.fit(x2)
x_num=ss.transform(x2)
x=pd.DataFrame(x_num)
x.columns=list(name)

x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=42)


xgbr2=XGBRegressor(random_state=42)
xgbr2.fit(x_train,y_train)

xgbr2.score(x_train,y_train)
xgbr2.score(x_test,y_test)
data.info()
d100.info()



for i in ['TRADE_COUNTRYCD','KR_2017','SP_POP_TOTL','KR_RATIO','SNDIST','TRADE_HSCD_COUNTRYCD','IC_BUS_EASE_DFRN_DB','KR_HSCD_COUNTRY_RATIO','TRADE_HSCD_RATIO','RDIST','TARIFF_AVG','PA_NUS_FCRF','GDP_RATIO','KMDIST','KR_HSCD_RATIO','TRADE_HSCD','KR_COUNTRY_RATIO','NY_GDP_MKTP_CD']:
    model_c=d100
    model_c=model_c.reset_index().drop('index',axis=1)
    y=model_c['y']
    x=model_c.drop('y',axis=1)
    x2=x[['TRADE_COUNTRYCD','KR_2017','SP_POP_TOTL','KR_RATIO','SNDIST','TRADE_HSCD_COUNTRYCD','IC_BUS_EASE_DFRN_DB','KR_HSCD_COUNTRY_RATIO','TRADE_HSCD_RATIO','RDIST','TARIFF_AVG','PA_NUS_FCRF','GDP_RATIO','KMDIST','KR_HSCD_RATIO','TRADE_HSCD','KR_COUNTRY_RATIO','NY_GDP_MKTP_CD']]
    x2=x2.drop(i,axis=1)
    name=x2.columns
    ss=StandardScaler()
    ss.fit(x2)
    x_num=ss.transform(x2)
    x=pd.DataFrame(x_num)
    mean=[]
    for j in range(101,200):
        x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=j)


        xgbr2=XGBRegressor(random_state=42)
        xgbr2.fit(x_train,y_train)

        model_compare=d10[['TRADE_COUNTRYCD','KR_2017','SP_POP_TOTL','KR_RATIO','SNDIST','TRADE_HSCD_COUNTRYCD','IC_BUS_EASE_DFRN_DB','KR_HSCD_COUNTRY_RATIO','TRADE_HSCD_RATIO','RDIST','TARIFF_AVG','PA_NUS_FCRF','GDP_RATIO','KMDIST','KR_HSCD_RATIO','TRADE_HSCD','KR_COUNTRY_RATIO','NY_GDP_MKTP_CD','y']]
        y_c=model_compare['y']
        x_c=model_compare.drop('y',axis=1)
        x2_c=x_c[['TRADE_COUNTRYCD','KR_2017','SP_POP_TOTL','KR_RATIO','SNDIST','TRADE_HSCD_COUNTRYCD','IC_BUS_EASE_DFRN_DB','KR_HSCD_COUNTRY_RATIO','TRADE_HSCD_RATIO','RDIST','TARIFF_AVG','PA_NUS_FCRF','GDP_RATIO','KMDIST','KR_HSCD_RATIO','TRADE_HSCD','KR_COUNTRY_RATIO','NY_GDP_MKTP_CD']]
        x2_c=x2_c.drop(i,axis=1)
        name2=x2.columns
        x_num_c=ss.transform(x2_c)
        x_c=pd.DataFrame(x_num_c)
    
        pred=xgbr2.predict(x_c)
        mean.append(np.sqrt(((np.expm1(y_c)-np.expm1(pred))**2).sum())/100000000)
    print(i,np.mean(mean))

# 100개 -> KR 2017 215


## 전체 모형 점수 
model_c=d100
model_c=model_c.reset_index().drop('index',axis=1)
y=model_c['y']
x=model_c.drop('y',axis=1)
x2=x[['TRADE_COUNTRYCD','KR_2017','SP_POP_TOTL','KR_RATIO','SNDIST','TRADE_HSCD_COUNTRYCD','IC_BUS_EASE_DFRN_DB','KR_HSCD_COUNTRY_RATIO','TRADE_HSCD_RATIO','RDIST','TARIFF_AVG','PA_NUS_FCRF','GDP_RATIO','KMDIST','KR_HSCD_RATIO','TRADE_HSCD','KR_COUNTRY_RATIO','NY_GDP_MKTP_CD']]
x2=x2.drop(i,axis=1)
name=x2.columns
ss=StandardScaler()
ss.fit(x2)
x_num=ss.transform(x2)
x=pd.DataFrame(x_num)
mean=[]
for j in range(101,200):
    x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=j)


    xgbr2=XGBRegressor(random_state=42)
    xgbr2.fit(x_train,y_train)

    model_compare=d10[['TRADE_COUNTRYCD','KR_2017','SP_POP_TOTL','KR_RATIO','SNDIST','TRADE_HSCD_COUNTRYCD','IC_BUS_EASE_DFRN_DB','KR_HSCD_COUNTRY_RATIO','TRADE_HSCD_RATIO','RDIST','TARIFF_AVG','PA_NUS_FCRF','GDP_RATIO','KMDIST','KR_HSCD_RATIO','TRADE_HSCD','KR_COUNTRY_RATIO','NY_GDP_MKTP_CD','y']]
    y_c=model_compare['y']
    x_c=model_compare.drop('y',axis=1)
    x2_c=x_c[['TRADE_COUNTRYCD','KR_2017','SP_POP_TOTL','KR_RATIO','SNDIST','TRADE_HSCD_COUNTRYCD','IC_BUS_EASE_DFRN_DB','KR_HSCD_COUNTRY_RATIO','TRADE_HSCD_RATIO','RDIST','TARIFF_AVG','PA_NUS_FCRF','GDP_RATIO','KMDIST','KR_HSCD_RATIO','TRADE_HSCD','KR_COUNTRY_RATIO','NY_GDP_MKTP_CD']]
    x2_c=x2_c.drop(i,axis=1)
    name2=x2.columns
    x_num_c=ss.transform(x2_c)
    x_c=pd.DataFrame(x_num_c)
    
    pred=xgbr2.predict(x_c)
    mean.append(np.sqrt(((np.expm1(y_c)-np.expm1(pred))**2).sum())/100000000)
print(np.mean(mean))

# 316 100개
## 322.95767129 42~46
## 257.2203877 714~718

## 200 KR_2017 삭제 후 2번째 반복 

## 215 KR_2017 (100개) 삭제후 반복
## 137 KR_2017
for i in ['TRADE_COUNTRYCD','SP_POP_TOTL','KR_RATIO','SNDIST','TRADE_HSCD_COUNTRYCD','IC_BUS_EASE_DFRN_DB','KR_HSCD_COUNTRY_RATIO','TRADE_HSCD_RATIO','RDIST','TARIFF_AVG','PA_NUS_FCRF','GDP_RATIO','KMDIST','KR_HSCD_RATIO','TRADE_HSCD','KR_COUNTRY_RATIO','NY_GDP_MKTP_CD']:
    model_c=d100
    model_c=model_c.reset_index().drop('index',axis=1)
    y=model_c['y']
    x=model_c.drop('y',axis=1)
    x2=x[['TRADE_COUNTRYCD','SP_POP_TOTL','KR_RATIO','SNDIST','TRADE_HSCD_COUNTRYCD','IC_BUS_EASE_DFRN_DB','KR_HSCD_COUNTRY_RATIO','TRADE_HSCD_RATIO','RDIST','TARIFF_AVG','PA_NUS_FCRF','GDP_RATIO','KMDIST','KR_HSCD_RATIO','TRADE_HSCD','KR_COUNTRY_RATIO','NY_GDP_MKTP_CD']]
    x2=x2.drop(i,axis=1)
    name=x2.columns
    ss=StandardScaler()
    ss.fit(x2)
    x_num=ss.transform(x2)
    x=pd.DataFrame(x_num)
    mean=[]
    for j in range(101,200):
        x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=j)


        xgbr2=XGBRegressor(random_state=42)
        xgbr2.fit(x_train,y_train)

        model_compare=d10[['TRADE_COUNTRYCD','SP_POP_TOTL','KR_RATIO','SNDIST','TRADE_HSCD_COUNTRYCD','IC_BUS_EASE_DFRN_DB','KR_HSCD_COUNTRY_RATIO','TRADE_HSCD_RATIO','RDIST','TARIFF_AVG','PA_NUS_FCRF','GDP_RATIO','KMDIST','KR_HSCD_RATIO','TRADE_HSCD','KR_COUNTRY_RATIO','NY_GDP_MKTP_CD','y']]
        y_c=model_compare['y']
        x_c=model_compare.drop('y',axis=1)
        x2_c=x_c[['TRADE_COUNTRYCD','SP_POP_TOTL','KR_RATIO','SNDIST','TRADE_HSCD_COUNTRYCD','IC_BUS_EASE_DFRN_DB','KR_HSCD_COUNTRY_RATIO','TRADE_HSCD_RATIO','RDIST','TARIFF_AVG','PA_NUS_FCRF','GDP_RATIO','KMDIST','KR_HSCD_RATIO','TRADE_HSCD','KR_COUNTRY_RATIO','NY_GDP_MKTP_CD']]
        x2_c=x2_c.drop(i,axis=1)
        name2=x2.columns
        
        x_num_c=ss.transform(x2_c)
        x_c=pd.DataFrame(x_num_c)
    
        pred=xgbr2.predict(x_c)
        mean.append(np.sqrt(((np.expm1(y_c)-np.expm1(pred))**2).sum())/100000000)
    print(i,np.mean(mean))


# SNDIST 삭제 204 
for i in ['TRADE_COUNTRYCD','SP_POP_TOTL','KR_RATIO','TRADE_HSCD_COUNTRYCD','IC_BUS_EASE_DFRN_DB','KR_HSCD_COUNTRY_RATIO','TRADE_HSCD_RATIO','RDIST','TARIFF_AVG','PA_NUS_FCRF','GDP_RATIO','KMDIST','KR_HSCD_RATIO','TRADE_HSCD','KR_COUNTRY_RATIO','NY_GDP_MKTP_CD']:
    model_c=d100
    model_c=model_c.reset_index().drop('index',axis=1)
    y=model_c['y']
    x=model_c.drop('y',axis=1)
    x2=x[['TRADE_COUNTRYCD','SP_POP_TOTL','KR_RATIO','TRADE_HSCD_COUNTRYCD','IC_BUS_EASE_DFRN_DB','KR_HSCD_COUNTRY_RATIO','TRADE_HSCD_RATIO','RDIST','TARIFF_AVG','PA_NUS_FCRF','GDP_RATIO','KMDIST','KR_HSCD_RATIO','TRADE_HSCD','KR_COUNTRY_RATIO','NY_GDP_MKTP_CD']]
    x2=x2.drop(i,axis=1)
    name=x2.columns
    ss=StandardScaler()
    ss.fit(x2)
    x_num=ss.transform(x2)
    x=pd.DataFrame(x_num)
    mean=[]
    for j in range(101,200):
        x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=j)


        xgbr2=XGBRegressor(random_state=42)
        xgbr2.fit(x_train,y_train)

        model_compare=d10[['TRADE_COUNTRYCD','SP_POP_TOTL','KR_RATIO','TRADE_HSCD_COUNTRYCD','IC_BUS_EASE_DFRN_DB','KR_HSCD_COUNTRY_RATIO','TRADE_HSCD_RATIO','RDIST','TARIFF_AVG','PA_NUS_FCRF','GDP_RATIO','KMDIST','KR_HSCD_RATIO','TRADE_HSCD','KR_COUNTRY_RATIO','NY_GDP_MKTP_CD','y']]
        y_c=model_compare['y']
        x_c=model_compare.drop('y',axis=1)
        x2_c=x_c[['TRADE_COUNTRYCD','SP_POP_TOTL','KR_RATIO','TRADE_HSCD_COUNTRYCD','IC_BUS_EASE_DFRN_DB','KR_HSCD_COUNTRY_RATIO','TRADE_HSCD_RATIO','RDIST','TARIFF_AVG','PA_NUS_FCRF','GDP_RATIO','KMDIST','KR_HSCD_RATIO','TRADE_HSCD','KR_COUNTRY_RATIO','NY_GDP_MKTP_CD']]
        x2_c=x2_c.drop(i,axis=1)
        name2=x2.columns
        
        x_num_c=ss.transform(x2_c)
        x_c=pd.DataFrame(x_num_c)
    
        pred=xgbr2.predict(x_c)
        mean.append(np.sqrt(((np.expm1(y_c)-np.expm1(pred))**2).sum())/100000000)
    print(i,np.mean(mean))



xgb_params={'learning_rate':uniform(0.001,0.5),
            'n_estimator':randint(),
            'max_depth':randint(),
            'gamma':uniform(0,3),
            'min_child_weight':randint(1,3),
            'subsample':uniform(0.5,1)}



rs=RandomizedSearchCV(xgb,xgb_params,n_iter=50,cv=5,scoring='neg_mean_squared_error',verbose=2,random_state=42)
rs.fit(x_train,y_train)
rs.best_params_
est=rs.best_estimator_

print(est.score(x_train,y_train))
print(est.score(x_test,y_test))



gs=GridSearchCV(xgb,xgb_params,cv=5,scoring='neg_mean_squared_error',verbose=2,return_train_score=True)
gs.fit(x_train,y_train)

gs.best_params_

est=gs.best_estimator_

cvres=gs.cv_results_

for mean_score, params in zip(cvres["mean_test_score"],cvres["params"]):
    print(np.sqrt(-mean_score),params)














