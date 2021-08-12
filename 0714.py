# -*- coding: utf-8 -*-
"""
Created on Wed Jul 14 16:55:43 2021

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

feature = ['TRADE_COUNTRYCD', 'TRADE_HSCD', 'TARIFF_AVG', 'SNDIST',
       'NY_GDP_MKTP_CD', 'SP_POP_TOTL', 'PA_NUS_FCRF', 'IC_BUS_EASE_DFRN_DB',
       'KMDIST', 'RDIST', 'KR_2017', 'HSCD_RATIO', 'COUNTRY_RATIO',"GDP_RATIO","TRADE_HSCD_COUNTRYCD","KR_TRADE_HSCD_COUNTRYCD"]
new = data[feature]

new.info()
new.head()

mask = ((new['KR_TRADE_HSCD_COUNTRYCD']==0)&(new['KR_2017']!=0))|((new['KR_TRADE_HSCD_COUNTRYCD']!=0)&(new['KR_2017']==0)) 

new=data[['TRADE_COUNTRYCD', 'TRADE_HSCD', 'TARIFF_AVG', 'SNDIST','NY_GDP_MKTP_CD', 'SP_POP_TOTL', 'PA_NUS_FCRF', 'IC_BUS_EASE_DFRN_DB','KMDIST', 'KR_2017', 'GDP_RATIO', 'RDIST', 'HSCD_RATIO','COUNTRY_RATIO','TRADE_HSCD_COUNTRYCD','KR_TRADE_HSCD_COUNTRYCD']]

mask = ((new['KR_TRADE_HSCD_COUNTRYCD']==0)&(new['KR_2017']!=0))|((new['KR_TRADE_HSCD_COUNTRYCD']!=0)&(new['KR_2017']==0)) 
## 전체 모형의 잔차 seed 46 시작 -- 1120.3981 
model_c=new[~mask]
model_c=model_c[model_c['TRADE_HSCD_COUNTRYCD'].isnull()==False]
model_c=model_c.reset_index().drop('index',axis=1)
y=model_c["TRADE_HSCD_COUNTRYCD"]
x=model_c.drop("TRADE_HSCD_COUNTRYCD",axis=1)
x2=x[['TRADE_COUNTRYCD', 'TRADE_HSCD', 'TARIFF_AVG', 'SNDIST','NY_GDP_MKTP_CD', 'SP_POP_TOTL', 'PA_NUS_FCRF', 'IC_BUS_EASE_DFRN_DB','KMDIST', 'KR_2017', 'GDP_RATIO', 'RDIST', 'HSCD_RATIO','COUNTRY_RATIO']]
name=x2.columns
x2=np.log1p(x2)
y=np.log1p(y)
ss=StandardScaler()
ss.fit(x2)
x_num=ss.transform(x2)
x=pd.DataFrame(x_num)
x.columns=list(name)
    
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=41)


xgbr2=XGBRegressor(random_state=42)
xgbr2.fit(x_train,y_train)

model_compare=new[['TRADE_COUNTRYCD', 'TRADE_HSCD', 'TARIFF_AVG', 'SNDIST','NY_GDP_MKTP_CD', 'SP_POP_TOTL', 'PA_NUS_FCRF', 'IC_BUS_EASE_DFRN_DB','KMDIST', 'KR_2017', 'GDP_RATIO', 'RDIST', 'HSCD_RATIO','COUNTRY_RATIO','TRADE_HSCD_COUNTRYCD']]
model_compare=model_compare[model_compare['TRADE_HSCD_COUNTRYCD'].isnull()==False]
model_compare=model_compare.reset_index().drop('index',axis=1)
y_c=model_compare["TRADE_HSCD_COUNTRYCD"]
x_c=model_compare.drop("TRADE_HSCD_COUNTRYCD",axis=1)
x2_c=x_c[['TRADE_COUNTRYCD', 'TRADE_HSCD', 'TARIFF_AVG', 'SNDIST','NY_GDP_MKTP_CD', 'SP_POP_TOTL', 'PA_NUS_FCRF', 'IC_BUS_EASE_DFRN_DB','KMDIST', 'KR_2017', 'GDP_RATIO', 'RDIST', 'HSCD_RATIO','COUNTRY_RATIO']]
name=x2.columns
x2_c=np.log1p(x2_c)
y_c=np.log1p(y_c)
x_num_c=ss.transform(x2_c)
x_c=pd.DataFrame(x_num_c)
x_c.columns=list(name)
    
pred=xgbr2.predict(x_c)
np.sqrt(((np.expm1(y_c)-np.expm1(pred))**2).sum())/100000000

# 첫 번째 반복 seed 46 
for i in ['TRADE_COUNTRYCD', 'TRADE_HSCD', 'TARIFF_AVG', 'SNDIST','NY_GDP_MKTP_CD', 'SP_POP_TOTL', 'PA_NUS_FCRF', 'IC_BUS_EASE_DFRN_DB','KMDIST', 'KR_2017', 'GDP_RATIO', 'RDIST', 'HSCD_RATIO','COUNTRY_RATIO']:
    model_c=new[~mask]
    model_c=model_c[model_c['TRADE_HSCD_COUNTRYCD'].isnull()==False]
    model_c=model_c.reset_index().drop('index',axis=1)
    y=model_c["TRADE_HSCD_COUNTRYCD"]
    x=model_c.drop("TRADE_HSCD_COUNTRYCD",axis=1)
    x2=x[['TRADE_COUNTRYCD', 'TRADE_HSCD', 'TARIFF_AVG', 'SNDIST','NY_GDP_MKTP_CD', 'SP_POP_TOTL', 'PA_NUS_FCRF', 'IC_BUS_EASE_DFRN_DB','KMDIST', 'KR_2017', 'GDP_RATIO', 'RDIST', 'HSCD_RATIO','COUNTRY_RATIO']]
    x2=x2.drop(i,axis=1)
    name=x2.columns
    x2=np.log1p(x2)
    y=np.log1p(y)
    ss=StandardScaler()
    ss.fit(x2)
    x_num=ss.transform(x2)
    x=pd.DataFrame(x_num)
    x.columns=list(name)
    
    x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=41)


    xgbr2=XGBRegressor(random_state=42)
    xgbr2.fit(x_train,y_train)

    model_compare=new[['TRADE_COUNTRYCD', 'TRADE_HSCD', 'TARIFF_AVG', 'SNDIST','NY_GDP_MKTP_CD', 'SP_POP_TOTL', 'PA_NUS_FCRF', 'IC_BUS_EASE_DFRN_DB','KMDIST', 'KR_2017', 'GDP_RATIO', 'RDIST', 'HSCD_RATIO','COUNTRY_RATIO','TRADE_HSCD_COUNTRYCD']]
    model_compare=model_compare[model_compare['TRADE_HSCD_COUNTRYCD'].isnull()==False]
    model_compare=model_compare.reset_index().drop('index',axis=1)
    y_c=model_compare["TRADE_HSCD_COUNTRYCD"]
    x_c=model_compare.drop("TRADE_HSCD_COUNTRYCD",axis=1)
    x2_c=x_c[['TRADE_COUNTRYCD', 'TRADE_HSCD', 'TARIFF_AVG', 'SNDIST','NY_GDP_MKTP_CD', 'SP_POP_TOTL', 'PA_NUS_FCRF', 'IC_BUS_EASE_DFRN_DB','KMDIST', 'KR_2017', 'GDP_RATIO', 'RDIST', 'HSCD_RATIO','COUNTRY_RATIO']]
    x2_c=x2_c.drop(i,axis=1)
    name=x2.columns
    x2_c=np.log1p(x2_c)
    y_c=np.log1p(y_c)
    x_num_c=ss.transform(x2_c)
    x_c=pd.DataFrame(x_num_c)
    x_c.columns=list(name)
    
    pred=xgbr2.predict(x_c)
    print(i,np.sqrt(((np.expm1(y_c)-np.expm1(pred))**2).sum())/100000000)


# 결과 : IC_BUS_EASE_DFRN_DB 952.3637984516174 제거 
# 두 번째 반복
for i in ['TRADE_COUNTRYCD', 'TRADE_HSCD', 'TARIFF_AVG','SNDIST','NY_GDP_MKTP_CD',"KR_2017",'SP_POP_TOTL','PA_NUS_FCRF','KMDIST', 'RDIST', 'HSCD_RATIO', 'COUNTRY_RATIO',"GDP_RATIO"]:
    model_c=new[~mask]
    model_c=model_c[model_c['TRADE_HSCD_COUNTRYCD'].isnull()==False]
    model_c=model_c.reset_index().drop('index',axis=1)
    y=model_c["TRADE_HSCD_COUNTRYCD"]
    x=model_c.drop("TRADE_HSCD_COUNTRYCD",axis=1)
    x2=x[['TRADE_COUNTRYCD', 'TRADE_HSCD', 'TARIFF_AVG','SNDIST','NY_GDP_MKTP_CD',"KR_2017",'SP_POP_TOTL','PA_NUS_FCRF','KMDIST', 'RDIST', 'HSCD_RATIO', 'COUNTRY_RATIO',"GDP_RATIO"]]
    x2=x2.drop(i,axis=1)
    name=x2.columns
    x2=np.log1p(x2)
    y=np.log1p(y)
    ss=StandardScaler()
    ss.fit(x2)
    x_num=ss.transform(x2)
    x=pd.DataFrame(x_num)
    x.columns=list(name)
    
    x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=46)


    xgbr2=XGBRegressor(random_state=42)
    xgbr2.fit(x_train,y_train)

    model_compare=new[['TRADE_COUNTRYCD', 'TRADE_HSCD', 'TARIFF_AVG','SNDIST','NY_GDP_MKTP_CD',"KR_2017",'SP_POP_TOTL','PA_NUS_FCRF','KMDIST', 'RDIST', 'HSCD_RATIO', 'COUNTRY_RATIO',"GDP_RATIO","TRADE_HSCD_COUNTRYCD"]]
    model_compare=model_compare[model_compare['TRADE_HSCD_COUNTRYCD'].isnull()==False]
    model_compare=model_compare.reset_index().drop('index',axis=1)
    y_c=model_compare["TRADE_HSCD_COUNTRYCD"]
    x_c=model_compare.drop("TRADE_HSCD_COUNTRYCD",axis=1)
    x2_c=x_c[['TRADE_COUNTRYCD', 'TRADE_HSCD', 'TARIFF_AVG','SNDIST','NY_GDP_MKTP_CD',"KR_2017",'SP_POP_TOTL','PA_NUS_FCRF','KMDIST', 'RDIST', 'HSCD_RATIO', 'COUNTRY_RATIO',"GDP_RATIO"]]
    x2_c=x2_c.drop(i,axis=1)
    name=x2.columns
    x2_c=np.log1p(x2_c)
    y_c=np.log1p(y_c)
    x_num_c=ss.transform(x2_c)
    x_c=pd.DataFrame(x_num_c)
    x_c.columns=list(name)
    
    pred=xgbr2.predict(x_c)
    print(i,np.sqrt(((np.expm1(y_c)-np.expm1(pred))**2).sum())/100000000)
    
# 결과 : SNDIST 919.0138125417852 제거
# 세 번째 반복
for i in ['TRADE_COUNTRYCD', 'TRADE_HSCD', 'TARIFF_AVG','NY_GDP_MKTP_CD',"KR_2017",'SP_POP_TOTL','PA_NUS_FCRF','KMDIST', 'RDIST', 'HSCD_RATIO', 'COUNTRY_RATIO',"GDP_RATIO"]:
    model_c=new[~mask]
    model_c=model_c[model_c['TRADE_HSCD_COUNTRYCD'].isnull()==False]
    model_c=model_c.reset_index().drop('index',axis=1)
    y=model_c["TRADE_HSCD_COUNTRYCD"]
    x=model_c.drop("TRADE_HSCD_COUNTRYCD",axis=1)
    x2=x[['TRADE_COUNTRYCD', 'TRADE_HSCD', 'TARIFF_AVG','NY_GDP_MKTP_CD',"KR_2017",'SP_POP_TOTL','PA_NUS_FCRF','KMDIST', 'RDIST', 'HSCD_RATIO', 'COUNTRY_RATIO',"GDP_RATIO"]]
    x2=x2.drop(i,axis=1)
    name=x2.columns
    x2=np.log1p(x2)
    y=np.log1p(y)
    ss=StandardScaler()
    ss.fit(x2)
    x_num=ss.transform(x2)
    x=pd.DataFrame(x_num)
    x.columns=list(name)
    
    x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=46)


    xgbr2=XGBRegressor(random_state=42)
    xgbr2.fit(x_train,y_train)

    model_compare=new[['TRADE_COUNTRYCD', 'TRADE_HSCD', 'TARIFF_AVG','NY_GDP_MKTP_CD',"KR_2017",'SP_POP_TOTL','PA_NUS_FCRF','KMDIST', 'RDIST', 'HSCD_RATIO', 'COUNTRY_RATIO',"GDP_RATIO","TRADE_HSCD_COUNTRYCD"]]
    model_compare=model_compare[model_compare['TRADE_HSCD_COUNTRYCD'].isnull()==False]
    model_compare=model_compare.reset_index().drop('index',axis=1)
    y_c=model_compare["TRADE_HSCD_COUNTRYCD"]
    x_c=model_compare.drop("TRADE_HSCD_COUNTRYCD",axis=1)
    x2_c=x_c[['TRADE_COUNTRYCD', 'TRADE_HSCD', 'TARIFF_AVG','NY_GDP_MKTP_CD',"KR_2017",'SP_POP_TOTL','PA_NUS_FCRF','KMDIST', 'RDIST', 'HSCD_RATIO', 'COUNTRY_RATIO',"GDP_RATIO"]]
    x2_c=x2_c.drop(i,axis=1)
    name=x2.columns
    x2_c=np.log1p(x2_c)
    y_c=np.log1p(y_c)
    x_num_c=ss.transform(x2_c)
    x_c=pd.DataFrame(x_num_c)
    x_c.columns=list(name)
    
    pred=xgbr2.predict(x_c)
    print(i,np.sqrt(((np.expm1(y_c)-np.expm1(pred))**2).sum())/100000000)

# 결과 : TRADE_COUNTRYCD 914.733371018035 제거 
# 4 번째반복
for i in [ 'TRADE_HSCD', 'TARIFF_AVG','NY_GDP_MKTP_CD',"KR_2017",'SP_POP_TOTL','PA_NUS_FCRF','KMDIST', 'RDIST', 'HSCD_RATIO', 'COUNTRY_RATIO',"GDP_RATIO"]:
    model_c=new[~mask]
    model_c=model_c[model_c['TRADE_HSCD_COUNTRYCD'].isnull()==False]
    model_c=model_c.reset_index().drop('index',axis=1)
    y=model_c["TRADE_HSCD_COUNTRYCD"]
    x=model_c.drop("TRADE_HSCD_COUNTRYCD",axis=1)
    x2=x[['TRADE_HSCD', 'TARIFF_AVG','NY_GDP_MKTP_CD',"KR_2017",'SP_POP_TOTL','PA_NUS_FCRF','KMDIST', 'RDIST', 'HSCD_RATIO', 'COUNTRY_RATIO',"GDP_RATIO"]]
    x2=x2.drop(i,axis=1)
    name=x2.columns
    x2=np.log1p(x2)
    y=np.log1p(y)
    ss=StandardScaler()
    ss.fit(x2)
    x_num=ss.transform(x2)
    x=pd.DataFrame(x_num)
    x.columns=list(name)
    
    x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=46)


    xgbr2=XGBRegressor(random_state=42)
    xgbr2.fit(x_train,y_train)

    model_compare=new[['TRADE_HSCD', 'TARIFF_AVG','NY_GDP_MKTP_CD',"KR_2017",'SP_POP_TOTL','PA_NUS_FCRF','KMDIST', 'RDIST', 'HSCD_RATIO', 'COUNTRY_RATIO',"GDP_RATIO","TRADE_HSCD_COUNTRYCD"]]
    model_compare=model_compare[model_compare['TRADE_HSCD_COUNTRYCD'].isnull()==False]
    model_compare=model_compare.reset_index().drop('index',axis=1)
    y_c=model_compare["TRADE_HSCD_COUNTRYCD"]
    x_c=model_compare.drop("TRADE_HSCD_COUNTRYCD",axis=1)
    x2_c=x_c[['TRADE_HSCD', 'TARIFF_AVG','NY_GDP_MKTP_CD',"KR_2017",'SP_POP_TOTL','PA_NUS_FCRF','KMDIST', 'RDIST', 'HSCD_RATIO', 'COUNTRY_RATIO',"GDP_RATIO"]]
    x2_c=x2_c.drop(i,axis=1)
    name=x2.columns
    x2_c=np.log1p(x2_c)
    y_c=np.log1p(y_c)
    x_num_c=ss.transform(x2_c)
    x_c=pd.DataFrame(x_num_c)
    x_c.columns=list(name)
    
    pred=xgbr2.predict(x_c)
    print(i,np.sqrt(((np.expm1(y_c)-np.expm1(pred))**2).sum())/100000000)
    
# 결과 : TRADE_HSCD 909.3505592790019 제거
# 5번째 반복
for i in [ 'TARIFF_AVG','NY_GDP_MKTP_CD',"KR_2017",'SP_POP_TOTL','PA_NUS_FCRF','KMDIST', 'RDIST', 'HSCD_RATIO', 'COUNTRY_RATIO',"GDP_RATIO"]:
    model_c=new[~mask]
    model_c=model_c[model_c['TRADE_HSCD_COUNTRYCD'].isnull()==False]
    model_c=model_c.reset_index().drop('index',axis=1)
    y=model_c["TRADE_HSCD_COUNTRYCD"]
    x=model_c.drop("TRADE_HSCD_COUNTRYCD",axis=1)
    x2=x[[ 'TARIFF_AVG','NY_GDP_MKTP_CD',"KR_2017",'SP_POP_TOTL','PA_NUS_FCRF','KMDIST', 'RDIST', 'HSCD_RATIO', 'COUNTRY_RATIO',"GDP_RATIO"]]
    x2=x2.drop(i,axis=1)
    name=x2.columns
    x2=np.log1p(x2)
    y=np.log1p(y)
    ss=StandardScaler()
    ss.fit(x2)
    x_num=ss.transform(x2)
    x=pd.DataFrame(x_num)
    x.columns=list(name)
    
    x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=46)


    xgbr2=XGBRegressor(random_state=42)
    xgbr2.fit(x_train,y_train)

    model_compare=new[['TARIFF_AVG','NY_GDP_MKTP_CD',"KR_2017",'SP_POP_TOTL','PA_NUS_FCRF','KMDIST', 'RDIST', 'HSCD_RATIO', 'COUNTRY_RATIO',"GDP_RATIO","TRADE_HSCD_COUNTRYCD"]]
    model_compare=model_compare[model_compare['TRADE_HSCD_COUNTRYCD'].isnull()==False]
    model_compare=model_compare.reset_index().drop('index',axis=1)
    y_c=model_compare["TRADE_HSCD_COUNTRYCD"]
    x_c=model_compare.drop("TRADE_HSCD_COUNTRYCD",axis=1)
    x2_c=x_c[['TARIFF_AVG','NY_GDP_MKTP_CD',"KR_2017",'SP_POP_TOTL','PA_NUS_FCRF','KMDIST', 'RDIST', 'HSCD_RATIO', 'COUNTRY_RATIO',"GDP_RATIO"]]
    x2_c=x2_c.drop(i,axis=1)
    name=x2.columns
    x2_c=np.log1p(x2_c)
    y_c=np.log1p(y_c)
    x_num_c=ss.transform(x2_c)
    x_c=pd.DataFrame(x_num_c)
    x_c.columns=list(name)
    x_c.describe()
    pred=xgbr2.predict(x_c)
    print(i,np.sqrt(((np.expm1(y_c)-np.expm1(pred))**2).sum())/100000000)

# 결과 : 다 큼 결과 -> IC_BUS_EASE_DFRN_DB, SNDIST, TRADE_COUNTRYCD,TRADE_HSCD 제거

########################################################################################
# 시드 47

## 전체 모형의 잔차 seed 47 시작 --  1257.2736659560292
model_c=new[~mask]
model_c=model_c[model_c['TRADE_HSCD_COUNTRYCD'].isnull()==False]
model_c=model_c.reset_index().drop('index',axis=1)
y=model_c["TRADE_HSCD_COUNTRYCD"]
x=model_c.drop("TRADE_HSCD_COUNTRYCD",axis=1)
x2=x[['TRADE_COUNTRYCD', 'TRADE_HSCD', 'TARIFF_AVG','SNDIST','NY_GDP_MKTP_CD',"KR_2017",'SP_POP_TOTL','PA_NUS_FCRF', 'IC_BUS_EASE_DFRN_DB','KMDIST', 'RDIST', 'HSCD_RATIO', 'COUNTRY_RATIO',"GDP_RATIO"]]
name=x2.columns
x2=np.log1p(x2)
y=np.log1p(y)
ss=StandardScaler()
ss.fit(x2)
x_num=ss.transform(x2)
x=pd.DataFrame(x_num)
x.columns=list(name)
    
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=47)


xgbr2=XGBRegressor(random_state=42)
xgbr2.fit(x_train,y_train)

model_compare=new[['TRADE_COUNTRYCD', 'TRADE_HSCD', 'TARIFF_AVG','SNDIST','NY_GDP_MKTP_CD',"KR_2017",'SP_POP_TOTL','PA_NUS_FCRF', 'IC_BUS_EASE_DFRN_DB','KMDIST', 'RDIST', 'HSCD_RATIO', 'COUNTRY_RATIO',"GDP_RATIO","TRADE_HSCD_COUNTRYCD"]]
model_compare=model_compare[model_compare['TRADE_HSCD_COUNTRYCD'].isnull()==False]
model_compare=model_compare.reset_index().drop('index',axis=1)
y_c=model_compare["TRADE_HSCD_COUNTRYCD"]
x_c=model_compare.drop("TRADE_HSCD_COUNTRYCD",axis=1)
x2_c=x_c[['TRADE_COUNTRYCD', 'TRADE_HSCD', 'TARIFF_AVG','SNDIST','NY_GDP_MKTP_CD',"KR_2017",'SP_POP_TOTL','PA_NUS_FCRF', 'IC_BUS_EASE_DFRN_DB','KMDIST', 'RDIST', 'HSCD_RATIO', 'COUNTRY_RATIO',"GDP_RATIO"]]
name=x2.columns
x2_c=np.log1p(x2_c)
y_c=np.log1p(y_c)
x_num_c=ss.transform(x2_c)
x_c=pd.DataFrame(x_num_c)
x_c.columns=list(name)
    
pred=xgbr2.predict(x_c)
np.sqrt(((np.expm1(y_c)-np.expm1(pred))**2).sum())/100000000


## 마스킹을 먼저하고 하면 안되는 이유
## 
# 첫 번째 반복 seed 47 
for i in ['TRADE_COUNTRYCD', 'TRADE_HSCD', 'TARIFF_AVG','SNDIST','NY_GDP_MKTP_CD',"KR_2017",'SP_POP_TOTL','PA_NUS_FCRF', 'IC_BUS_EASE_DFRN_DB','KMDIST', 'RDIST', 'HSCD_RATIO', 'COUNTRY_RATIO',"GDP_RATIO"]:
    model_c=new[~mask]
    model_c=model_c[model_c['TRADE_HSCD_COUNTRYCD'].isnull()==False]
    model_c=model_c.reset_index().drop('index',axis=1)
    y=model_c["TRADE_HSCD_COUNTRYCD"]
    x=model_c.drop("TRADE_HSCD_COUNTRYCD",axis=1)
    x2=x[['TRADE_COUNTRYCD', 'TRADE_HSCD', 'TARIFF_AVG','SNDIST','NY_GDP_MKTP_CD',"KR_2017",'SP_POP_TOTL','PA_NUS_FCRF', 'IC_BUS_EASE_DFRN_DB','KMDIST', 'RDIST', 'HSCD_RATIO', 'COUNTRY_RATIO',"GDP_RATIO"]]
    x2=x2.drop(i,axis=1)
    name=x2.columns
    x2=np.log1p(x2)
    y=np.log1p(y)
    ss=StandardScaler()
    ss.fit(x2)
    x_num=ss.transform(x2)
    x=pd.DataFrame(x_num)
    x.columns=list(name)
    
    x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=47)


    xgbr2=XGBRegressor(random_state=42)
    xgbr2.fit(x_train,y_train)

    model_compare=new[['TRADE_COUNTRYCD', 'TRADE_HSCD', 'TARIFF_AVG','SNDIST','NY_GDP_MKTP_CD',"KR_2017",'SP_POP_TOTL','PA_NUS_FCRF', 'IC_BUS_EASE_DFRN_DB','KMDIST', 'RDIST', 'HSCD_RATIO', 'COUNTRY_RATIO',"GDP_RATIO",'TRADE_HSCD_COUNTRYCD']]
    model_compare=model_compare[model_compare['TRADE_HSCD_COUNTRYCD'].isnull()==False]
    model_compare=model_compare.reset_index().drop('index',axis=1)
    y_c=model_compare["TRADE_HSCD_COUNTRYCD"]
    x_c=model_compare.drop("TRADE_HSCD_COUNTRYCD",axis=1)
    x2_c=x_c[['TRADE_COUNTRYCD', 'TRADE_HSCD', 'TARIFF_AVG','SNDIST','NY_GDP_MKTP_CD',"KR_2017",'SP_POP_TOTL','PA_NUS_FCRF', 'IC_BUS_EASE_DFRN_DB','KMDIST', 'RDIST', 'HSCD_RATIO', 'COUNTRY_RATIO',"GDP_RATIO"]]
    x2_c=x2_c.drop(i,axis=1)
    name=x2.columns
    x2_c=np.log1p(x2_c)
    y_c=np.log1p(y_c)
    x_num_c=ss.transform(x2_c)
    x_c=pd.DataFrame(x_num_c)
    x_c.columns=list(name)
    
    pred=xgbr2.predict(x_c)
    print(i,np.sqrt(((np.expm1(y_c)-np.expm1(pred))**2).sum())/100000000)

# 결과 : SNDIST 1192.752573027487
# 두 번째 반복
# TRADE_COUNTRYCD 1179.020359886674 
for i in ['TRADE_COUNTRYCD', 'TRADE_HSCD', 'TARIFF_AVG','NY_GDP_MKTP_CD',"KR_2017",'SP_POP_TOTL','PA_NUS_FCRF', 'IC_BUS_EASE_DFRN_DB','KMDIST', 'RDIST', 'HSCD_RATIO', 'COUNTRY_RATIO',"GDP_RATIO"]:
    model_c=new[~mask]
    model_c=model_c[model_c['TRADE_HSCD_COUNTRYCD'].isnull()==False]
    model_c=model_c.reset_index().drop('index',axis=1)
    y=model_c["TRADE_HSCD_COUNTRYCD"]
    x=model_c.drop("TRADE_HSCD_COUNTRYCD",axis=1)
    x2=x[['TRADE_COUNTRYCD', 'TRADE_HSCD', 'TARIFF_AVG','NY_GDP_MKTP_CD',"KR_2017",'SP_POP_TOTL','PA_NUS_FCRF', 'IC_BUS_EASE_DFRN_DB','KMDIST', 'RDIST', 'HSCD_RATIO', 'COUNTRY_RATIO',"GDP_RATIO"]]
    x2=x2.drop(i,axis=1)
    name=x2.columns
    x2=np.log1p(x2)
    y=np.log1p(y)
    ss=StandardScaler()
    ss.fit(x2)
    x_num=ss.transform(x2)
    x=pd.DataFrame(x_num)
    x.columns=list(name)
    
    x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=47)


    xgbr2=XGBRegressor(random_state=42)
    xgbr2.fit(x_train,y_train)

    model_compare=new[['TRADE_COUNTRYCD', 'TRADE_HSCD', 'TARIFF_AVG','NY_GDP_MKTP_CD',"KR_2017",'SP_POP_TOTL','PA_NUS_FCRF', 'IC_BUS_EASE_DFRN_DB','KMDIST', 'RDIST', 'HSCD_RATIO', 'COUNTRY_RATIO',"GDP_RATIO","TRADE_HSCD_COUNTRYCD"]]
    model_compare=model_compare[model_compare['TRADE_HSCD_COUNTRYCD'].isnull()==False]
    model_compare=model_compare.reset_index().drop('index',axis=1)
    y_c=model_compare["TRADE_HSCD_COUNTRYCD"]
    x_c=model_compare.drop("TRADE_HSCD_COUNTRYCD",axis=1)
    x2_c=x_c[['TRADE_COUNTRYCD', 'TRADE_HSCD', 'TARIFF_AVG','NY_GDP_MKTP_CD',"KR_2017",'SP_POP_TOTL','PA_NUS_FCRF', 'IC_BUS_EASE_DFRN_DB','KMDIST', 'RDIST', 'HSCD_RATIO', 'COUNTRY_RATIO',"GDP_RATIO"]]
    x2_c=x2_c.drop(i,axis=1)
    name=x2.columns
    x2_c=np.log1p(x2_c)
    y_c=np.log1p(y_c)
    x_num_c=ss.transform(x2_c)
    x_c=pd.DataFrame(x_num_c)
    x_c.columns=list(name)
    
    pred=xgbr2.predict(x_c)
    print(i,np.sqrt(((np.expm1(y_c)-np.expm1(pred))**2).sum())/100000000)

# 결과 : TRADE_COUNTRYCD 1179.020359886674
# 세 번째 반복
for i in ['TRADE_HSCD', 'TARIFF_AVG','NY_GDP_MKTP_CD',"KR_2017",'SP_POP_TOTL','PA_NUS_FCRF', 'IC_BUS_EASE_DFRN_DB','KMDIST', 'RDIST', 'HSCD_RATIO', 'COUNTRY_RATIO',"GDP_RATIO"]:
    model_c=new[~mask]
    model_c=model_c[model_c['TRADE_HSCD_COUNTRYCD'].isnull()==False]
    model_c=model_c.reset_index().drop('index',axis=1)
    y=model_c["TRADE_HSCD_COUNTRYCD"]
    x=model_c.drop("TRADE_HSCD_COUNTRYCD",axis=1)
    x2=x[['TRADE_HSCD', 'TARIFF_AVG','NY_GDP_MKTP_CD',"KR_2017",'SP_POP_TOTL','PA_NUS_FCRF', 'IC_BUS_EASE_DFRN_DB','KMDIST', 'RDIST', 'HSCD_RATIO', 'COUNTRY_RATIO',"GDP_RATIO"]]
    x2=x2.drop(i,axis=1)
    name=x2.columns
    x2=np.log1p(x2)
    y=np.log1p(y)
    ss=StandardScaler()
    ss.fit(x2)
    x_num=ss.transform(x2)
    x=pd.DataFrame(x_num)
    x.columns=list(name)
    
    x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=47)


    xgbr2=XGBRegressor(random_state=42)
    xgbr2.fit(x_train,y_train)

    model_compare=new[['TRADE_HSCD', 'TARIFF_AVG','NY_GDP_MKTP_CD',"KR_2017",'SP_POP_TOTL','PA_NUS_FCRF', 'IC_BUS_EASE_DFRN_DB','KMDIST', 'RDIST', 'HSCD_RATIO', 'COUNTRY_RATIO',"GDP_RATIO","TRADE_HSCD_COUNTRYCD"]]
    model_compare=model_compare[model_compare['TRADE_HSCD_COUNTRYCD'].isnull()==False]
    model_compare=model_compare.reset_index().drop('index',axis=1)
    y_c=model_compare["TRADE_HSCD_COUNTRYCD"]
    x_c=model_compare.drop("TRADE_HSCD_COUNTRYCD",axis=1)
    x2_c=x_c[['TRADE_HSCD', 'TARIFF_AVG','NY_GDP_MKTP_CD',"KR_2017",'SP_POP_TOTL','PA_NUS_FCRF', 'IC_BUS_EASE_DFRN_DB','KMDIST', 'RDIST', 'HSCD_RATIO', 'COUNTRY_RATIO',"GDP_RATIO"]]
    x2_c=x2_c.drop(i,axis=1)
    name=x2.columns
    x2_c=np.log1p(x2_c)
    y_c=np.log1p(y_c)
    x_num_c=ss.transform(x2_c)
    x_c=pd.DataFrame(x_num_c)
    x_c.columns=list(name)
    
    pred=xgbr2.predict(x_c)
    print(i,np.sqrt(((np.expm1(y_c)-np.expm1(pred))**2).sum())/100000000)

# 결과 : HSCD_RATIO 1179.020359886674
# 네 번째 반복
for i in ['TRADE_HSCD', 'TARIFF_AVG','NY_GDP_MKTP_CD',"KR_2017",'SP_POP_TOTL','PA_NUS_FCRF', 'IC_BUS_EASE_DFRN_DB','KMDIST', 'RDIST', 'HSCD_RATIO', 'COUNTRY_RATIO',"GDP_RATIO"]:
    model_c=new[~mask]
    model_c=model_c[model_c['TRADE_HSCD_COUNTRYCD'].isnull()==False]
    model_c=model_c.reset_index().drop('index',axis=1)
    y=model_c["TRADE_HSCD_COUNTRYCD"]
    x=model_c.drop("TRADE_HSCD_COUNTRYCD",axis=1)
    x2=x[['TRADE_HSCD', 'TARIFF_AVG','NY_GDP_MKTP_CD',"KR_2017",'SP_POP_TOTL','PA_NUS_FCRF', 'IC_BUS_EASE_DFRN_DB','KMDIST', 'RDIST', 'HSCD_RATIO', 'COUNTRY_RATIO',"GDP_RATIO"]]
    x2=x2.drop(i,axis=1)
    name=x2.columns
    x2=np.log1p(x2)
    y=np.log1p(y)
    ss=StandardScaler()
    ss.fit(x2)
    x_num=ss.transform(x2)
    x=pd.DataFrame(x_num)
    x.columns=list(name)
    
    x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=47)


    xgbr2=XGBRegressor(random_state=42)
    xgbr2.fit(x_train,y_train)

    model_compare=new[['TRADE_HSCD', 'TARIFF_AVG','NY_GDP_MKTP_CD',"KR_2017",'SP_POP_TOTL','PA_NUS_FCRF', 'IC_BUS_EASE_DFRN_DB','KMDIST', 'RDIST', 'HSCD_RATIO', 'COUNTRY_RATIO',"GDP_RATIO","TRADE_HSCD_COUNTRYCD"]]
    model_compare=model_compare[model_compare['TRADE_HSCD_COUNTRYCD'].isnull()==False]
    model_compare=model_compare.reset_index().drop('index',axis=1)
    y_c=model_compare["TRADE_HSCD_COUNTRYCD"]
    x_c=model_compare.drop("TRADE_HSCD_COUNTRYCD",axis=1)
    x2_c=x_c[['TRADE_HSCD', 'TARIFF_AVG','NY_GDP_MKTP_CD',"KR_2017",'SP_POP_TOTL','PA_NUS_FCRF', 'IC_BUS_EASE_DFRN_DB','KMDIST', 'RDIST', 'HSCD_RATIO', 'COUNTRY_RATIO',"GDP_RATIO"]]
    x2_c=x2_c.drop(i,axis=1)
    name=x2.columns
    x2_c=np.log1p(x2_c)
    y_c=np.log1p(y_c)
    x_num_c=ss.transform(x2_c)
    x_c=pd.DataFrame(x_num_c)
    x_c.columns=list(name)
    
    pred=xgbr2.predict(x_c)
    print(i,np.sqrt(((np.expm1(y_c)-np.expm1(pred))**2).sum())/100000000)    
    
    
######### 그리드 서치 시작 ##########
############## 47번 --- 50번 ################

### 47번 ###
## SP_POP_TOTL TRADE_COUNTRYCD HSCD_RATIO 제거
new=data[['TRADE_COUNTRYCD', 'TRADE_HSCD', 'TARIFF_AVG', 'SNDIST','NY_GDP_MKTP_CD', 'SP_POP_TOTL', 'PA_NUS_FCRF', 'IC_BUS_EASE_DFRN_DB','KMDIST', 'KR_2017', 'GDP_RATIO', 'RDIST', 'HSCD_RATIO','COUNTRY_RATIO','TRADE_HSCD_COUNTRYCD','KR_TRADE_HSCD_COUNTRYCD']]

mask = ((new['KR_TRADE_HSCD_COUNTRYCD']==0)&(new['KR_2017']!=0))|((new['KR_TRADE_HSCD_COUNTRYCD']!=0)&(new['KR_2017']==0))

model_c=new[~mask]
model_c=model_c[model_c['TRADE_HSCD_COUNTRYCD'].isnull()==False]
model_c=model_c.reset_index().drop('index',axis=1)
y=model_c["TRADE_HSCD_COUNTRYCD"]
x=model_c.drop("TRADE_HSCD_COUNTRYCD",axis=1)
x2=x[['TRADE_HSCD', 'TARIFF_AVG', 'SNDIST','NY_GDP_MKTP_CD', 'PA_NUS_FCRF', 'IC_BUS_EASE_DFRN_DB','KMDIST', 'KR_2017', 'GDP_RATIO', 'RDIST','COUNTRY_RATIO']]
name=x2.columns
x2=np.log1p(x2)
y=np.log1p(y)
ss=StandardScaler()
ss.fit(x2)
x_num=ss.transform(x2)
x=pd.DataFrame(x_num)
x.columns=list(name)
    
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=47)


from sklearn.metrics import make_scorer

def RMSE(y_true, y_predict):
    residual_sum_of_squares = sum((np.expm1(y_true) - np.expm1(y_predict)) ** 2)
    mse=residual_sum_of_squares/len(y_true)
    rmse=np.sqrt(mse)
    return rmse

## 1번째 랜덤서치
xgb_params={'learning_rate':uniform(0.1,0.5),
            'n_estimators':randint(150,550),
            'max_depth':randint(1,8),
            'min_child_weight':uniform(0,5)}

xgb_47=XGBRegressor(random_state=42)
rs=RandomizedSearchCV(xgb_47,xgb_params,n_iter=10,cv=5,scoring=make_scorer(RMSE),verbose=2,random_state=42)
rs.fit(x_train,y_train)
est=rs.best_estimator_


print(est.score(x_train,y_train))
print(est.score(x_test,y_test))

cvres=rs.cv_results_

for mean_score, params in zip(cvres["mean_test_score"],cvres["params"]):
    print(np.sqrt(-mean_score),params,'\n')

rs.best_params_
est.fit(x_train,y_train)
pred=est.predict(x_c)

np.sqrt(((np.expm1(y_c)-np.expm1(pred))**2).sum())/100000000



## 2번째 랜덤서치 
xgb_params={'learning_rate':uniform(0.03,0.1),
            'n_estimators':randint(330,450),
            'max_depth':randint(3,8),
            'min_child_weight':uniform(1,5)}

xgb_47=XGBRegressor(random_state=42)
rs=RandomizedSearchCV(xgb_47,xgb_params,n_iter=10,cv=5,scoring=make_scorer(RMSE),verbose=2,random_state=42)
rs.fit(x_train,y_train)
est=rs.best_estimator_

print(est.score(x_train,y_train))
print(est.score(x_test,y_test))

cvres=rs.cv_results_
cvres
for mean_score, params in zip(cvres["mean_test_score"],cvres["params"]):
    print(mean_score,params,'\n')

rs.best_params_

xgb_params={'learning_rate':np.arange(0.03,0.1,0.01),
            'n_estimators':range(300,400,20),
            'max_depth':range(2,5),
            'min_split_loss':np.arange(2,6,0.5),
            'min_child_weight':np.arange(1.5,5.5,1),
            'subsample':np.arange(0.6,1,0.1)}

xgb_47=XGBRegressor(random_state=42)
gs=GridSearchCV(xgb_47,xgb_params,cv=3,scoring=make_scorer(RMSE),verbose=2,return_train_score=True)
gs.fit(x_train,y_train)

gs.best_params_

est=gs.best_estimator_

cvres=gs.cv_results_

for mean_score, params in zip(cvres["mean_test_score"],cvres["params"]):
    print(np.sqrt(-mean_score),params)


model_compare=new[['TRADE_COUNTRYCD', 'TRADE_HSCD', 'TARIFF_AVG', 'SNDIST','NY_GDP_MKTP_CD', 'SP_POP_TOTL', 'PA_NUS_FCRF', 'IC_BUS_EASE_DFRN_DB','KMDIST', 'KR_2017', 'GDP_RATIO', 'RDIST', 'HSCD_RATIO','COUNTRY_RATIO','TRADE_HSCD_COUNTRYCD']]
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
    
pred=xgbr2.predict(x_c)
np.sqrt(((np.expm1(y_c)-np.expm1(pred))**2).sum())/100000000

