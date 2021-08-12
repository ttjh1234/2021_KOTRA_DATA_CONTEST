# -*- coding: utf-8 -*-
"""
Created on Wed Jul  7 01:17:59 2021

@author: 82104
"""

import pandas as pd
import numpy as np
import os
from sklearn.preprocessing import StandardScaler
import sklearn
import matplotlib.pyplot as plt
import seaborn as sns

pd.set_option('display.max_columns',17)

os.chdir("C:/Users/82104/Desktop/공모전1")
import matplotlib.pyplot as plt
import seaborn as sns

pd.set_option('display.max_columns',17)
data=pd.read_csv("./공모전데이터_분석용_KOTRA_0525.csv")

환율=data[data['PA_NUS_FCRF'].isnull()==True]
환율.shape
환율.COUNTRYCD.unique()
환율.COUNTRYNM.unique()

data['PA_NUS_FCRF'].fillna(1.13,inplace=True)
data.info()

유로국가=환율.COUNTRYNM.unique()

data2=pd.read_csv("./Metadata_Country_API_PA.NUS.FCRF_DS2_en_csv_v2_2445345.csv")
data2.head()
data2=data2[['IncomeGroup','TableName']]

# Czechia = Czech Republic
# China, Hong Kong SAR = Hong Kong SAR, China
# Iran, Islamic Rep
# Viet Nam=Vietnam
# Egypt = Egypt, Arab Rep.
# USA = United States.


data2.loc[data2.TableName=='Czech Republic','TableName']='Czechia'
data2.loc[data2.TableName=='Hong Kong SAR, China','TableName']='China, Hong Kong SAR'
data2.loc[data2.TableName=='Iran, Islamic Rep.','TableName']='Iran'
data2.loc[data2.TableName=='Vietnam','TableName']='Viet Nam'
data2.loc[data2.TableName=='Egypt, Arab Rep.','TableName']='Egypt'
data2.loc[data2.TableName=='United States','TableName']='USA'

data3=pd.merge(data,data2,how='left',left_on='COUNTRYNM',right_on='TableName')
data3.head()
data3.info()


data3['PA_NUS_FCRF'].fillna(1.13,inplace=True)
data3.info()

data3=data3.drop("TableName",axis=1)
data3['GDP_DIFF']=data3.NY_GDP_MKTP_CD-data3.NY_GDP_MKTP_CD_1Y
data3.describe()


data4=pd.read_csv("./WtoData_20210629174857.csv")
data4.info()
len(data4['Reporting Economy'].unique())
data4=data4[['Reporting Economy','Product/Sector Code','Year','Value']]
data4.info()
data4.columns=['COUNTRYNM',"HS","Year","tariff"]
data4.loc[data4.COUNTRYNM=='Hong Kong, China','COUNTRYNM']='China, Hong Kong SAR'
data4.loc[data4.COUNTRYNM=='Saudi Arabia, Kingdom of','COUNTRYNM']='Saudi Arabia'
data4.loc[data4.COUNTRYNM=='United States of America','COUNTRYNM']='USA'

data4_2017=data4[data4.Year==2017]
data4_2018=data4[data4.Year==2018]
data4_2017.shape
data4_2018.shape
data4_2017.head()
data3.info()
data4_2017.info()
data4_2017.loc[data4_2017.tariff.isnull()==True,'tariff']=0

len(data4.COUNTRYNM.unique())

data5=data3
sorted(data4.COUNTRYNM.unique())
sorted(data5.COUNTRYNM.unique())
data5.info()

# 품목 분류가 안된 999999 코드는 관세 평균을 0으로 대치.
data5.loc[data5.HSCD==999999,'TARIFF_AVG']=0

data6=data5[data5.TARIFF_AVG.isnull()==True]
HSCD=data6.HSCD.unique()
HSCD

hs_2=[]
for i in range(len(data5["UNC_YEAR"])):
    hs_2.append(str(data5["HSCD"][i])[:2])

data5['HS']=hs_2

data6=data5[data5.TARIFF_AVG.isnull()==True]
data6.head()
data6.loc[:,'HS']=data6.HS.astype('int64')
data6.info()
merge=pd.merge(data6,data4_2017,how='left',left_on=['HS','COUNTRYNM'],right_on=['HS','COUNTRYNM'])
merge.info()
merge.loc[:,'TARIFF_AVG']=merge.tariff

merge.info()

merge[merge.TARIFF_AVG.isnull()==True]
masking=(merge.TARIFF_AVG.isnull()==True) & (merge.COUNTRYNM.isin(유로국가)) &(merge.HS==38)

merge.loc[masking,'TARIFF_AVG']=data4_2017.loc[(data4_2017.COUNTRYNM=='European Union')&(data4_2017.HS==38),'tariff'].values[0]

masking2=(merge.TARIFF_AVG.isnull()==True) & (merge.COUNTRYNM.isin(유로국가)) &(merge.HS==85)

merge.loc[masking2,'TARIFF_AVG']=data4_2017.loc[(data4_2017.COUNTRYNM=='European Union')&(data4_2017.HS==85),'tariff'].values[0]

merge.info()

merge[merge.TARIFF_AVG.isnull()==True]

merge.info()
merge=merge[['HSCD','COUNTRYNM','TARIFF_AVG']]
merge.info()
merge
merge.columns=['HSCD','COUNTRYNM','Imputate']

data7=data5

data7=pd.merge(data7,merge,how='left',left_on=['HSCD','COUNTRYNM'],right_on=['HSCD','COUNTRYNM'])
data7.info()
data7.loc[data7.TARIFF_AVG.isnull()==True,'TARIFF_AVG']=data7.loc[data7.TARIFF_AVG.isnull()==True,'Imputate']
data7.info()


# 29개만 남음 => 같은 HS와 비교해서 대입하기.
data7=data7.drop('Imputate',axis=1)
tt=data7.loc[data7.TARIFF_AVG.isnull()==True,['COUNTRYNM','HS']]
data7.describe()
data7.info()

maskout=(data7.TARIFF_AVG.isnull()==True)

data7[maskout]

for i,j in zip(tt.COUNTRYNM,tt.HS):
    maskfor=(data7.COUNTRYNM==i)&(data7.HS==j)
    data7.loc[maskout&(maskfor),'TARIFF_AVG']=data7.loc[maskfor,'TARIFF_AVG'].mean()
    
data7.info()

tt2=data7.loc[data7.SNDIST.isnull()==True,'COUNTRYNM'].unique()

for i in tt2:
    print(data7.loc[data7.COUNTRYNM==i,'SNDIST'].mean())
    print(data7.loc[data7.COUNTRYNM==i,'SNDIST'].median())
    
for i in tt2:
    data7.loc[(data7.SNDIST.isnull()==True)&(data7.COUNTRYNM==i),'SNDIST']=data7.loc[data7.COUNTRYNM==i,'SNDIST'].mean()

data7.info()

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

ex.info()    
ex1 = ex.drop_duplicates(["hscd"])

ex1.info()

data8 = pd.merge(data7,ex1,how="left",left_on='HSCD',right_on="hscd")
data8 = data8.drop("hscd",axis=1)
data8.info()

data8.loc[data8.group.isnull()==True,'group']='5'
data8.info()

#FTA체결 변수 추가 
FTA = pd.read_excel("./국가별FTA체결현황.xlsx")

FTA.head()

data9=pd.merge(data8,FTA,how='left',left_on='COUNTRYNM',right_on='country')
data9.info()

data9["FTA"] = data9["FTA"].astype(str)
data7=data9.drop(['country'],axis=1)
data7['RDIST']=data7['KMDIST']/data7['SNDIST']
data7['RDIST']=data7['RDIST'].astype('float')
data7.info()


data7.info()
data9=data7
data11=data7
data11.info()

from sklearn.linear_model import LinearRegression
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures

from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import GridSearchCV
from scipy.stats import uniform, randint

data8=data7
data8.info()
data8=data8.drop(['UNC_YEAR','COUNTRYCD','KMDIST','RDIST','HSCD','KR_TRADE_HSCD_COUNTRYCD','NY_GDP_MKTP_CD_1Y','FTA','GDP_DIFF'],axis=1)
data8=pd.get_dummies(data8,drop_first=True)
pred=data8[data8.TRADE_HSCD_COUNTRYCD.isnull()==True]
play=data8[data8.TRADE_HSCD_COUNTRYCD.isnull()==False]
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
y_pv=np.exp(y_pv-1)
y_pv.shape
pred['pred']=y_pv

pred.info()

data7.info()
data9=data7
data9.info()
pred_value=list(pred['pred'])

data9.loc[data9.TRADE_HSCD_COUNTRYCD.isnull()==True,'TRADE_HSCD_COUNTRYCD']=pred_value

data9.info()
data7.info()

## 데이터 누락값 처리 완료 ##

## 상관성 보는 중 ##
num=data11.drop(['UNC_YEAR','HSCD','COUNTRYCD','COUNTRYNM','IncomeGroup','HS','group','FTA'],axis=1)
sns.heatmap(num.corr(),annot=True)

# Standard Scalar 사용 + 타깃 변수 로그
num.columns
stop=num[['GDP_DIFF','TARIFF_AVG']]
num=num.drop(['GDP_DIFF','TARIFF_AVG'],axis=1)
num=np.log1p(num)

num2=pd.concat([num,stop],axis=1)
columns=num2.columns

ss=StandardScaler()
ss.fit(num2)
num2=ss.transform(num2)
num2=pd.DataFrame(num2)
num2.columns=columns
num2

sns.heatmap(num2.corr(),annot=True)

num2.info()
cat=data11.loc[:,['COUNTRYNM','IncomeGroup','HS','group','FTA']]
cat2=pd.get_dummies(cat,drop_first=True)
data=pd.concat([num2,cat2],axis=1)

data.info()

y=data['KR_TRADE_HSCD_COUNTRYCD']
x=data.drop('KR_TRADE_HSCD_COUNTRYCD',axis=1)

x=np.array(x)

x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.3,random_state=42)

import matplotlib.pyplot as plt
import seaborn as sns

## 
data

data10=data9.drop('GDP_DIFF',axis=1)
data10['GDP_RATIO']=data9['NY_GDP_MKTP_CD']/data9['NY_GDP_MKTP_CD_1Y']

data10.info()

data10['dist']=pd.qcut(data10['SNDIST'],3,labels=[1,3,5])

data10[['dist','KR_TRADE_HSCD_COUNTRYCD']].groupby('dist').mean()

data10['TARIFF']=0
data10.loc[data10['TARIFF_AVG']>0,'TARIFF']=1
data10[['TARIFF','KR_TRADE_HSCD_COUNTRYCD']].groupby('TARIFF').mean()

data10['PA']=pd.qcut(data10['PA_NUS_FCRF'],2,labels=[0,1])
data10[['PA','KR_TRADE_HSCD_COUNTRYCD']].groupby('PA').mean()

data10['BE']=pd.qcut(data10['IC_BUS_EASE_DFRN_DB'],3,labels=[1,3,5])
data10[['BE','KR_TRADE_HSCD_COUNTRYCD']].groupby('BE').mean()

data10.info()

try1=data10
try1=try1.drop(['SNDIST','NY_GDP_MKTP_CD_1Y','UNC_YEAR','COUNTRYCD','PA_NUS_FCRF','IC_BUS_EASE_DFRN_DB','TARIFF_AVG'],axis=1)

try1.info()
try1.HSCD=try1.HSCD.astype('object')

try1.TARIFF=try1.TARIFF.astype('category')

## 2021-07-07


data7.describe()
data7.info()

data7['dist']=pd.qcut(data7['SNDIST'],3,labels=[1,3,5])

data7[['dist','KR_TRADE_HSCD_COUNTRYCD']].groupby('dist').mean()

data7['TARIFF']=0
data7.loc[data7['TARIFF_AVG']>0,'TARIFF']=1
data7[['TARIFF','KR_TRADE_HSCD_COUNTRYCD']].groupby('TARIFF').mean()

data7['PA']=pd.qcut(data7['PA_NUS_FCRF'],2,labels=[0,1])
data7[['PA','KR_TRADE_HSCD_COUNTRYCD']].groupby('PA').mean()

data7['BE']=pd.qcut(data7['IC_BUS_EASE_DFRN_DB'],3,labels=[1,3,5])
data7[['BE','KR_TRADE_HSCD_COUNTRYCD']].groupby('BE').mean()

data7.info()
data7.TARIFF=data7.TARIFF.astype('category')

do=data7


# 데이터 확인용
# (외국기준)수입 데이터
file_=os.listdir("./2018_import")
file_
len(file_)
file_path_1="./2018_import"

idata=pd.DataFrame()
for i in range(1,12):
    filename = "comtrade (" + str(i) + ").csv"
    tem = pd.read_csv(file_path_1+'/'+filename)
    tem = tem[["Reporter","Commodity Code","Trade Value (US$)","Mode of Transport","Customs","2nd Partner"]]
    tem = tem[(tem["Mode of Transport"]=="All MOTs")]
    tem = tem[tem["Customs"]=="All CPCs"]
    tem = tem[tem["2nd Partner"] == "World"]
    tem.columns = ["COUNTRYNM","HSCD","2018_KR_TRADE_HSCD_COUNTRYCD","type","type2","type3"]
    tem=tem.drop(['type','type2','type3'],axis=1)
    
    tem["HSCD"] = tem.HSCD.astype("int64")
    tem=tem.reset_index().drop('index',axis=1)
    idata = pd.concat([idata,tem])
    idata=idata.reset_index().drop('index',axis=1)

idata.info()

# 한국 수출 데이터

file_2=os.listdir("./2018_export")
file_2
len(file_2)
file_path_2="./2018_export"

edata=pd.DataFrame()
for i in range(1,12):
    filename = "comtrade (" + str(i) + ").csv"
    tem = pd.read_csv(file_path_2+'/'+filename)
    tem = tem[["Partner","Commodity Code","Trade Value (US$)","Mode of Transport","Customs","2nd Partner"]]
    tem = tem[(tem["Mode of Transport"]=="All MOTs")]
    tem = tem[tem["Customs"]=="All CPCs"]
    tem = tem[tem["2nd Partner"] == "World"]
    tem.columns = ["COUNTRYNM","HSCD","2018_KR_TRADE_HSCD_COUNTRYCD","type","type2","type3"]
    tem=tem.drop(['type','type2','type3'],axis=1)
    
    tem["HSCD"] = tem.HSCD.astype("int64")
    tem=tem.reset_index().drop('index',axis=1)
    edata = pd.concat([edata,tem])
    edata=edata.reset_index().drop('index',axis=1)

edata.info()

idata.columns=['COUNTRYNM','HSCD','2018_KR_IMPORT']
edata.columns=['COUNTRYNM','HSCD','2018_KR_EXPORT']

# 잘못 짬.
ie=pd.merge(idata,edata,how='left',left_on=['COUNTRYNM','HSCD'],right_on=['COUNTRYNM','HSCD'])

ie.loc[ie['2018_KR_EXPORT'].isnull()==True,'COUNTRYNM'].unique()

data7.HSCD.min()

ie=ie[(189999<ie.HSCD)&(ie.HSCD<1000000)]

ie['KR_COMPARE']=0

ie.info()
ie.loc[ie['2018_KR_EXPORT'].isnull()==True,'2018_KR_EXPORT'].unique()

ie['2018_KR_EXPORT'].fillna(0,inplace=True)

ie=ie.reset_index().drop('index',axis=1)
ie['2018_KR_EXPORT']=ie['2018_KR_EXPORT'].astype('int64')

def imputate_KR(a,b):
    if a>b:
        return a
    else: 
        return b


ie['KR_COMPARE']=ie.apply(lambda x:imputate_KR(x['2018_KR_IMPORT'], x['2018_KR_EXPORT']),axis=1)

len(data7.COUNTRYNM.unique())
list1=list(data7.COUNTRYNM.unique())
list2=list(ie.COUNTRYNM.unique())

list1=sorted(list1)
list2=sorted(list2)

list3=[]

for i in list1:
    if i in list2:
        continue
    else : list3.append(i)
list3

## Algeria , Sri Lanka 를 제외하고는 다 있음.
#data7.info()
#c_data=data7[['HSCD','COUNTRYNM','KR_TRADE_HSCD_COUNTRYCD']]

#comp=pd.merge(c_data,ie,how='left',left_on=['HSCD','COUNTRYNM'],right_on=['HSCD','COUNTRYNM'])

#comp['comp1']=comp['KR_TRADE_HSCD_COUNTRYCD']-comp['2018_KR_IMPORT']
#comp['comp2']=comp['KR_TRADE_HSCD_COUNTRYCD']-comp['2018_KR_EXPORT']
#comp['comp3']=comp['KR_TRADE_HSCD_COUNTRYCD']-comp['KR_COMPARE']

#comp.describe()
#np.corr
##np.corrcoef(comp['KR_TRADE_HSCD_COUNTRYCD'],comp['2018_KR_IMPORT'])


ie2=pd.merge(idata,edata,how='outer',left_on=['COUNTRYNM','HSCD'],right_on=['COUNTRYNM','HSCD'])

ie2.info()

list1=list(data7.COUNTRYNM.unique())
list2=list(ie2.COUNTRYNM.unique())

list1=sorted(list1)
list2=sorted(list2)

list3=[]

for i in list1:
    if i in list2:
        continue
    else : list3.append(i)
list3

list3=[]

for i in list2:
    if i in list1:
        continue
    else : list3.append(i)
list3

## 모든 국가 다 존재
# 사용하지 않는 HSCD 다 제거
ie2=ie2[(189999<ie2.HSCD)&(ie2.HSCD<1000000)]

ie2.info()

ie2=ie2.reset_index().drop('index',axis=1)

ie2['2018_KR_EXPORT'].fillna(0,inplace=True)
ie2['2018_KR_IMPORT'].fillna(0,inplace=True)

def imputate_KR(a,b):
    if a>b:
        return a
    else: 
        return b


ie2['KR_COMPARE']=ie2.apply(lambda x:imputate_KR(x['2018_KR_IMPORT'], x['2018_KR_EXPORT']),axis=1)

ie2.info()
data7.info()
c_data=data7[['HSCD','COUNTRYNM','KR_TRADE_HSCD_COUNTRYCD']]

comp=pd.merge(c_data,ie2,how='left',left_on=['HSCD','COUNTRYNM'],right_on=['HSCD','COUNTRYNM'])
comp.info()

comp['comp1']=comp['KR_TRADE_HSCD_COUNTRYCD']-comp['2018_KR_IMPORT']
comp['comp2']=comp['KR_TRADE_HSCD_COUNTRYCD']-comp['2018_KR_EXPORT']
comp['comp3']=comp['KR_TRADE_HSCD_COUNTRYCD']-comp['KR_COMPARE']

comp.loc[comp.KR_COMPARE.isnull()==True,'COUNTRYNM'].unique()

comp2=comp[comp.KR_COMPARE.isnull()==False]
comp2=comp2.reset_index().drop('index',axis=1)
comp2.info()

comp2.iloc[:,2:5]
np.corrcoef(comp2['KR_TRADE_HSCD_COUNTRYCD'],comp2['2018_KR_IMPORT'])
np.corrcoef(comp2['KR_TRADE_HSCD_COUNTRYCD'],comp2['2018_KR_EXPORT'])
np.corrcoef(comp2['KR_TRADE_HSCD_COUNTRYCD'],comp2['KR_COMPARE'])

from sklearn.linear_model import LinearRegression
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures

x=comp2[['2018_KR_IMPORT','2018_KR_EXPORT','KR_COMPARE']]
y=comp2['KR_TRADE_HSCD_COUNTRYCD']

lr=LinearRegression()
lr.fit(x,y)
print(lr.score(x,y))

## 2021-07-07 저녁
file_=os.listdir("./2017_import")
file_
file_path_1="./2017_import"

idata=pd.DataFrame()
for i in range(1,len(file_)+1):
    filename = "comtrade (" + str(i) + ").csv"
    tem = pd.read_csv(file_path_1+'/'+filename)
    tem = tem[["Reporter","Commodity Code","Trade Value (US$)","Mode of Transport","Customs","2nd Partner"]]
    tem = tem[(tem["Mode of Transport"]=="All MOTs")]
    tem = tem[tem["Customs"]=="All CPCs"]
    tem = tem[tem["2nd Partner"] == "World"]
    tem.columns = ["COUNTRYNM","HSCD","2017_KR_TRADE_HSCD_COUNTRYCD","type","type2","type3"]
    tem=tem.drop(['type','type2','type3'],axis=1)
    
    tem["HSCD"] = tem.HSCD.astype("int64")
    tem=tem.reset_index().drop('index',axis=1)
    idata = pd.concat([idata,tem])
    idata=idata.reset_index().drop('index',axis=1)

idata.info()

file_2=os.listdir("./2017_export")
file_2
len(file_2)
file_path_2="./2017_export"

edata=pd.DataFrame()
for i in range(1,len(file_2)+1):
    filename = "comtrade (" + str(i) + ").csv"
    tem = pd.read_csv(file_path_2+'/'+filename)
    tem = tem[["Partner","Commodity Code","Trade Value (US$)","Mode of Transport","Customs","2nd Partner"]]
    tem = tem[(tem["Mode of Transport"]=="All MOTs")]
    tem = tem[tem["Customs"]=="All CPCs"]
    tem = tem[tem["2nd Partner"] == "World"]
    tem.columns = ["COUNTRYNM","HSCD","2017_KR_TRADE_HSCD_COUNTRYCD","type","type2","type3"]
    tem=tem.drop(['type','type2','type3'],axis=1)
    
    tem["HSCD"] = tem.HSCD.astype("int64")
    tem=tem.reset_index().drop('index',axis=1)
    edata = pd.concat([edata,tem])
    edata=edata.reset_index().drop('index',axis=1)

edata.info()

idata.columns=['COUNTRYNM','HSCD','2017_KR_IMPORT']
edata.columns=['COUNTRYNM','HSCD','2017_KR_EXPORT']

#ie2=pd.merge(idata,edata,how='outer',left_on=['COUNTRYNM','HSCD'],right_on=['COUNTRYNM','HSCD'])

#ie2.info()

#list1=list(data7.COUNTRYNM.unique())
#list2=list(ie2.COUNTRYNM.unique())

#list1=sorted(list1)
#list2=sorted(list2)

#list3=[]

#for i in list1:
    #if i in list2:
        #continue
    #else : list3.append(i)
#list3

#list3=[]

#for i in list2:
    #if i in list1:
        #continue
    #else : list3.append(i)
#list3

## 모든 국가 다 존재
# 사용하지 않는 HSCD 다 제거
#ie2=ie2[(189999<ie2.HSCD)&(ie2.HSCD<1000000)]

#ie2.info()

#ie2=ie2.reset_index().drop('index',axis=1)

except_country=['Guatemala','Viet Nam','Iran','Egypt']

data20=pd.merge(data7,idata,how='left',left_on=['HSCD','COUNTRYNM'],right_on=['HSCD','COUNTRYNM'])
data20.info()
data20=pd.merge(data20,edata,how='left',left_on=['HSCD','COUNTRYNM'],right_on=['HSCD','COUNTRYNM'])
data20.info()


ie2=data20
ie2.info()
ie2=ie2.reset_index().drop('index',axis=1)
ie2.head()

ie2=ie2.fillna({'2017_KR_IMPORT':-99,'2017_KR_EXPORT':-99})

def imputate_KR(a,b,c):
    if (c not in except_country):
        if a!=-99:
            return a
        elif (a==-99)&(b!=-99):
            return b
        else:
            return np.nan
    else:
        if b!=-99:
            return b
        elif a!=-99:
            return a
        else: return np.nan

ie2['KR_2017']=ie2.apply(lambda x:imputate_KR(x['2017_KR_IMPORT'], x['2017_KR_EXPORT'],x['COUNTRYNM']),axis=1)

ie2.info()
ie2[(ie2['2017_KR_IMPORT']==-99)&(ie2['2017_KR_EXPORT']==-99)]
#123개 존재

# 확인용 => 두 개 다 누락치인 except_country인 경우 nan 사용 및 import가 있으면 import 데이터 사용으로 함수 바꿈
#ie2[(ie2['2017_KR_EXPORT']==-99)&(ie2['COUNTRYNM'].isin(except_country))]

ie2.info()
ie3=ie2.dropna(axis=0,how='any')
ie3=ie3.reset_index().drop('index',axis=1)
ie3.info()

model_data=ie3.drop(['UNC_YEAR','COUNTRYCD'],axis=1)
model_data.HSCD=model_data.HSCD.astype('object')
model_data=model_data.drop('GDP_DIFF',axis=1)
model_data['GDP_RATIO']=model_data['NY_GDP_MKTP_CD']/model_data['NY_GDP_MKTP_CD_1Y']
model_data=model_data.drop(['2017_KR_IMPORT','2017_KR_EXPORT'],axis=1)

model_data.info()
#model_cat=model_data[['dist','TARIFF','PA','BE']]
#model_data=model_data.drop(['dist','TARIFF','PA','BE'],axis=1)
model_data=model_data.drop('HSCD',axis=1)

model_target=model_data['KR_TRADE_HSCD_COUNTRYCD']
model_data=model_data.drop('KR_TRADE_HSCD_COUNTRYCD',axis=1)

model_data2=pd.get_dummies(model_data,drop_first=True)
columns=model_data2.columns
model_data2.info()
model_num=model_data2.loc[:,:'GDP_RATIO']
model_num.shape
model_obj=model_data2.loc[:,'COUNTRYNM_Australia':]
model_obj.shape

model_num=np.log1p(model_num)
model_target=np.log1p(model_target)

ss=StandardScaler()
ss.fit(model_num)
x_num=ss.transform(model_num)

#x=np.concatenate([x_num,model_obj,model_cat],axis=1)
x=np.concatenate([x_num,model_obj],axis=1)
x_train,x_test,y_train,y_test=train_test_split(x,model_target,test_size=0.3,random_state=42)

from xgboost import XGBRegressor

xgb=XGBRegressor(random_state=42)
xgb.fit(x_train,y_train)
print(xgb.score(x_train,y_train))

print(xgb.score(x_test,y_test))

from lightgbm import LGBMRegressor
lgb=LGBMRegressor(random_state=42)
lgb.fit(x_train,y_train)
print(lgb.score(x_train,y_train))

print(lgb.score(x_test,y_test))


from lightgbm import plot_importance
%matplotlib inline

fig,ax=plt.subplots(figsize=(10,12))
plot_importance(lgb,ax=ax)

name=list(columns)+['dist','TARIFF','PA','BE']
name[104]
best=[12,1,10,11,3,2,8,0,13,6,7,4,93,26,101]
for i,j in zip(best,range(1,len(best))):
    print(j,'번째 순위 : ',name[i])

from sklearn.svm import SVR

svr_reg=SVR(kernel='rbf')
svr_reg.fit(x_train,y_train)
print(svr_reg.score(x_train,y_train))
print(svr_reg.score(x_test,y_test))
pred=svr_reg.predict(x_test)

from sklearn.linear_model import SGDRegressor

sr=SGDRegressor(random_state=42)
train_score=[]
test_score=[]

for _ in range(0,300):
    sr.partial_fit(x_train,y_train)
    train_score.append(sr.score(x_train,y_train))
    test_score.append(sr.score(x_test,y_test))
    
import matplotlib.pyplot as plt

plt.plot(train_score)
plt.plot(test_score)
plt.show()
















