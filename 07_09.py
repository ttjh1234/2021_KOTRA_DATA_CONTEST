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


data10=data9.drop('GDP_DIFF',axis=1)
data10['GDP_RATIO']=data9['NY_GDP_MKTP_CD']/data9['NY_GDP_MKTP_CD_1Y']
data10['TRADE_RATIO']=data10['TRADE_HSCD_COUNTRYCD']/data10['TRADE_HSCD']


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
            return 0
    else:
        if b!=-99:
            return b
        else: return 0

ie2['KR_2017']=ie2.apply(lambda x:imputate_KR(x['2017_KR_IMPORT'], x['2017_KR_EXPORT'],x['COUNTRYNM']),axis=1)

ie2.info()
ie2.to_csv("data_sungsu.csv")

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
model_cat=model_data[['dist','TARIFF','PA','BE']]
model_data=model_data.drop(['dist','TARIFF','PA','BE'],axis=1)
model_data=model_data.drop('HSCD',axis=1)

model_target=model_data['KR_TRADE_HSCD_COUNTRYCD']
model_data=model_data.drop('KR_TRADE_HSCD_COUNTRYCD',axis=1)

model_data2=pd.get_dummies(model_data,drop_first=True)
columns=model_data2.columns
model_data2.info()
model_num=model_data2.loc[:,:'GDP_RATIO']
model_num.shape
model_obj=model_data2.iloc[:,14:]
model_obj.shape


model_num=np.log1p(model_num)
model_target=np.log1p(model_target)

ss=StandardScaler()
ss.fit(model_num)
x_num=ss.transform(model_num)

x=np.concatenate([x_num,model_obj,model_cat],axis=1)
# x=np.concatenate([x_num,model_obj],axis=1)
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


## 중요했던 변수만 사용해보기
x_import=pd.DataFrame(x)
x_import.columns=name
x_import.head()
x_import=x_import.iloc[:,best[0:10]]
x_import.info()

x_train,x_test,y_train,y_test=train_test_split(x_import,model_target,test_size=0.3,random_state=42)

from xgboost import XGBRegressor

xgb=XGBRegressor(random_state=42)
xgb.fit(x_train,y_train)
print(xgb.score(x_train,y_train))

print(xgb.score(x_test,y_test))

# 갑자기 문득 든 생각인데, 우리가 구한 2자리 HS 코드 분류를 통해 총 4가지(+99 미분류 포함 5가지)의 분류를 했는데
# 그 정보를 바로 사용하지 않고, 국가별 4가지 대분류 중에 어떤 것을 많이 사는 나라인지에 대한 정보가 들어갔으면 좋겠음. (간단)
# 복잡한 것 : 가장 좋은 것은 국가별로 어떤 품목을 선호하고 비선호 하는지 .. 

check=ie3
check.info()
check=check[['COUNTRYNM','HS','HSCD','TRADE_HSCD_COUNTRYCD','KR_2017']]
group=check.groupby(['COUNTRYNM','HS'])
print(group)

for key,group in group:
    print(key[0])
    print(group)

group_mean=group.agg({'TRADE_HSCD_COUNTRYCD':'mean','KR_2017':'mean'})

group_mean[0]

dict_1={}
dict_2={}
for key,group in group:
    print(key)
    group_mean=group.agg({'TRADE_HSCD_COUNTRYCD':'mean','KR_2017':'mean'})
    country=key[0]
    hs=key[1]
    

dict_1.keys()

# 각각 적합시켜볼까?
ie3=ie2.dropna(axis=0,how='any')
ie3=ie3.reset_index().drop('index',axis=1)

model_data=ie3.drop(['UNC_YEAR','COUNTRYCD'],axis=1)
model_data.HSCD=model_data.HSCD.astype('object')
model_data=model_data.drop('GDP_DIFF',axis=1)
model_data['GDP_RATIO']=model_data['NY_GDP_MKTP_CD']/model_data['NY_GDP_MKTP_CD_1Y']
model_data=model_data.drop(['2017_KR_IMPORT','2017_KR_EXPORT'],axis=1)

model_data.info()
model_data['group'].unique()

for i in model_data['group'].unique():
    if i=='1':
        model_1=model_data[model_data['group']==i]
        model_1_cat=model_1[['dist','TARIFF','PA','BE']]
        model_1=model_1.drop(['dist','TARIFF','PA','BE'],axis=1)
        model_1=model_1.drop('HSCD',axis=1)

        model_1_target=model_1['KR_TRADE_HSCD_COUNTRYCD']
        model_1=model_1.drop('KR_TRADE_HSCD_COUNTRYCD',axis=1)

        model_1=pd.get_dummies(model_1,drop_first=True)
        columns=model_1.columns
        model_1_num=model_1.loc[:,:'GDP_RATIO']
        model_1_obj=model_1.iloc[:,14:]
        
        model_1_num=np.log1p(model_1_num)
        model_1_target=np.log1p(model_1_target)

        ss=StandardScaler()
        ss.fit(model_1_num)
        x_num_1=ss.transform(model_1_num)

        x_1=np.concatenate([x_num_1,model_1_obj,model_1_cat],axis=1)
    
    elif i=='2':
        model_2=model_data[model_data['group']==i]
        model_2_cat=model_2[['dist','TARIFF','PA','BE']]
        model_2=model_2.drop(['dist','TARIFF','PA','BE'],axis=1)
        model_2=model_2.drop('HSCD',axis=1)

        model_2_target=model_2['KR_TRADE_HSCD_COUNTRYCD']
        model_2=model_2.drop('KR_TRADE_HSCD_COUNTRYCD',axis=1)

        model_2=pd.get_dummies(model_2,drop_first=True)
        columns=model_2.columns
        model_2_num=model_2.loc[:,:'GDP_RATIO']
        model_2_obj=model_2.iloc[:,14:]
        
        model_2_num=np.log1p(model_2_num)
        model_2_target=np.log1p(model_2_target)

        ss=StandardScaler()
        ss.fit(model_2_num)
        x_num_2=ss.transform(model_2_num)

        x_2=np.concatenate([x_num_2,model_2_obj,model_2_cat],axis=1)
    
    elif i=='3':
        model_3=model_data[model_data['group']==i]
        model_3_cat=model_3[['dist','TARIFF','PA','BE']]
        model_3=model_3.drop(['dist','TARIFF','PA','BE'],axis=1)
        model_3=model_3.drop('HSCD',axis=1)

        model_3_target=model_3['KR_TRADE_HSCD_COUNTRYCD']
        model_3=model_3.drop('KR_TRADE_HSCD_COUNTRYCD',axis=1)

        model_3=pd.get_dummies(model_3,drop_first=True)
        columns=model_3.columns
        model_3_num=model_3.loc[:,:'GDP_RATIO']
        model_3_obj=model_3.iloc[:,14:]
        
        model_3_num=np.log1p(model_3_num)
        model_3_target=np.log1p(model_3_target)

        ss=StandardScaler()
        ss.fit(model_3_num)
        x_num_3=ss.transform(model_3_num)

        x_3=np.concatenate([x_num_3,model_3_obj,model_3_cat],axis=1)
    
    elif i=='4':
        model_4=model_data[model_data['group']==i]
        model_4_cat=model_4[['dist','TARIFF','PA','BE']]
        model_4=model_4.drop(['dist','TARIFF','PA','BE'],axis=1)
        model_4=model_4.drop('HSCD',axis=1)

        model_4_target=model_4['KR_TRADE_HSCD_COUNTRYCD']
        model_4=model_4.drop('KR_TRADE_HSCD_COUNTRYCD',axis=1)

        model_4=pd.get_dummies(model_4,drop_first=True)
        columns=model_4.columns
        model_4_num=model_4.loc[:,:'GDP_RATIO']
        model_4_obj=model_4.iloc[:,14:]
        
        model_4_num=np.log1p(model_4_num)
        model_4_target=np.log1p(model_4_target)

        ss=StandardScaler()
        ss.fit(model_4_num)
        x_num_4=ss.transform(model_4_num)

        x_4=np.concatenate([x_num_4,model_4_obj,model_4_cat],axis=1)

    else:
        model_5=model_data[model_data['group']==i]
        model_5_cat=model_5[['dist','TARIFF','PA','BE']]
        model_5=model_5.drop(['dist','TARIFF','PA','BE'],axis=1)
        model_5=model_5.drop('HSCD',axis=1)

        model_5_target=model_5['KR_TRADE_HSCD_COUNTRYCD']
        model_5=model_5.drop('KR_TRADE_HSCD_COUNTRYCD',axis=1)

        model_5=pd.get_dummies(model_5,drop_first=True)
        columns=model_5.columns
        model_5_num=model_5.loc[:,:'GDP_RATIO']
        model_5_obj=model_5.iloc[:,14:]
        
        model_5_num=np.log1p(model_5_num)
        model_5_target=np.log1p(model_5_target)

        ss=StandardScaler()
        ss.fit(model_5_num)
        x_num_5=ss.transform(model_5_num)

        x_5=np.concatenate([x_num_5,model_5_obj,model_5_cat],axis=1)


from xgboost import XGBRegressor
for i in model_data['group'].unique():
    if i=='1':
        x_train,x_test,y_train,y_test=train_test_split(x_1,model_1_target,test_size=0.3,random_state=42)
        xgb=XGBRegressor(random_state=42)
        xgb.fit(x_train,y_train)
        print('Group 1 \n')
        print("shape : ",x_1.shape)
        print(xgb.score(x_train,y_train))
        print(xgb.score(x_test,y_test),"\n")
        
    elif i=='2':
        x_train,x_test,y_train,y_test=train_test_split(x_2,model_2_target,test_size=0.3,random_state=42)
        xgb=XGBRegressor(random_state=42)
        xgb.fit(x_train,y_train)
        print('Group 2 \n')
        print("shape : ",x_2.shape)
        print(xgb.score(x_train,y_train))
        print(xgb.score(x_test,y_test),"\n")
        
    elif i=='3':
        x_train,x_test,y_train,y_test=train_test_split(x_3,model_3_target,test_size=0.3,random_state=42)
        xgb=XGBRegressor(random_state=42)
        xgb.fit(x_train,y_train)
        print('Group 3 \n')
        print("shape : ",x_3.shape)
        print(xgb.score(x_train,y_train))
        print(xgb.score(x_test,y_test),"\n")

    elif i=='4':
        x_train,x_test,y_train,y_test=train_test_split(x_4,model_4_target,test_size=0.3,random_state=42)
        xgb=XGBRegressor(random_state=42)
        xgb.fit(x_train,y_train)
        print('Group 4 \n')
        print("shape : ",x_4.shape)
        print(xgb.score(x_train,y_train))
        print(xgb.score(x_test,y_test),"\n")


    else:
        x_train,x_test,y_train,y_test=train_test_split(x_5,model_5_target,test_size=0.3,random_state=42)
        xgb=XGBRegressor(random_state=42)
        xgb.fit(x_train,y_train)
        print('Group 5 \n')
        print("shape : ",x_5.shape)
        print(xgb.score(x_train,y_train))
        print(xgb.score(x_test,y_test),"\n")


x=pd.DataFrame(x)
name=list(columns)+['dist','TARIFF','PA','BE']
x.columns=name
x=x.drop(['group_2','group_3','group_4','group_5'],axis=1)
x['group']=ie3['group']
x.info()
x['y']=model_target
x['y']

for i in model_data['group'].unique():
    if i=='1':
        x_1=x[x['group']=='1']
        model_1_target=x_1['y']
        x_1=x_1.drop(['y','group'],axis=1)
        x_train,x_test,y_train,y_test=train_test_split(x_1,model_1_target,test_size=0.3,random_state=42)
        xgb=XGBRegressor(random_state=42)
        xgb.fit(x_train,y_train)
        print('Group 1 \n')
        print("shape : ",x_1.shape)
        print(xgb.score(x_train,y_train))
        print(xgb.score(x_test,y_test),"\n")
        
    elif i=='2':
        x_2=x[x['group']=='2']
        model_2_target=x_2['y']
        x_2=x_2.drop(['y','group'],axis=1)
        x_train,x_test,y_train,y_test=train_test_split(x_2,model_2_target,test_size=0.3,random_state=42)
        xgb=XGBRegressor(random_state=42)
        xgb.fit(x_train,y_train)
        print('Group 2 \n')
        print("shape : ",x_2.shape)
        print(xgb.score(x_train,y_train))
        print(xgb.score(x_test,y_test),"\n")
        
    elif i=='3':
        x_3=x[x['group']=='3']
        model_3_target=x_3['y']
        x_3=x_3.drop(['y','group'],axis=1)
        x_train,x_test,y_train,y_test=train_test_split(x_3,model_3_target,test_size=0.3,random_state=42)
        xgb=XGBRegressor(random_state=42)
        xgb.fit(x_train,y_train)
        print('Group 3 \n')
        print("shape : ",x_3.shape)
        print(xgb.score(x_train,y_train))
        print(xgb.score(x_test,y_test),"\n")

    elif i=='4':
        x_4=x[x['group']=='4']
        model_4_target=x_4['y']
        x_4=x_4.drop(['y','group'],axis=1)
        x_train,x_test,y_train,y_test=train_test_split(x_4,model_4_target,test_size=0.3,random_state=42)
        xgb=XGBRegressor(random_state=42)
        xgb.fit(x_train,y_train)
        print('Group 4 \n')
        print("shape : ",x_4.shape)
        print(xgb.score(x_train,y_train))
        print(xgb.score(x_test,y_test),"\n")


    else:
        x_5=x[x['group']=='5']
        model_5_target=x_5['y']
        x_5=x_5.drop(['y','group'],axis=1)
        x_train,x_test,y_train,y_test=train_test_split(x_5,model_5_target,test_size=0.3,random_state=42)
        xgb=XGBRegressor(random_state=42)
        xgb.fit(x_train,y_train)
        print('Group 5 \n')
        print("shape : ",x_5.shape)
        print(xgb.score(x_train,y_train))
        print(xgb.score(x_test,y_test),"\n")

## 품목 5그룹 별로 적합한 결과도 별로 좋지 않았다. 오히려 과적합
## 그러면 어떠한 변수를 추가해야지 이러한 변수들이 설명력이 있을까
## 

data=pd.read_csv('data_sungsu.csv')

data.head()

data.info()

data=data.drop('Unnamed: 0',axis=1)
data.info()

import seaborn as sns

visual=data.drop(['UNC_YEAR','COUNTRYCD','COUNTRYNM','IncomeGroup','dist','TARIFF','PA','BE','2017_KR_IMPORT','2017_KR_EXPORT'],axis=1)

sns.heatmap(visual.corr(),annot=True,annot_kws={'size':5})

cor1=visual.corr().loc['KR_TRADE_HSCD_COUNTRYCD',:]

cor1
visual.info()
stay=visual['KR_TRADE_HSCD_COUNTRYCD']
visual=visual.drop('KR_TRADE_HSCD_COUNTRYCD',axis=1)
logd=np.log1p(visual)
logd['y']=stay

cor2=logd.corr().loc['y',:]

cor2

for i,j in zip(cor1,cor2):
    print(i, j)
    
    
# log변환을 하면 KR_2017 은 떨어짐 -> 적합할 때, 변환 x
ie3.info()
model_data=ie3.drop(['UNC_YEAR','COUNTRYCD'],axis=1)
model_data.HSCD=model_data.HSCD.astype('object')
model_data=model_data.drop('GDP_DIFF',axis=1)
model_data['GDP_RATIO']=model_data['NY_GDP_MKTP_CD']/model_data['NY_GDP_MKTP_CD_1Y']
model_data['TRADE_RATIO']=model_data['TRADE_HSCD_COUNTRYCD']/model_data['TRADE_HSCD']
model_data=model_data.drop(['2017_KR_IMPORT','2017_KR_EXPORT'],axis=1)

model_data.info()
model_cat=model_data[['dist','TARIFF','PA','BE']]
model_data=model_data.drop(['dist','TARIFF','PA','BE'],axis=1)
model_data=model_data.drop('HSCD',axis=1)

model_target=model_data['KR_TRADE_HSCD_COUNTRYCD']
model_data=model_data.drop('KR_TRADE_HSCD_COUNTRYCD',axis=1)

model_data2=pd.get_dummies(model_data,drop_first=True)
columns=model_data2.columns
model_data2.info()
model_num=model_data2.loc[:,:'TRADE_RATIO']
model_num.shape
model_obj=model_data2.iloc[:,15:]
model_obj.shape


model_num=np.log1p(model_num)
model_target=np.log1p(model_target)

ss=StandardScaler()
ss.fit(model_num)
x_num=ss.transform(model_num)

x=np.concatenate([x_num,model_obj,model_cat],axis=1)
# x=np.concatenate([x_num,model_obj],axis=1)

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

data10['dist']=pd.qcut(data10['SNDIST'],3,labels=[1,3,5])

data10.loc[data10['dist']==1,'dist_avg']=data10[['dist','SNDIST']].groupby('dist').mean().iloc[0,0]
data10.loc[data10['dist']==3,'dist_avg']=data10[['dist','SNDIST']].groupby('dist').mean().iloc[1,0]
data10.loc[data10['dist']==5,'dist_avg']=data10[['dist','SNDIST']].groupby('dist').mean().iloc[2,0]

data10['PA']=pd.qcut(data10['PA_NUS_FCRF'],2,labels=[0,1])
data10[['PA','PA_NUS_FCRF']].groupby('PA').mean()
data10.loc[data10['PA']==0,'PA_AVG']=data10[['PA','PA_NUS_FCRF']].groupby('PA').mean().iloc[0,0]
data10.loc[data10['PA']==1,'PA_AVG']=data10[['PA','PA_NUS_FCRF']].groupby('PA').mean().iloc[1,0]


data10['BE']=pd.qcut(data10['IC_BUS_EASE_DFRN_DB'],3,labels=[1,3,5])
data10[['BE','KR_TRADE_HSCD_COUNTRYCD']].groupby('BE').mean()


data10.loc[data10['BE']==1,'BE_avg']=data10[['BE','KR_TRADE_HSCD_COUNTRYCD']].groupby('BE').mean().iloc[0,0]
data10.loc[data10['BE']==3,'BE_avg']=data10[['BE','KR_TRADE_HSCD_COUNTRYCD']].groupby('BE').mean().iloc[1,0]
data10.loc[data10['BE']==5,'BE_avg']=data10[['BE','KR_TRADE_HSCD_COUNTRYCD']].groupby('BE').mean().iloc[2,0]

data10.info()

except_country=['Guatemala','Viet Nam','Iran','Egypt']

data20=pd.merge(data10,idata,how='left',left_on=['HSCD','COUNTRYNM'],right_on=['HSCD','COUNTRYNM'])
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
            return 0
    else:
        if b!=-99:
            return b
        else: return 0

ie2['KR_2017']=ie2.apply(lambda x:imputate_KR(x['2017_KR_IMPORT'], x['2017_KR_EXPORT'],x['COUNTRYNM']),axis=1)

ie2.info()
ie2.to_csv("data_sungsu.csv")

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
model_cat=model_data[['dist','TARIFF','PA','BE']]
model_data=model_data.drop(['dist','TARIFF','PA','BE'],axis=1)
model_data=model_data.drop('HSCD',axis=1)

model_target=model_data['KR_TRADE_HSCD_COUNTRYCD']
model_data=model_data.drop('KR_TRADE_HSCD_COUNTRYCD',axis=1)

model_data2=pd.get_dummies(model_data,drop_first=True)
columns=model_data2.columns
model_data2.info()
model_num=model_data2.loc[:,:'BE_avg']
model_num.shape
model_obj=model_data2.iloc[:,16:]
model_obj.shape


model_num=np.log1p(model_num)
model_target=np.log1p(model_target)

ss=StandardScaler()
ss.fit(model_num)
x_num=ss.transform(model_num)

x=np.concatenate([x_num,model_obj,model_cat],axis=1)
# x=np.concatenate([x_num,model_obj],axis=1)
x_train,x_test,y_train,y_test=train_test_split(x,model_target,test_size=0.3,random_state=42)

from xgboost import XGBRegressor

xgb=XGBRegressor(random_state=42)
xgb.fit(x_train,y_train)
print(xgb.score(x_train,y_train))

print(xgb.score(x_test,y_test))

## 다른거 범주화해보기.

test=data10[['HSCD','TRADE_HSCD_COUNTRYCD']].groupby('HSCD').mean()


test['Prefer']=pd.qcut(test['TRADE_HSCD_COUNTRYCD'],4,labels=[1,3,5,7])
test.info()
test.head()
test[['Prefer','TRADE_HSCD_COUNTRYCD']].groupby('Prefer').mean()
test=test.reset_index()

data21=pd.merge(data10,test,how='left',left_on='HSCD',right_on='HSCD')

data21.info()

data21=data21.drop('TRADE_HSCD_COUNTRYCD_y',axis=1)

data21['TARIFF']=data21['TARIFF'].astype('category')

data21['TRADE_HSCD_COUNTRYCD']=data21['TRADE_HSCD_COUNTRYCD_x']
data21=data21.drop('TRADE_HSCD_COUNTRYCD_x',axis=1)

data21.TRADE_RATIO.describe()

## 특정 국가가 어느 물품을 어느정도 사는지에 대한 비율 변수 추가.

data21['COUNTRYCD'].unique()

alger=data21[data21['COUNTRYCD']==12]

alger.info()

alger.loc[:,'HSCD_RATIO']=alger['TRADE_HSCD_COUNTRYCD']/alger['TRADE_HSCD_COUNTRYCD'].sum()

temp=[]
for i in data21['COUNTRYCD'].unique():
    temp.append(data21[data21['COUNTRYCD']==i]['TRADE_HSCD_COUNTRYCD'].sum())   
len(temp)

for i,j in zip(data21['COUNTRYCD'].unique(),temp):
    data21.loc[data21['COUNTRYCD']==i,'HSCD_RATIO']=((data21.loc[data21['COUNTRYCD']==i,'TRADE_HSCD_COUNTRYCD'])/j)

data21.info()

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

except_country=['Guatemala','Viet Nam','Iran','Egypt']

data22=pd.merge(data21,idata,how='left',left_on=['HSCD','COUNTRYNM'],right_on=['HSCD','COUNTRYNM'])
data22.info()
data22=pd.merge(data22,edata,how='left',left_on=['HSCD','COUNTRYNM'],right_on=['HSCD','COUNTRYNM'])
data22.info()


ie2=data22
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
            return 0
    else:
        if b!=-99:
            return b
        else: return 0

ie2['KR_2017']=ie2.apply(lambda x:imputate_KR(x['2017_KR_IMPORT'], x['2017_KR_EXPORT'],x['COUNTRYNM']),axis=1)

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
model_cat=model_data[['dist','TARIFF','PA','BE','Prefer']]
model_ratio=model_data[['GDP_RATIO','TRADE_RATIO','HSCD_RATIO']]

model_data=model_data.drop(['dist','TARIFF','PA','BE','Prefer'],axis=1)
model_data=model_data.drop(['GDP_RATIO','TRADE_RATIO','HSCD_RATIO'],axis=1)
model_data=model_data.drop('HSCD',axis=1)

model_target=model_data['KR_TRADE_HSCD_COUNTRYCD']
model_data=model_data.drop('KR_TRADE_HSCD_COUNTRYCD',axis=1)

model_data2=pd.get_dummies(model_data,drop_first=True)
columns=model_data2.columns
columns[0:17]
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

x=np.concatenate([x_num,model_obj,model_cat],axis=1)

# x=np.concatenate([x_num,model_obj],axis=1)
x_train,x_test,y_train,y_test=train_test_split(x,model_target,test_size=0.3,random_state=42)

from xgboost import XGBRegressor

xgb=XGBRegressor(random_state=42)
xgb.fit(x_train,y_train)
print(xgb.score(x_train,y_train))

print(xgb.score(x_test,y_test))

model_data.info()
model_num=model_data.drop(['COUNTRYNM','IncomeGroup','HS','group','FTA'],axis=1)
model_num.info()

data=np.log1p(model_num)

cor1=data.corr().loc['KR_TRADE_HSCD_COUNTRYCD',:]

cor1

model_num.info()
model_ratio=model_num[['RDIST','TRADE_RATIO','HSCD_RATIO','GDP_RATIO']]
model_num=model_num.drop(['RDIST','TRADE_RATIO','HSCD_RATIO','GDP_RATIO'],axis=1)

data=np.log1p(model_num)

data[['RDIST','TRADE_RATIO','HSCD_RATIO','GDP_RATIO']]=model_ratio

data.info()

cor2=data.corr().loc['KR_TRADE_HSCD_COUNTRYCD']

cor2
cor1
for i,j in zip(cor1,cor2):
    print(i, j)















