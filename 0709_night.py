# -*- coding: utf-8 -*-
"""
Created on Fri Jul  9 21:44:45 2021

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
data10['TARIFF']=data10['TARIFF'].astype('category')
data10.info()

## 2021-07-07
#data7=data10

#data7.describe()
#data7.info()

#data7['dist2']=pd.qcut(data7['SNDIST'],3,labels=[1,3,5])

#data7[['dist2','KR_TRADE_HSCD_COUNTRYCD']].groupby('dist2').mean()

#data7['TARIFF']=0
#data7.loc[data7['TARIFF_AVG']>0,'TARIFF']=1
#data7[['TARIFF','KR_TRADE_HSCD_COUNTRYCD']].groupby('TARIFF').mean()

#data7['PA']=pd.qcut(data7['PA_NUS_FCRF'],2,labels=[0,1])
#data7[['PA','KR_TRADE_HSCD_COUNTRYCD']].groupby('PA').mean()

#data7['BE']=pd.qcut(data7['IC_BUS_EASE_DFRN_DB'],3,labels=[1,3,5])
#data7[['BE','KR_TRADE_HSCD_COUNTRYCD']].groupby('BE').mean()

#data7.info()
#data7.TARIFF=data7.TARIFF.astype('category')



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

data20=pd.merge(data10,idata,how='left',left_on=['HSCD','COUNTRYNM'],right_on=['HSCD','COUNTRYNM'])
data20.info()
data20=pd.merge(data20,edata,how='left',left_on=['HSCD','COUNTRYNM'],right_on=['HSCD','COUNTRYNM'])
data20.info()

i=idata[idata['COUNTRYNM']=='Guatemala']

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
ie2.to_csv("data_sungsu_0709.csv")

ie2.info()
ie3=ie2.dropna(axis=0,how='any')
ie3=ie3.reset_index().drop('index',axis=1)
ie3.info()

model_data=ie3.drop(['UNC_YEAR','COUNTRYCD'],axis=1)
model_data.HSCD=model_data.HSCD.astype('object')
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
model_num=model_data2.loc[:,:'KR_2017']
model_num.shape
model_obj=model_data2.iloc[:,15:]
model_obj.shape


model_num=np.log1p(model_num)
model_target=np.log1p(model_target)

ss=StandardScaler()
ss.fit(model_num)
x_num=ss.transform(model_num)

x=np.concatenate([x_num,model_obj,model_cat],axis=1)

data=pd.DataFrame(x)
data.columns=list(columns)+['dist','TARIFF','PA','BE']
data.head()

data['COUNTRYNM']=ie2['COUNTRYNM'] 
data['y']=model_target
## 국가별로 적합 정도를 확인 .

data_c1=data[data['COUNTRYNM']==data['COUNTRYNM'].unique()[0]]
data_c2=data[data['COUNTRYNM']==data['COUNTRYNM'].unique()[1]]
data_c3=data[data['COUNTRYNM']==data['COUNTRYNM'].unique()[2]]
data_c4=data[data['COUNTRYNM']==data['COUNTRYNM'].unique()[3]]
data_c5=data[data['COUNTRYNM']==data['COUNTRYNM'].unique()[4]]
data_c6=data[data['COUNTRYNM']==data['COUNTRYNM'].unique()[5]]
data_c7=data[data['COUNTRYNM']==data['COUNTRYNM'].unique()[6]]
data_c8=data[data['COUNTRYNM']==data['COUNTRYNM'].unique()[7]]
data_c9=data[data['COUNTRYNM']==data['COUNTRYNM'].unique()[8]]
data_c10=data[data['COUNTRYNM']==data['COUNTRYNM'].unique()[9]]
data_c11=data[data['COUNTRYNM']==data['COUNTRYNM'].unique()[10]]
data_c12=data[data['COUNTRYNM']==data['COUNTRYNM'].unique()[11]]
data_c13=data[data['COUNTRYNM']==data['COUNTRYNM'].unique()[12]]
data_c14=data[data['COUNTRYNM']==data['COUNTRYNM'].unique()[13]]
data_c15=data[data['COUNTRYNM']==data['COUNTRYNM'].unique()[14]]
data_c16=data[data['COUNTRYNM']==data['COUNTRYNM'].unique()[15]]
data_c17=data[data['COUNTRYNM']==data['COUNTRYNM'].unique()[16]]
data_c18=data[data['COUNTRYNM']==data['COUNTRYNM'].unique()[17]]
data_c19=data[data['COUNTRYNM']==data['COUNTRYNM'].unique()[18]]
data_c20=data[data['COUNTRYNM']==data['COUNTRYNM'].unique()[19]]
data_c21=data[data['COUNTRYNM']==data['COUNTRYNM'].unique()[20]]
data_c22=data[data['COUNTRYNM']==data['COUNTRYNM'].unique()[21]]
data_c23=data[data['COUNTRYNM']==data['COUNTRYNM'].unique()[22]]
data_c24=data[data['COUNTRYNM']==data['COUNTRYNM'].unique()[23]]
data_c25=data[data['COUNTRYNM']==data['COUNTRYNM'].unique()[24]]
data_c26=data[data['COUNTRYNM']==data['COUNTRYNM'].unique()[25]]
data_c27=data[data['COUNTRYNM']==data['COUNTRYNM'].unique()[26]]
data_c28=data[data['COUNTRYNM']==data['COUNTRYNM'].unique()[27]]
data_c29=data[data['COUNTRYNM']==data['COUNTRYNM'].unique()[28]]
data_c30=data[data['COUNTRYNM']==data['COUNTRYNM'].unique()[29]]
data_c31=data[data['COUNTRYNM']==data['COUNTRYNM'].unique()[30]]
data_c32=data[data['COUNTRYNM']==data['COUNTRYNM'].unique()[31]]
data_c33=data[data['COUNTRYNM']==data['COUNTRYNM'].unique()[32]]
data_c34=data[data['COUNTRYNM']==data['COUNTRYNM'].unique()[33]]
data_c35=data[data['COUNTRYNM']==data['COUNTRYNM'].unique()[34]]
data_c36=data[data['COUNTRYNM']==data['COUNTRYNM'].unique()[35]]
data_c37=data[data['COUNTRYNM']==data['COUNTRYNM'].unique()[36]]
data_c38=data[data['COUNTRYNM']==data['COUNTRYNM'].unique()[37]]
data_c39=data[data['COUNTRYNM']==data['COUNTRYNM'].unique()[38]]
data_c40=data[data['COUNTRYNM']==data['COUNTRYNM'].unique()[39]]
data_c41=data[data['COUNTRYNM']==data['COUNTRYNM'].unique()[40]]
data_c42=data[data['COUNTRYNM']==data['COUNTRYNM'].unique()[41]]
data_c43=data[data['COUNTRYNM']==data['COUNTRYNM'].unique()[42]]

from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
from sklearn.svm import SVR

less=[]
worst=[]
notbad=[]
best=[]
for i in data['COUNTRYNM'].unique():
    model_c=data[data['COUNTRYNM']==i]
    model_c.reset_index().drop('index',axis=1)
    model_c=model_c.drop('COUNTRYNM',axis=1)
    y=model_c['y']
    x=model_c.drop('y',axis=1)
    x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=42)
    xgbr=XGBRegressor(random_state=42)
    xgbr.fit(x_train,y_train)
    print("국가 ",i,"의 train :",xgbr.score(x_train,y_train),"\n")
    print("국가 ",i,"의 test :",xgbr.score(x_test,y_test),"\n")
    print("\n")
    if (xgbr.score(x_test,y_test)<0.6)&(xgbr.score(x_test,y_test)>=0.4):
        less.append(i)
    elif xgbr.score(x_test,y_test)<0.4:
        worst.append(i)
    elif (xgbr.score(x_test,y_test)>=0.6)&(xgbr.score(x_test,y_test)<0.8):
        notbad.append(i)
    else:
        best.append(i)

less
worst
notbad
best

less=[]
worst=[]
best=[]

for i in data['COUNTRYNM'].unique():
    model_c=data[data['COUNTRYNM']==i]
    model_c.reset_index().drop('index',axis=1)
    model_c=model_c.drop('COUNTRYNM',axis=1)
    y=model_c['y']
    x=model_c.drop('y',axis=1)
    x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=42)
    if i in worst_country:
        svm=SVR(kernel='poly',degree=3,C=10,epsilon=0.2)
        svm.fit(x_train,y_train)
        print("국가 ",i,"의 train :",svm.score(x_train,y_train),"\n")
        print("국가 ",i,"의 test :",svm.score(x_test,y_test),"\n")
        print("\n")
        if svm.score(x_test,y_test)<0.6:
            less.append(i)
        if svm.score(x_test,y_test)<0.4:
            worst.append(i)
        if svm.score(x_test,y_test)>0.9:
            best.append(i)
    else:
        xgbr=XGBRegressor(random_state=42)
        xgbr.fit(x_train,y_train)
        print("국가 ",i,"의 train :",xgbr.score(x_train,y_train),"\n")
        print("국가 ",i,"의 test :",xgbr.score(x_test,y_test),"\n")
        print("\n")
        if xgbr.score(x_test,y_test)<0.6:
            less.append(i)
        if xgbr.score(x_test,y_test)<0.4:
            worst.append(i)
        if xgbr.score(x_test,y_test)>0.9:
            best.append(i)

data['COUNTRYNM'].unique()[0] in worst_country

less
worst
best

worst_country=worst
data_alg=data[data['COUNTRYNM']=='Algeria']

data_alg.corr().loc[abs(data_alg.corr().loc['y'])>0.1].index

data_alg.corr().loc[abs(data_alg.corr().loc['y'])>0.1].index



fig=plt.figure(figsize=(20,10))
plt.plot(abs(data_alg.corr().loc['y',data_alg.corr().loc[abs(data_alg.corr().loc['y'])>0.1].index]))
plt.xticks(size=10,rotation='vertical')
plt.show()

ie2.info()

ie2.describe()

ie_alg=ie2[ie2['COUNTRYNM']=='Algeria']



ie_alg=ie2[ie2['COUNTRYNM']=='Algeria']
ie_mya=ie2[ie2['COUNTRYNM']=='Myanmar']
ie_sri=ie2[ie2['COUNTRYNM']=='Sri Lanka']
ie_gua=ie2[ie2['COUNTRYNM']=='Guatemala']
ie_new=ie2[ie2['COUNTRYNM']=='New Zealand']
ie_egy=ie2[ie2['COUNTRYNM']=='Egypt']

ie_alg.corr().loc['KR_TRADE_HSCD_COUNTRYCD']
ie_mya.corr().loc['KR_TRADE_HSCD_COUNTRYCD']
ie_sri.corr().loc['KR_TRADE_HSCD_COUNTRYCD']
ie_gua.corr().loc['KR_TRADE_HSCD_COUNTRYCD']
ie_new.corr().loc['KR_TRADE_HSCD_COUNTRYCD']
ie_egy.corr().loc['KR_TRADE_HSCD_COUNTRYCD']


l_country=less
w_country=worst
n_country=notbad
b_country=best


less=[]
worst=[]
notbad=[]
best=[]


for i in data['COUNTRYNM'].unique():
    model_c=data[data['COUNTRYNM']==i]
    model_c.reset_index().drop('index',axis=1)
    model_c=model_c.drop('COUNTRYNM',axis=1)
    y=model_c['y']
    x=model_c.drop('y',axis=1)
    x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=42)
    if i in w_country:
        xgbr=XGBRegressor(learning_rate=0.001,n_estimators=1000,random_state=42)
        xgbr.fit(x_train,y_train)
        print("국가 ",i,"의 train :",xgbr.score(x_train,y_train),"\n")
        print("국가 ",i,"의 test :",xgbr.score(x_test,y_test),"\n")
        print("\n")
        if (xgbr.score(x_test,y_test)<0.6)&(xgbr.score(x_test,y_test)>=0.4):
            less.append(i)
        elif xgbr.score(x_test,y_test)<0.4:
            worst.append(i)
        elif (xgbr.score(x_test,y_test)>=0.6)&(xgbr.score(x_test,y_test)<0.8):
            notbad.append(i)
        else:
            best.append(i)
    elif i in l_country:
        xgbr=XGBRegressor(learning_rate=0.01,n_estimators=800,random_state=42)
        xgbr.fit(x_train,y_train)
        print("국가 ",i,"의 train :",xgbr.score(x_train,y_train),"\n")
        print("국가 ",i,"의 test :",xgbr.score(x_test,y_test),"\n")
        print("\n")
        if (xgbr.score(x_test,y_test)<0.6)&(xgbr.score(x_test,y_test)>=0.4):
            less.append(i)
        elif xgbr.score(x_test,y_test)<0.4:
            worst.append(i)
        elif (xgbr.score(x_test,y_test)>=0.6)&(xgbr.score(x_test,y_test)<0.8):
            notbad.append(i)
        else:
            best.append(i)

    elif i in n_country:
        xgbr=XGBRegressor(learning_rate=0.15,n_estimators=1000,random_state=42)
        xgbr.fit(x_train,y_train)
        print("국가 ",i,"의 train :",xgbr.score(x_train,y_train),"\n")
        print("국가 ",i,"의 test :",xgbr.score(x_test,y_test),"\n")
        print("\n")
        if (xgbr.score(x_test,y_test)<0.6)&(xgbr.score(x_test,y_test)>=0.4):
            less.append(i)
        elif xgbr.score(x_test,y_test)<0.4:
            worst.append(i)
        elif (xgbr.score(x_test,y_test)>=0.6)&(xgbr.score(x_test,y_test)<0.8):
            notbad.append(i)
        else:
            best.append(i)

    else:
        xgbr=XGBRegressor(learning_rate=0.2,n_estimators=200,random_state=42)
        xgbr.fit(x_train,y_train)
        print("국가 ",i,"의 train :",xgbr.score(x_train,y_train),"\n")
        print("국가 ",i,"의 test :",xgbr.score(x_test,y_test),"\n")
        print("\n")
        if (xgbr.score(x_test,y_test)<0.6)&(xgbr.score(x_test,y_test)>=0.4):
            less.append(i)
        elif xgbr.score(x_test,y_test)<0.4:
            worst.append(i)
        elif (xgbr.score(x_test,y_test)>=0.6)&(xgbr.score(x_test,y_test)<0.8):
            notbad.append(i)
        else:
            best.append(i)
            


less
worst

from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import GridSearchCV
from scipy.stats import uniform, randint

w_country



xgb_params={'learning_rate':uniform(0.001,0.1),
            'n_estimators':randint(1,300),
            'lambda':uniform(2,4)}

model_c=data[data['COUNTRYNM']=='Algeria']
model_c.reset_index().drop('index',axis=1)
model_c=model_c.drop('COUNTRYNM',axis=1)
y=model_c['y']
x=model_c.drop('y',axis=1)
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=42)


gs=RandomizedSearchCV(XGBRegressor(random_state=42),xgb_params,n_iter=100,n_jobs=-1,random_state=42)
gs.fit(x_train,y_train)
dt=gs.best_estimator_
gs.best_params_
print(dt.score(x_train,y_train))
print(dt.score(x_test,y_test))


xgbr=XGBRegressor(learning_rate=0.001,n_estimators=1000,random_state=42)
xgbr.fit(x_train,y_train)
xgbr.score(x_train,y_train)
xgbr.score(x_test,y_test)

len(data[data['y']==0])


tryto=data[(data['y']!=0)&data['KR_2017']!=0]
tryto.info()

tryto=tryto.reset_index().drop('index',axis=1)



less=[]
worst=[]
notbad=[]
best=[]
for i in tryto['COUNTRYNM'].unique():
    model_c=tryto[tryto['COUNTRYNM']==i]
    model_c.reset_index().drop('index',axis=1)
    model_c=model_c.drop('COUNTRYNM',axis=1)
    y=model_c['y']
    x=model_c.drop('y',axis=1)
    x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=42)
    xgbr=XGBRegressor(learning_rate=0.04,random_state=42)
    xgbr.fit(x_train,y_train)
    print("국가 ",i,"의 train :",xgbr.score(x_train,y_train),"\n")
    print("국가 ",i,"의 test :",xgbr.score(x_test,y_test),"\n")
    print("\n")
    if (xgbr.score(x_test,y_test)<0.6)&(xgbr.score(x_test,y_test)>=0.4):
        less.append(i)
    elif xgbr.score(x_test,y_test)<0.4:
        worst.append(i)
    elif (xgbr.score(x_test,y_test)>=0.6)&(xgbr.score(x_test,y_test)<0.8):
        notbad.append(i)
    else:
        best.append(i)
        

worst_5=worst
less_5=less

notbad_5=notbad

best_5=best

try_alg=tryto[tryto['COUNTRYNM']=='Algeria']
try_mya=tryto[tryto['COUNTRYNM']=='Myanmar']
try_sri=tryto[tryto['COUNTRYNM']=='Sri Lanka']
try_gua=tryto[tryto['COUNTRYNM']=='Guatemala']
try_new=tryto[tryto['COUNTRYNM']=='New Zealand']
try_egy=tryto[tryto['COUNTRYNM']=='Egypt']



try_alg.corr().loc['y']
try_mya.corr().loc['y']
try_sri.corr().loc['y']
try_gua.corr().loc['y']
try_new.corr().loc['y']
try_egy.corr().loc['y']


fig=plt.figure(figsize=(20,10))
plt.plot(abs(try_alg.corr().loc['y',try_alg.corr().loc[abs(try_alg.corr().loc['y'])>0.1].index]))
plt.xticks(size=10,rotation='vertical')
plt.show()

plz=ie2[ie2['KR_TRADE_HSCD_COUNTRYCD']==0]
plz=ie2.loc[ie2['KR_2017']==0,['COUNTRYNM','HSCD','KR_TRADE_HSCD_COUNTRYCD','2017_KR_IMPORT','2017_KR_EXPORT','KR_2017']]
plz2=ie2.loc[ie2['KR_TRADE_HSCD_COUNTRYCD']==0,['COUNTRYNM','HSCD','KR_TRADE_HSCD_COUNTRYCD','2017_KR_IMPORT','2017_KR_EXPORT','KR_2017']]
w_country
l_country
worst
less

plz.COUNTRYNM.value_counts()
plz2.COUNTRYNM.value_counts()

worst

less

notbad

best

worst_2
worst_3
worst_4
worst_5

less_2
less_3
less_4
less_5

## 알제리, 미얀마, 스리랑카, 이집트 이란, 노르웨이, 뉴질랜드

## 위 국가들의 잔차의 경향이 있을까?

tryto.info()
tryto3=tryto

tryto2=tryto[tryto['COUNTRYNM'].isin(['Algeria','Sri Lanka','Iran','Myanmar','Egypt','Norway','New Zealand'])]

tryto2.info()
tryto2=tryto2.reset_index().drop('index',axis=1)

for i in tryto2.COUNTRYNM.unique():
    model_c=tryto2[tryto2['COUNTRYNM']==i]
    model_c=model_c.reset_index().drop('index',axis=1)
    model_c=model_c.drop('COUNTRYNM',axis=1)
    y=model_c['y']
    x=model_c.drop('y',axis=1)
    x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=42)
    xgbr=XGBRegressor(learning_rate=0.04,random_state=42)
    xgbr.fit(x_train,y_train)
    y_pred=xgbr.predict(x)
    tryto2.loc[tryto2['COUNTRYNM']==i,'pred']=y_pred

tryto2['res']=tryto2['y']-tryto2['pred']

alg=tryto2.loc[tryto2['COUNTRYNM']=='Algeria',['y','pred','res']]
alg.plot(kind='scatter',x='y',y='res')

sri=tryto2.loc[tryto2['COUNTRYNM']=='Sri Lanka',['y','pred','res']]
sri.plot(kind='scatter',x='y',y='res')

iran=tryto2.loc[tryto2['COUNTRYNM']=='Iran',['y','pred','res']]
iran.plot(kind='scatter',x='y',y='res')

mya=tryto2.loc[tryto2['COUNTRYNM']=='Myanmar',['y','pred','res']]
mya.plot(kind='scatter',x='y',y='res')

egy=tryto2.loc[tryto2['COUNTRYNM']=='Egypt',['y','pred','res']]
egy.plot(kind='scatter',x='y',y='res')

nor=tryto2.loc[tryto2['COUNTRYNM']=='Norway',['y','pred','res']]
nor.plot(kind='scatter',x='y',y='res')

new=tryto2.loc[tryto2['COUNTRYNM']=='New Zealand',['y','pred','res']]
new.plot(kind='scatter',x='y',y='res')

## 선형성을 띄고 있음. 
tryto2.columns
tryto2[tryto2['res']>4]

# 2021 07 10 비율 추가

ie3.info()

data=ie3.drop(['UNC_YEAR','COUNTRYNM'],axis=1)

data.head()
data.info()

# 내가 고려했던 TRADE_RATIO는 TRADE_HSCD_COUNTRYCD/TRADE_HSCD로 
# 해당 연도 해당 국가의 해당 품목 수입금액 /해당 연도 해당 품목의 전세계 총 수입금액


# 민섭이 1번
data['HSCD_RATIO']=data['TRADE_HSCD']/data['TRADE_HSCD'].unique().sum()
# 민섭이 2번
data['COUNTRY_RATIO']=data['TRADE_COUNTRYCD']/data['TRADE_COUNTRYCD'].unique().sum()

# 민섭이 3번(국가별로 해야함)

for i in data['COUNTRYCD'].unique():
    temp=data.loc[data['COUNTRYCD']==i,['TRADE_HSCD_COUNTRYCD','TRADE_COUNTRYCD']]
    temp['temp']=temp['TRADE_HSCD_COUNTRYCD']/temp['TRADE_COUNTRYCD'].unique()[0]
    data.loc[data['COUNTRYCD']==i,'TRADE_HSCD_RATIO']=temp['temp']    

# 민섭이 4번 : KR_2017/TRADE_HSCD

for i in data['HSCD'].unique():
    temp=data.loc[data['HSCD']==i,['KR_2017','TRADE_HSCD']]
    temp['temp']=temp['KR_2017'].sum()/temp['TRADE_HSCD'].unique()[0]
    data.loc[data['HSCD']==i,'KR_HSCD_RATIO']=temp['temp']

data.info()

# 민섭이 5번 : KR_2017 의 합 / TRADE_COUNTRYCD 
i=data['COUNTRYCD'].unique()[0]
for i in data['COUNTRYCD'].unique():
    temp=data.loc[data['COUNTRYCD']==i,['KR_2017','TRADE_COUNTRYCD']]
    temp['temp']=temp['KR_2017'].sum()/temp['TRADE_COUNTRYCD'].unique()[0]
    data.loc[data['COUNTRYCD']==i,'KR_COUNTRY_RATIO']=temp['temp']

# 민섭이 6번 : KR_2017/TRADE_HSCD_COUNTRYCD

data['KR_HSCD_COUNTRY_RATIO']=data['KR_2017']/data['TRADE_HSCD_COUNTRYCD']


# 민섭이 7번 : 있는 데이터만 고려하지만, 일단해봄. (4,5번도 포함..)
# KR_2017/KR_2017합
for i in data['COUNTRYCD'].unique():
    temp=data.loc[data['COUNTRYCD']==i,['KR_2017']]
    temp['temp']=temp['KR_2017']/temp['KR_2017'].sum()
    data.loc[data['COUNTRYCD']==i,'KR_RATIO']=temp['temp']

data.info()

model_data=data.drop('COUNTRYCD',axis=1)
model_data.HSCD=model_data.HSCD.astype('object')
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
model_num=model_data2.iloc[:,:21]
model_num.shape
model_obj=model_data2.iloc[:,21:]
model_obj.shape


model_num=np.log1p(model_num)
model_target=np.log1p(model_target)

x=np.concatenate([model_num,model_obj,model_cat],axis=1)

d10=pd.DataFrame(x)
d10.columns=list(columns)+['dist','TARIFF','PA','BE']
d10.head()

d10['COUNTRYNM']=ie2['COUNTRYNM'] 
d10['y']=model_target
d10.loc[(d10['KR_2017']==0),'Have_zero']=1
d10.loc[(d10['KR_2017']!=0),'Have_zero']=0
d10.info()


less=[]
worst=[]
notbad=[]
best=[]


for i in d10['COUNTRYNM'].unique():
    model_c=d10[d10['COUNTRYNM']==i]
    model_c.reset_index().drop('index',axis=1)
    model_c=model_c.drop('COUNTRYNM',axis=1)
    y=model_c['y']
    x=model_c.drop('y',axis=1)
    name=x.columns
    x_num=x.iloc[:,:21]
    x=x.iloc[:,21:]
    ss=StandardScaler()
    ss.fit(x_num)
    x_num=ss.transform(x_num)
    x=np.concatenate([x_num,x],axis=1)
    x=pd.DataFrame(x)
    x.columns=name
    x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=42)
    xgbr=XGBRegressor(random_state=42)
    xgbr.fit(x_train,y_train)
    print("국가 ",i,"의 train :",xgbr.score(x_train,y_train),"\n")
    print("국가 ",i,"의 test :",xgbr.score(x_test,y_test),"\n")
    print("\n")
    if (xgbr.score(x_test,y_test)<0.6)&(xgbr.score(x_test,y_test)>=0.4):
        less.append(i)
    elif xgbr.score(x_test,y_test)<0.4:
        worst.append(i)
    elif (xgbr.score(x_test,y_test)>=0.6)&(xgbr.score(x_test,y_test)<0.8):
        notbad.append(i)
    else:
        best.append(i)


less
worst

notbad
best

less0_2=[]
worst0_2=[]
notbad0_2=[]
best0_2=[]


d11=d10[(d10['y']!=0)&(d10['KR_2017']!=0)]

d11.info()
d11=d11.reset_index().drop('index',axis=1)


for i in d11['COUNTRYNM'].unique():
    model_c=d11[d11['COUNTRYNM']==i]
    model_c.reset_index().drop('index',axis=1)
    model_c=model_c.drop('COUNTRYNM',axis=1)
    y=model_c['y']
    x=model_c.drop('y',axis=1)
    name=x.columns
    x_num=x.iloc[:,:21]
    x=x.iloc[:,21:]
    ss=StandardScaler()
    ss.fit(x_num)
    x_num=ss.transform(x_num)
    x=np.concatenate([x_num,x],axis=1)
    x=pd.DataFrame(x)
    x.columns=name
    x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=42)
    xgbr=XGBRegressor(learning_rate=0.4,random_state=42)
    xgbr.fit(x_train,y_train)
    print("국가 ",i,"의 train :",xgbr.score(x_train,y_train),"\n")
    print("국가 ",i,"의 test :",xgbr.score(x_test,y_test),"\n")
    print("\n")
    if (xgbr.score(x_test,y_test)<0.6)&(xgbr.score(x_test,y_test)>=0.4):
        less0_2.append(i)
    elif xgbr.score(x_test,y_test)<0.4:
        worst0_2.append(i)
    elif (xgbr.score(x_test,y_test)>=0.6)&(xgbr.score(x_test,y_test)<0.8):
        notbad0_2.append(i)
    else:
        best0_2.append(i)
 
       
 
   
worst0        
less0
notbad0
best0

worst0_2        
less0_2
notbad0_2
best0_2

worst_g0=[]
less_g0=[]
notbad_g0=[]
best_g0=[]


for i in d10['COUNTRYNM'].unique():
    model_c=d10[d10['COUNTRYNM']==i]
    model_c.reset_index().drop('index',axis=1)
    model_c=model_c.drop('COUNTRYNM',axis=1)
    y=model_c['y']
    x=model_c.drop('y',axis=1)
    name=x.columns
    x_num=x.iloc[:,:21]
    x=x.iloc[:,21:]
    ss=StandardScaler()
    ss.fit(x_num)
    x_num=ss.transform(x_num)
    x=np.concatenate([x_num,x],axis=1)
    x=pd.DataFrame(x)
    x.columns=name
    x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=42)
    xgbr=XGBRegressor(learning_rate=0.04,random_state=42)
    xgbr.fit(x_train,y_train)
    print("국가 ",i,"의 train :",xgbr.score(x_train,y_train),"\n")
    print("국가 ",i,"의 test :",xgbr.score(x_test,y_test),"\n")
    print("\n")
    if (xgbr.score(x_test,y_test)<0.6)&(xgbr.score(x_test,y_test)>=0.4):
        less_g0.append(i)
    elif xgbr.score(x_test,y_test)<0.4:
        worst_g0.append(i)
    elif (xgbr.score(x_test,y_test)>=0.6)&(xgbr.score(x_test,y_test)<0.8):
        notbad_g0.append(i)
    else:
        best_g0.append(i)



worst_g0
less_g0
notbad_g0
best_g0

d100=pd.get_dummies(d10,drop_first=True)
d100.info()
d10.info()

model_c=d100
model_c=model_c.reset_index().drop('index',axis=1)
model_c=model_c.drop('COUNTRYNM',axis=1)
y=model_c['y']
x=model_c.drop('y',axis=1)
name=x.columns
x_num=x.iloc[:,:21]
x=x.iloc[:,21:]
ss=StandardScaler()
ss.fit(x_num)
x_num=ss.transform(x_num)
x=np.concatenate([x_num,x],axis=1)
x=pd.DataFrame(x)
x.columns=name

x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=42)


from sklearn.linear_model import LinearRegression
from sklearn import preprocessing

lr=LinearRegression()
lr.fit(x_train,y_train)
lr.score(x_train,y_train)
lr.score(x_test,y_test)

from sklearn.svm import SVR
svr_clf=SVR(kernel='rbf')
svr_clf.fit(x_train,y_train)
print(svr_clf.score(x_train,y_train))
print(svr_clf.score(x_test,y_test))

from lightgbm import LGBMRegressor
lgb=LGBMRegressor(random_state=42)
lgb.fit(x_train,y_train)
print(lgb.score(x_train,y_train))
print(lgb.score(x_test,y_test))

xgbr=XGBRegressor(random_state=42)
xgbr.fit(x_train,y_train)

xgbr.score(x_train,y_train)
xgbr.score(x_test,y_test)

pred=xgbr.predict(x)


data.info()

data['KR_HSCD_COUNTRY_RATIO']

## 비율로 한다면?? 지금 현재 2017년도 데이터가 다 존재함.
## 이걸로 2018년을 예측한다? 불가능 혹시나 2017 데이터 

data.head()

pred=np.exp(pred)-1

data['pred']=pred

data.head()
check=data[['KR_2017','KR_TRADE_HSCD_COUNTRYCD']]
check=np.log1p(check)
check['HSCD']=data['HSCD']
check['COUNTRYCD']=data['COUNTRYCD']
check['pred']=pred
check.info()

check.head()

check['res']=check['KR_TRADE_HSCD_COUNTRYCD']-check['pred']

check.describe()


res=check[abs(check['res'])>4]

res['COUNTRYNM'].value_counts()

## 과테말라 49개 알제리아 37개 

gua_res=res[res['COUNTRYNM']=='Guatemala']
ind=list(gua_res.HSCD.unique())


d=i[i['HSCD'].isin(ind)==True]
d=d[['HSCD','2017_KR_IMPORT']]
d['2017_KR_IMPORT']=np.log1p(d['2017_KR_IMPORT'])

d
res_gua=pd.merge(gua_res,d,how='left',left_on='HSCD',right_on='HSCD')

res_gua

res_gua[['KR_2017','KR_TRADE_HSCD_COUNTRYCD','HSCD','pred','res','2017_KR_IMPORT']]





















