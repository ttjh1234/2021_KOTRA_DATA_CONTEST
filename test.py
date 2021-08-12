# -*- coding: utf-8 -*-
"""
Created on Sun Jun 27 16:42:33 2021

@author: 82104
"""
# 코트라 분석 과제 데이터 초기 분석 #
# 최성수

import pandas as pd
import numpy as np
import os
from sklearn.preprocessing import StandardScaler
import sklearn
import matplotlib.pyplot as plt
import seaborn as sns

pd.set_option('display.max_columns',17)

os.chdir("C:/Users/82104/Desktop/공모전1")
data=pd.read_csv("./공모전데이터_분석용_KOTRA_0525.csv")
data.describe()
data.head()

data.info()
# 관세 평균 129개 누락
# 수입 국가 간 평균 거리 22개 누락 
# 공식 환율 3400개쯤 누락 -> unique 값으로 대치 가능
# 해당 연도 해당 국가의 해당 품목 수입금액 21개 누락


data.HSCD.unique().shape
# 500 개
data.COUNTRYCD.unique().shape
# 43 개
data.COUNTRYNM.unique().shape
# 43 개
data.shape
# 21189개의 데이터 + 16개의 특성을 가짐. 

data2=data.loc[data.loc[:,:].isnull().sum(axis=1)>0,:]
data2.shape
# 3606개의 데이터
data2.head()

## 물품 항목별로 데이터 나누기
data.HSCD.unique().shape
# 항목이 500개 존재하므로 국가별로 나누는 것이 명확.

## 물품 국가별로 나눔
data.COUNTRYCD.unique()

# test 데이터로 국가 번호 12 알제리아만 추출
data_12=data.loc[data.COUNTRYCD==12,:]

data_12.shape
data_12.SNDIST.describe()
# 404 16개
# 이를 통해 국가 별로 수입품목이 다름.

test=data_12.loc[data_12.loc[:,:].isnull().sum(axis=1)>0,:]

test.shape
data_12.head()

test

data_36=data.loc[data.COUNTRYCD==36,:]

data_36.shape
# 500 16개
# 이를 통해 국가 별로 수입품목이 다름.

test2=data_36.loc[data_36.loc[:,:].isnull().sum(axis=1)>0,:]

test2.shape
data_12.head()

test2

# 관세 평균과 거리 평균에서 누락값이 존재
# 거리 평균은 평균으로 대치할 수 있을 거 같음.
# 기초 통계량 뽑아보기
data_36.SNDIST.describe()
# 최소값 8532 최대 13886 
# 평균 12081 중앙 12134
# 최소와 최대의 차이가 왜 이렇게 큰지는 확인 불가능.
# 누락값 중 공식 환율이 있다면 어떻게 처리?
data_12.PA_NUS_FCRF.unique()
data_36.PA_NUS_FCRF.unique()

# 공식 환율은 모든 품목 중 한 개에만 기입 되어있으면 그 값으로 대치 가능.

# 수출액으로 품목 별 그룹화가 가능할까? 
item=data.HSCD.unique()

item[0]
data.loc[data.HSCD==item[2],'KR_TRADE_HSCD_COUNTRYCD'].sum(axis=0)
# 품목 별 한국의 수출량을 보자.
type(item)
testlist=[]
i=item[0]
for i in item:
    testlist.append(data.loc[data.HSCD==i,'KR_TRADE_HSCD_COUNTRYCD'].sum(axis=0))

itemdata=pd.DataFrame(testlist)
itemdata.index=item
itemdata.columns=['total']
itemdata

itemdata.total.describe()

itemdata2=itemdata.sort_values(by='total')

import seaborn as sns

sns.barplot(x=itemdata2.index,y='total',data=itemdata2)
itemdata2['log_total']=np.log(itemdata2['total'])

itemdata2.head()

sns.barplot(x=itemdata2.index,y='log_total',data=itemdata2)

itemdata2['total'].describe()

item_low=itemdata2.iloc[:125,:]
item_mid=itemdata2.iloc[125:250,:]
item_high=itemdata2.iloc[250:375,:]
item_max=itemdata2.iloc[375:,:]

sns.barplot(x=item_low.index,y='total',data=item_low)
itemdata2.iloc[:50,:]

import matplotlib.pyplot as plt

fig=plt.figure(figsize=(20,5))
ax1=fig.add_subplot(2,2,1)
ax2=fig.add_subplot(2,2,2)
ax3=fig.add_subplot(2,2,3)
ax4=fig.add_subplot(2,2,4)

sns.barplot(x=item_low.index,y='total',data=item_low,ax=ax1)
sns.barplot(x=item_mid.index,y='total',data=item_mid,ax=ax2)
sns.barplot(x=item_high.index,y='total',data=item_high,ax=ax3)
sns.barplot(x=item_max.index,y='total',data=item_max,ax=ax4)

ax1.set_title('low')
ax1.set_title('mid')
ax1.set_title('high')
ax1.set_title('max')

plt.show()

# max 중 다른 품목 보다 굉장히 큰 한 품목이 존재
# 그 품목? => 854232 품목
cate_list= []
for i in range(len(data["HSCD"])):
    cate_list.append(str(data["HSCD"][i])[:4])

data["category"]=cate_list

large_cate_list= []
for i in range(len(data["UNC_YEAR"])):
    large_cate_list.append(str(data["HSCD"][i])[:2])

data["large_cate"] = large_cate_list

data["category"].unique().shape
data["large_cate"].unique().shape

item2=data["category"].unique()


test2=[]
test3=[]
for i in item2:
    test2.append(data.loc[data.category==i,'KR_TRADE_HSCD_COUNTRYCD'].sum(axis=0))
    test3.append(data.loc[data.category==i,'KR_TRADE_HSCD_COUNTRYCD'].mean(axis=0))
itemdata2=pd.DataFrame(test2)
itemdata2['mean']=test3
itemdata2.index=item2
itemdata2.columns=['sumt','meant']
itemdata2.sumt
itemdata2.index=item2.astype(int)

fig=plt.figure(figsize=(20,5))
ax1=fig.add_subplot(1,2,1)
ax2=fig.add_subplot(1,2,2)
sns.barplot(x=itemdata2.index,y='sumt',data=itemdata2,ax=ax1)
sns.barplot(x=itemdata2.index,y='meant',data=itemdata2,ax=ax2)
item3=itemdata2.sort_values(by='meant',ascending=False)
plt.close()

fig=plt.figure(figsize=(20,5))
ax1=fig.add_subplot(1,2,1)
ax2=fig.add_subplot(1,2,2)
sns.barplot(x=item3.index[0:25],y=item3.sumt[0:25],ax=ax1)
sns.barplot(x=itemdata2.index[0:25],y=item3.meant[0:25],ax=ax2)
ax1.set_xticklabels(item3.index[0:25],rotation=90)
ax2.set_xticklabels(item3.index[0:25],rotation=90)
plt.close()

# 1~24 25~40 41~83 나머지
data.info()
data.large_cate=data.large_cate.astype('int64')
#mask1=sum(data.large_cate<=24)
#mask2=sum((data.large_cate>24) & (data.large_cate<=40))
#mask3=sum((data.large_cate>40) & (data.large_cate<=83))
#mask4=sum(data.large_cate>83)

mask1=data.large_cate<=24
mask2=(data.large_cate>24) & (data.large_cate<=40)
mask3=(data.large_cate>40) & (data.large_cate<=83)
mask4=data.large_cate>83
group_1=data[mask1]
group_2=data[mask2]
group_3=data[mask3]
group_4=data[mask4]

group_1.shape
group_2.shape
group_3.shape
group_4.shape

diff_mean=[]
diff_sum=[]
diff_mean.append(group_1.KR_TRADE_HSCD_COUNTRYCD.mean())
diff_mean.append(group_2.KR_TRADE_HSCD_COUNTRYCD.mean())
diff_mean.append(group_3.KR_TRADE_HSCD_COUNTRYCD.mean())
diff_mean.append(group_4.KR_TRADE_HSCD_COUNTRYCD.mean())

diff_sum.append(group_1.KR_TRADE_HSCD_COUNTRYCD.sum())
diff_sum.append(group_2.KR_TRADE_HSCD_COUNTRYCD.sum())
diff_sum.append(group_3.KR_TRADE_HSCD_COUNTRYCD.sum())
diff_sum.append(group_4.KR_TRADE_HSCD_COUNTRYCD.sum())

plt.plot(diff_sum)
plt.plot(diff_mean)

group_1.describe()
group_2.describe()
group_3.describe()
group_4.describe()


## 2021-06-29 누락값 데이터 확인 및 처리
data.info()
환율=data[data['PA_NUS_FCRF'].isnull()==True]
환율.shape
환율.COUNTRYCD.unique()
환율.COUNTRYNM.unique()
환율_40=환율[환율.COUNTRYCD==40]
환율_40.shape
환율_40.head()
data_40=data[data.COUNTRYCD==40]
data_40.shape
# 40번 Austria는 499개 중 환율이 있는 것이 없다.
환율_56=환율[환율.COUNTRYCD==56]
환율_56.shape
환율_56.head()
data_56=data[data.COUNTRYCD==56]
data_56.shape
# 56번 Belgium도 493개 중 환율이 있는 것이 없다.
환율_251=환율[환율.COUNTRYCD==251]
환율_251.shape
환율_251.head()
data_251=data[data.COUNTRYCD==251]
data_251.shape
#251번 France도 499개 중 환율이 있는 것이 없다.
환율_276=환율[환율.COUNTRYCD==276]
환율_276.shape
환율_276.head()
data_276=data[data.COUNTRYCD==276]
data_276.shape
#276번 Germany도 499개 중 환율이 있는 것이 없다.
환율_381=환율[환율.COUNTRYCD==381]
환율_381.shape
환율_381.head()
data_381=data[data.COUNTRYCD==381]
data_381.shape
#381번 Italy도 499개 중 환율이 있는 것이 없다.
환율_528=환율[환율.COUNTRYCD==528]
환율_528.shape
환율_528.head()
data_528=data[data.COUNTRYCD==528]
data_528.shape
#528번 Netherlands도 499개 중 환율이 있는 것이 없다.
환율_724=환율[환율.COUNTRYCD==724]
환율_724.shape
환율_724.head()
data_724=data[data.COUNTRYCD==724]
data_724.shape
#724번 Spain도 499개 중 환율이 있는 것이 없다.

# 환율은 외부 데이터를 참조하여 사용.
data.COUNTRYNM.unique()
data[data.COUNTRYCD==12].head()

# 누락 국가들은 모두 유로를 사용하는 국가들로 
# 코트라에서 제공되는 확정 자료 2017년 기준 1.13 2018년 기준 1.18을 대입해서 사용
data['PA_NUS_FCRF'].fillna(1.13,inplace=True)
data.info()

# 관세 평균이 누락된 데이터 확인하기.
관세=data[data['TARIFF_AVG'].isnull()==True]
관세.shape
관세.COUNTRYCD.unique()
len(관세.COUNTRYCD.unique())
# 43개 모든 국가가 특정 물품에 대해 관세 평균이 누락된 것이 있다.
관세.HSCD.unique()
len(관세.HSCD.unique())
# 총 10개의 물품에 대해 관세가 없는 것이 존재
관세[관세.HSCD==420232].shape
data[data.HSCD==420232].shape
# 420232 물품은 이란만 존재하지않음.
관세[관세.HSCD==999999].shape
# 999999 미분류? 물품에 대해 36개 국가가 존재하지않음. 
data[data.HSCD==999999].shape
# 원본 데이터에서도 999999 품목은 36개가 존재, 고로 이 물품에대한 관세를 특정할 수 없다.
관세[관세.HSCD==903300].shape
# 903300 물품에 대해 뉴질랜드만 존재하지않음.
유로국가=환율.COUNTRYNM.unique()
mask_10=(data.HSCD==903300) & (data.COUNTRYNM.isin(유로국가))
data[mask_10].shape
# 유로 국가들은 환율과 관세평균이 같다.
관세[관세.HSCD==903290]
# 뉴질랜드만 존재하지않음. 903290 물품.
data[data.HSCD==903290].shape
관세[관세.HSCD==903190]
data[data.HSCD==903190].shape
# 뉴질랜드만 존재하지않음. 903190
관세[관세.HSCD==902690]
data[data.HSCD==902690].shape
# 뉴질랜드만 존재x 902690
관세[관세.HSCD==851690]
data[data.HSCD==851690].shape
# 뉴질랜드만 존재x 851690
관세[관세.HSCD==847982]
data[data.HSCD==847982].shape
# 이집트만 존재x 847982
관세[관세.HSCD==382499].shape
data[data.HSCD==382499].shape
# 모든 국가가 382499에 대해 관세가 존재x
관세[관세.HSCD==852852].shape
data[data.HSCD==852852].shape
# 모든 국가가 852852에 대해 관세가 존재x

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

data7.info()


## KR_TRADE_HSCD_COUNTRYCD 빼고 모두 대치 완료.
# 파생변수로 KMDIST/SNDIST 사용
data7['RDIST']=data7['KMDIST']/data7['SNDIST']
data7.info()
data7.describe()
pred=data7[data7.TRADE_HSCD_COUNTRYCD.isnull()==True]
play=data7[data7.TRADE_HSCD_COUNTRYCD.isnull()==False]

play=play.drop(['UNC_YEAR','COUNTRYCD','COUNTRYNM','KMDIST','RDIST'],axis=1)
play=play.drop('HSCD',axis=1)

play.info()
play=play.drop('KR_TRADE_HSCD_COUNTRYCD',axis=1)
y=play['TRADE_HSCD_COUNTRYCD']
play=play.drop('TRADE_HSCD_COUNTRYCD',axis=1)
from sklearn.linear_model import LinearRegression
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures
play2=pd.get_dummies(play,drop_first=True)
y=np.log1p(y)

play2.info()
#x_num=play2.loc[:,:'GDP_DIFF']
#onehot_encoder=preprocessing.OneHotEncoder()
#group=onehot_encoder.fit_transform(np.array(play['IncomeGroup']).reshape(1,-1))
#play=play.drop('IncomeGroup',axis=1)
#play['Group']=group
#onehot_encoder2=preprocessing.OneHotEncoder()
#HS_2=onehot_encoder.fit_transform(np.array(play['HS']).reshape(1,-1))
#play=play.drop('HS',axis=1)
#play['HS2']=HS_2
#play.describe()
play2.info()
play.dropna(inplace=True)
#x=np.array(play2)
#y=np.array(y)
#x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=42)

poly=PolynomialFeatures(degree=2,include_bias=False)
poly.fit(x_train)
train_poly=poly.transform(x_train)
test_poly=poly.transform(x_test)

ss=StandardScaler()
ss.fit(train_poly)
x_train=ss.transform(train_poly)
x_test=ss.transform(test_poly)

lr=LinearRegression()
lr.fit(x_train,y_train)
print(lr.score(x_train,y_train))
print(lr.score(x_test,y_test))

from sklearn.linear_model import Lasso
lasso=Lasso()
lasso.fit(x_train,y_train)
lasso.coef_
print(lasso.score(x_train,y_train))

play2.info()
x_num=play2.loc[:,:'GDP_DIFF']
x_cat=play2.loc[:,'IncomeGroup_Lower middle income':]

ss=StandardScaler()
ss.fit(x_num)
x_num2=ss.transform(x_num)
x_cat2=np.array(x_cat)

x=np.concatenate([x_num2,x_cat2],axis=1)

x.shape
y.shape
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=42)
lr=LinearRegression()
lr.fit(x_train,y_train)
x_train.
print(lr.score(x_train,y_train))
print(lr.score(x_test,y_test))

from sklearn.linear_model import SGDRegressor
sc=SGDRegressor(max_iter=100,random_state=42)

sc.fit(x_train,y_train)
print(sc.score(x_train,y_train))
print(sc.score(x_test,y_test))

from sklearn.svm import SVR
svr_clf=SVR(kernel='poly',degree=3,coef0=1,C=5)
svr_clf.fit(x_train,y_train)
print(svr_clf.score(x_train,y_train))
print(svr_clf.score(x_test,y_test))

svr_reg=SVR(kernel='rbf',gamma=5,C=0.001)
svr_reg.fit(x_train,y_train)
print(svr_reg.score(x_train,y_train))
print(svr_reg.score(x_test,y_test))
pred=svr_reg.predict(x_test)

from sklearn.tree import DecisionTreeRegressor

dt=DecisionTreeRegressor(max_depth=10,random_state=42)
dt.fit(x_train,y_train)
print(dt.score(x_train,y_train))
print(dt.score(x_test,y_test))

from sklearn.model_selection import GridSearchCV
params={'min_impurity_decrease':[0.0001,0.0002,0.0003,0.0004,0.0005],
        'max_depth':range(5,50,1),}

gs=GridSearchCV(DecisionTreeRegressor(random_state=42),params,n_jobs=-1)
gs.fit(x_train,y_train)
dt=gs.best_estimator_
print(dt.score(x_train,y_train))
print(dt.score(x_test,y_test))
print(gs.cv_results_['mean_test_score'])

from sklearn.ensemble import GradientBoostingRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from sklearn.ensemble import RandomForestRegressor

rf_reg =RandomForestRegressor(random_state=0,n_estimators=1000)
gb_reg=GradientBoostingRegressor(random_state=0, n_estimators=1000)
xgb_reg=XGBRegressor(n_estimators=1000)
lgb_reg=LGBMRegressor(n_estimators=1000)

rf_reg.fit(x_train,y_train)
gb_reg.fit(x_train,y_train)
xgb_reg.fit(x_train,y_train)
lgb_reg.fit(x_train,y_train)

print(rf_reg.score(x_train,y_train))
print(rf_reg.score(x_test,y_test))
print(gb_reg.score(x_train,y_train))
print(gb_reg.score(x_test,y_test))
print(xgb_reg.score(x_train,y_train))
print(xgb_reg.score(x_test,y_test))
print(lgb_reg.score(x_train,y_train))
print(lgb_reg.score(x_test,y_test))

rf_params={'max_leaf_nodes':range(10,50,5),
           'n_estimators':range(500,1000,100)}
gb_params={}
xgb_params={'gamma':range(0,5,1),'max_depth':range(3,10,1)}
lgb_params={'num_iterations':range(100,500,100),
            'min_data_in_leaf':range(20,45,5),
            'max_depth':range(5,20,1)}

gs=GridSearchCV(RandomForestRegressor(random_state=42),rf_params,n_jobs=-1)
gs.fit(x_train,y_train)
dt=gs.best_estimator_
print(dt.score(x_train,y_train))
print(dt.score(x_test,y_test))

gs=GridSearchCV(LGBMRegressor(random_state=42),lgb_params,n_jobs=-1)
gs.fit(x_train,y_train)
dt=gs.best_estimator_
gs.best_params_
print(dt.score(x_train,y_train))
print(dt.score(x_test,y_test))

## 2021-07-01
## 483 줄부터 다시.
from sklearn.linear_model import LinearRegression
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures

data7['RDIST']=data7['KMDIST']/data7['SNDIST']
data7.info()
data7.describe()
data8=data7
data8.info()
data8=data8.drop(['UNC_YEAR','COUNTRYCD','KMDIST','RDIST','HSCD','KR_TRADE_HSCD_COUNTRYCD','NY_GDP_MKTP_CD_1Y','FTA'],axis=1)
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

x_num=play.loc[:,:'GDP_DIFF']
x_num.info()
x_cat=play.loc[:,'COUNTRYNM_Australia':]

pred=pred.reset_index().drop('index',axis=1)

x_num_p=pred.loc[:,:'GDP_DIFF']
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

from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import GridSearchCV
from scipy.stats import uniform, randint

#lgb_params={'num_iterations':range(100,500,100),
#            'min_data_in_leaf':range(20,45,5),
#            'max_depth':range(5,20,1)}

lgb_params={'min_data_in_leaf':randint(10,50),
            'max_depth':randint(5,50),
            'learning_rate':uniform(0.0001,0.1),
            'lambda_l2':uniform(0,5),
            'num_iterations':randint(100,500)}

gs=RandomizedSearchCV(LGBMRegressor(random_state=42),lgb_params,n_iter=100,n_jobs=-1,random_state=42)
gs.fit(x_train,y_train)
dt=gs.best_estimator_
gs.best_params_
print(dt.score(x_train,y_train))
print(dt.score(x_test,y_test))

lgb_params={'min_data_in_leaf':randint(15,25),
            'max_depth':randint(30,50),
            'learning_rate':uniform(0.01,0.1),
            'lambda_l2':uniform(2,5),
            'num_iterations':randint(400,800)}


gs=RandomizedSearchCV(LGBMRegressor(random_state=42),lgb_params,n_iter=100,n_jobs=-1,random_state=42)
gs.fit(x_train,y_train)
dt=gs.best_estimator_
gs.best_params_
print(dt.score(x_train,y_train))
print(dt.score(x_test,y_test))


lgb_params={'min_data_in_leaf':range(15,25,1),
            'max_depth':range(30,50,2),
            'learning_rate':range(0.01,0.1,0.01),
            'lambda_l2':range(2,5,0.1),
            'num_iterations':range(400,500,20)}


gs=GridSearchCV(LGBMRegressor(random_state=42),lgb_params,n_jobs=-1)
gs.fit(x_train,y_train)
dt=gs.best_estimator_
gs.best_params_
print(dt.score(x_train,y_train))
print(dt.score(x_test,y_test))

from lightgbm import LGBMRegressor
lgb_reg=LGBMRegressor(max_depth = 19, min_data_in_leaf = 20, num_iterations=400)
lgb_reg.fit(x_train,y_train)

print(lgb_reg.score(x_train,y_train))
print(lgb_reg.score(x_test,y_test))

from lightgbm import plot_importance
%matplotlib inline

lgb_reg=LGBMRegressor(max_depth = 13, min_data_in_leaf = 15, num_iterations=448,learning_rate=0.07993,lambda_l2=4.08)
lgb_reg.fit(x_train,y_train)
print(lgb_reg.score(x_train,y_train))
print(lgb_reg.score(x_test,y_test))


fig,ax=plt.subplots(figsize=(10,12))
plot_importance(lgb_reg,ax=ax)

g=sns.FacetGrid(data=data7)
g=sns.pairplot(data7)

data7['logkr']=np.log1p(data7['KR_TRADE_HSCD_COUNTRYCD'])
sns.swarmplot(x="group",y='logkr',data=data7)
#sns.swarmplot(x="RDIST",y='KR_TRADE_HSCD_COUNTRYCD',data=data7)
#sns.swarmplot(x="KDIST",y='KR_TRADE_HSCD_COUNTRYCD',data=data7)
sns.swarmplot(x="FTA",y='logkr',data=data7)
sns.swarmplot(x="IncomeGroup",y='logkr',data=data7)

sns.distplot(data7['logkr'])
data7.info()
data7.describe()
sns.distplot(data7['PA_NUS_FCRF'])

g=sns.pairplot(data7[['KR_TRADE_HSCD_COUNTRYCD','TRADE_HSCD_COUNTRYCD','FTA','group','IncomeGroup','RDIST','KMDIST']])

## 2021 0702;
data7.info()

regdata=data7[['TRADE_HSCD','TRADE_COUNTRYCD','NY_GDP_MKTP_CD','NY_GDP_MKTP_CD_1Y','TRADE_HSCD_COUNTRYCD','HS']]
regdata=regdata[regdata.TRADE_HSCD_COUNTRYCD.isnull()==False]
x=regdata[['TRADE_HSCD','TRADE_COUNTRYCD','NY_GDP_MKTP_CD','NY_GDP_MKTP_CD_1Y']]
y=regdata['TRADE_HSCD_COUNTRYCD']

x.HS.unique()

x=np.log1p(x)
y=np.log1p(y)

y=np.array(y)
x=np.array(x)

x=pd.DataFrame(x)
x=x.reset_index().drop('index',axis=1)
y=pd.DataFrame(y)
y=y.reset_index().drop('index',axis=1)
HS=regdata['HS']
HS=HS.reset_index().drop('index',axis=1)
visual=pd.concat([x,y],axis=1)
visual=pd.concat([visual,HS],axis=1)

visual.columns=['TC','GDP','GDP_1','TH','THC','HS']

from sklearn.linear_model import LinearRegression
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures

x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.3,random_state=42)
x_train.shape
x_test.shape
y_train
lr=LinearRegression()

poly=PolynomialFeatures(degree=3,include_bias=False)
poly.fit(x_train)
train_poly=poly.transform(x_train)
test_poly=poly.transform(x_test)

lr.fit(train_poly,y_train)
print(lr.score(train_poly,y_train))
print(lr.score(test_poly,y_test))

import matplotlib.pyplot as plt
import seaborn as sns
visual.HS.unique()

for i in visual.HS.unique():
    fig=plt.figure(figsize=(20,10))
    ax1=fig.add_subplot(2,2,1)
    ax2=fig.add_subplot(2,2,2)
    ax3=fig.add_subplot(2,2,3)
    ax4=fig.add_subplot(2,2,4)
    visual.loc[visual['HS']==i,:].plot(kind='scatter',x='TC',y='THC',ax=ax1)
    visual.loc[visual['HS']==i,:].plot(kind='scatter',x='GDP',y='THC',ax=ax2)
    visual.loc[visual['HS']==i,:].plot(kind='scatter',x='GDP_1',y='THC',ax=ax3)
    visual.loc[visual['HS']==i,:].plot(kind='scatter',x='TH',y='THC',ax=ax4)
    plt.title(i)
    name=i+'.jpg' # 새로 만든 사진 데이터의 파일명 생성
    path="./품목별변수/"
    plt.savefig(path+name)
    plt.show()
    plt.close()


visual=visual.drop('TC',axis=1)


for i in visual.HS.unique():
    regfor=visual.loc[visual['HS']==i,:]
    x=regfor[['GDP','GDP_1','TH']]
    y=regfor['THC']
    x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.3,random_state=42)
    lr=LinearRegression()
    lr.fit(x_train,y_train)
    print(i,"품목 > ","Train :",lr.score(x_train,y_train),"Test : ",lr.score(x_test,y_test))



from sklearn.experimental import enable_hist_gradient_boosting
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.model_selection import cross_validate

hgb=HistGradientBoostingRegressor(learning_rate=0.4,random_state=42)
scores=cross_validate(hgb,x_train,y_train,return_train_score=True)

print(np.mean(scores['train_score']),np.mean(scores['test_score']))



# 2021-07-05

from xgboost import XGBRegressor
xgb_params={'max_depth':range(3,7,1),'reg_alpha':range(0,2,1)
           ,'n_estimators':range(100,500,100)}
xgb=GridSearchCV(XGBRegressor(random_state=42),xgb_params,n_jobs=-1)
xgb.fit(x_train,y_train)
xgb.best_estimator_
xgb.score(x_train,y_train)
xgb.score(x_test,y_test)
y_pv=xgb.predict(x_pred)
y_pv=np.exp(y_pv-1)
y_pv.shape
pred['pred']=y_pv

pred.info()

data7.info()
data9=data7

pred_value=list(pred['pred'])

data9.loc[data9.TRADE_HSCD_COUNTRYCD.isnull()==True,'TRADE_HSCD_COUNTRYCD']=pred_value

data9.info()

data9['target']=data9.KR_TRADE_HSCD_COUNTRYCD/data9.TRADE_HSCD_COUNTRYCD

data9.info()

from sklearn.linear_model import LinearRegression
data10=data9
data10=data10.drop(['KR_TRADE_HSCD_COUNTRYCD','COUNTRYCD','NY_GDP_MKTP_CD_1Y','UNC_YEAR','HSCD'],axis=1)
data10['GDP_DIFF']=data10['GDP_DIFF'].astype("float")

data10=pd.get_dummies(data10,drop_first=True)

data10.info()


y_data=data10['target']
x_data=data10.drop('target',axis=1)

y_data=np.log1p(y_data)
x_data.info()
x_data.describe()


x_num=x_data.loc[:,:'RDIST']
x_num.info()
x_cat=x_data.loc[:,'COUNTRYNM_Australia':]
x_cat.info()

x_num=np.log1p(x_num)
x_num.describe()

ss=StandardScaler()
ss.fit(x_num)
x_num2=ss.transform(x_num)
x_cat2=np.array(x_cat)

x=np.concatenate([x_num2,x_cat2],axis=1)

x_train,x_test,y_train,y_test=train_test_split(x,y_data,test_size=0.3,random_state=42)

lr=LinearRegression()
lr.fit(x_train,y_train)
lr.score(x_train,y_train)

xgbr=XGBRegressor(random_state=42)
xgbr.fit(x_train,y_train)
xgbr.score(x_train,y_train)
xgbr.score(x_test,y_test)




### 그냥 예측 
data9=data7

pred_value=list(pred['pred'])

data9.loc[data9.TRADE_HSCD_COUNTRYCD.isnull()==True,'TRADE_HSCD_COUNTRYCD']=pred_value

data9.info()



from sklearn.linear_model import LinearRegression
data10=data9
data10=data10.drop(['COUNTRYCD','NY_GDP_MKTP_CD_1Y','UNC_YEAR','HSCD','target'],axis=1)
data10['GDP_DIFF']=data10['GDP_DIFF'].astype("float")

data10=pd.get_dummies(data10,drop_first=True)

data10.info()


y_data=data10['KR_TRADE_HSCD_COUNTRYCD']
x_data=data10.drop('KR_TRADE_HSCD_COUNTRYCD',axis=1)

y_data=np.log1p(y_data)
x_data.info()
x_data.describe()


x_num=x_data.loc[:,:'RDIST']
x_num.info()
x_cat=x_data.loc[:,'COUNTRYNM_Australia':]
x_cat.info()

x_num=np.log1p(x_num)
x_num.describe()

ss=StandardScaler()
ss.fit(x_num)
x_num2=ss.transform(x_num)
x_cat2=np.array(x_cat)

x=np.concatenate([x_num2,x_cat2],axis=1)

x_train,x_test,y_train,y_test=train_test_split(x,y_data,test_size=0.3,random_state=42)

lr=LinearRegression()
lr.fit(x_train,y_train)
lr.score(x_test,y_test)

xgbr=XGBRegressor(random_state=42)
xgbr.fit(x_train,y_train)
xgbr.score(x_train,y_train)
xgbr.score(x_test,y_test)

from lightgbm import LGBMRegressor
lgbr=LGBMRegressor(random_state=42)
lgbr.fit(x_train,y_train)
lgbr.score(x_train,y_train)
lgbr.score(x_test,y_test)


import seaborn as sns
data7.info()

num=data7.drop(['COUNTRYCD','COUNTRYNM','IncomeGroup','HS','group','FTA'],axis=1)
num.columns
target=num[['GDP_DIFF','target']]
num=num.drop(['GDP_DIFF','target'],axis=1)
num=np.log1p(num)
num2=pd.concat([num,target],axis=1)
columns=num2.columns

ss=StandardScaler()
ss.fit(num2)
num2=ss.transform(num2)
num2=pd.DataFrame(num2)
num2.columns=columns
num2
sns.distplot(num2['target'])

sns.heatmap(num2.corr(),annot=True)



## 그냥 
data11=data9.drop('target',axis=1)
data11.info()

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

x_train,x_test,y_train,y_test=train_test_split(x,y_data,test_size=0.3,random_state=42)

from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import GridSearchCV
from scipy.stats import uniform, randint

#lgb_params={'num_iterations':range(100,500,100),
#            'min_data_in_leaf':range(20,45,5),
#            'max_depth':range(5,20,1)}


lgb_params={'min_data_in_leaf':randint(10,50),
            'max_depth':randint(5,50),
            'learning_rate':uniform(0.0001,0.1),
            'lambda_l2':uniform(0,5),
            'num_iterations':randint(100,500)}



gs=RandomizedSearchCV(LGBMRegressor(random_state=42),lgb_params,n_jobs=-1,random_state=42)
gs.fit(x_train,y_train)
dt=gs.best_estimator_
gs.best_params_
print(dt.score(x_train,y_train))
print(dt.score(x_test,y_test))

import tensorflow as tf

from tensorflow import keras

x_train.shape

dense=keras.layers.Dense(10,activation='softmax',input_shape=(104,))
model=keras.Sequential(dense)

model.compile(loss='mean_square_error')
model.fit(x_train,y_train,epochs=100)
