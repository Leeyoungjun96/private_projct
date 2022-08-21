# -*- coding: utf-8 -*-
"""
Created on Mon Mar 22 15:29:45 2021

@author: COM
"""
#%%
# 전처리 파트
#%%
# 모듈 선언
import pandas as pd
import numpy as np
import missingno
import re
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
pd.options.display.max_columns= 50
pd.options.display.max_rows= 100


#%%
#2020년 데이터 시간 정리하기
def date_change(parme):
    TM = parme.split(' ')
    tw = TM[1]
    time = TM[2]
    if tw == '오전':
        h, m, s = time.split(':')        
        h_c = int(h)
        if h_c == 12:트
            h_c = h_c -12            
        result = str(h_c)+':'+ m
    else :
        h, m, s = time.split(':')
        h_c = int(h)
        if h_c < 12:
            h_c = h_c + 12
        result = str(h_c)+':'+ m
    return result

#%%
new_aid_2020.dropna(subset=['STATMNT_TM','GOUT_TM','ONSPOT_ARVL_TM','HOMING_TM'], inplace=True)
new_aid_2020['STATMNT_TM'] = new_aid_2020['STATMNT_TM'].apply(date_change)
new_aid_2020['GOUT_TM'] = new_aid_2020['GOUT_TM'].apply(date_change)
new_aid_2020['ONSPOT_ARVL_TM'] = new_aid_2020['ONSPOT_ARVL_TM'].apply(date_change)
new_aid_2020['HOMING_TM'] = new_aid_2020['HOMING_TM'].apply(date_change)



#%%
# 년도별 데이터 합치기 및 PK 부여
new_aid_all=pd.DataFrame(columns=['SUM_YY', 'SIGUN_NM','SIGUN_CD', 'GOUT_FIRESTTN_NM', 'GOUT_SAFE_CENTER_NM',
       'STATMNT_YMD', 'STATMNT_TM', 'RECEPT_COURS', 'JURISD_DIV_NM', 'GOUT_YMD', 'GOUT_TM', 
       'ONSPOT_ARVL_YMD', 'ONSPOT_ARVL_TM', 'ONSPOT_DSTN', 'ONSPOT_DSTN_AOT1', 'ONSPOT_DSTN_AOT2',
       'HOMING_YMD', 'HOMING_TM', 'PATNT_AGE', 'PATNT_SEX_DIV_NM',
       'EMRLF_SIDO_NM', 'EMRLF_SIGNGU_NM', 'EMRLF_EMD_NM', 'EMRLF_LI_NM',
       'FRGNR_YN', 'NATNLTY_NM', 'PATNT_JOB', 'RELIF_OCCURPLC_TYPE', 'PATNT_SYMPTMS_TYPE','PATNT_SYMPTMS_TYPE2', 
       'ACC_CAT1', 'ACC_CAT2', 'ACC_CAT3', 'RELIF_ACDNT_ASORTMT_UP_NM', 'RELIF_ACDNT_ASORTMT_LOW_NM','PATNT_OCCUR_TYPE',
       'CONSCS_STATE_TYPE','AMBLNCWKR_QUALFCTN_RATG', 'AMBLNCWKR_RATG','RELIF_SECTCHEF_QUALFCTN_RATG','TROUBLE'])
print(new_aid_all)
for i in range(2010,2021):
    title = 'aid_utf8/2.eng/aid'+str(i)+'(u8s,eng).csv'
    data = pd.read_csv(title,encoding='utf-8-sig')
    new_aid_all = pd.concat([new_aid_all,data],axis=0)
new_aid_all.drop(['SIGUN_CD'],axis=1,inplace=True)
new_aid_all.dropna(subset=['STATMNT_TM','ONSPOT_ARVL_TM','HOMING_TM'], inplace=True)
new_aid_all.reset_index(drop=True,inplace=True)
new_aid_all.reset_index(drop=False,inplace=True)
new_aid_all.rename(columns ={'index':'PK'},inplace=True)
print(new_aid_all.info())
print(new_aid_all.columns)



#%%
# 소요시간 계산 함수
def timesplit2(list):
    STATMENT_TM = list[0].split(':')
    ONSPOT_TM = list[6].split(':')
    HOMING_TM = list[11].split(':')
    ST_H = int(STATMENT_TM[0])
    ST_M = int(STATMENT_TM[1])
    OT_H = int(ONSPOT_TM[0])
    OT_M = int(ONSPOT_TM[1])
    HT_H = int(HOMING_TM[0])
    HT_M = int(HOMING_TM[1])
    if(ST_H==OT_H):
    	st = OT_M - ST_M
    
    elif(ST_H!=OT_H): 
    	if(ST_H<OT_H): #도착시간 날짜가 안넘어감
    		if(OT_H==HT_H): #귀소시간 날짜가 안넘어감
    			if(OT_M<HT_M):
    				st = (OT_M-ST_M) + (OT_H-ST_H)*60
    			else:
    				st= -1
    			
    		if(OT_H<HT_H): #귀소시간 날짜가 안넘어감
    			st = (OT_M-ST_M) + (OT_H-ST_H)*60
    			
    		elif(OT_H>HT_H): #귀소시간 날짜가 넘어감
    			if(OT_H-HT_H > 20):
    				st = (OT_M-ST_M) + (OT_H-ST_H)*60
    			else:
    				st=-1
    		
    	elif(ST_H>OT_H): #도착시간 날짜가 넘어감
    		if (OT_H==HT_H): #도착날짜 = 귀소날짜
    			if (OT_M<HT_M):
    				st = (OT_M-ST_M) + (OT_H+24-ST_H)*60
    			else:
    				st = -1
    		
    		elif (OT_H<HT_H): #도착날짜 = 귀소날짜
    			st = (OT_M-ST_M) + (OT_H+24-ST_H)*60
    		
    		elif (OT_H>HT_H): #도착날짜 != 귀소날짜
    			st = -1    
    if (st<0):
    	result = None
    else :
    	result = st
        
    return result

#%%
# 소요시간 계산하여 원래 데이터 프레임에 추가.
siri = new_aid_all.loc[:,'STATMNT_TM':'HOMING_TM'].apply(timesplit2,axis=1, result_type='reduce').to_frame()
siri.columns=['MOVE_TM']
new_aid_all = pd.concat([new_aid_all,siri],axis=1)


#%%
# 파일 합본만들고 pk 붙이기.
def setaidall():
    aid_all_total = pd.DataFrame(columns=['SUM_YY', 'SIGUN_NM', 'GOUT_FIRESTTN_NM', 'GOUT_SAFE_CENTER_NM',
       'STATMNT_YMD', 'STATMNT_TM', 'RECEPT_COURS', 'JURISD_DIV_NM',
       'GOUT_YMD', 'GOUT_TM', 'ONSPOT_ARVL_YMD', 'ONSPOT_ARVL_TM',
       'ONSPOT_DSTN', 'ONSPOT_DSTN_AOT1', 'ONSPOT_DSTN_AOT2', 'HOMING_YMD',
       'HOMING_TM', 'PATNT_AGE', 'PATNT_SEX_DIV_NM', 'EMRLF_SIDO_NM',
       'EMRLF_SIGNGU_NM', 'EMRLF_EMD_NM', 'EMRLF_LI_NM', 'FRGNR_YN',
       'NATNLTY_NM', 'PATNT_JOB', 'RELIF_OCCURPLC_TYPE', 'PATNT_SYMPTMS_TYPE',
       'PATNT_SYMPTMS_TYPE2', 'ACC_CAT1', 'ACC_CAT2', 'ACC_CAT3',
       'RELIF_ACDNT_ASORTMT_UP_NM', 'RELIF_ACDNT_ASORTMT_LOW_NM',
       'PATNT_OCCUR_TYPE', 'CONSCS_STATE_TYPE', 'AMBLNCWKR_QUALFCTN_RATG',
       'AMBLNCWKR_RATG', 'RELIF_SECTCHEF_QUALFCTN_RATG', 'TROUBLE', 'MOVE_TM'])
    for i in range(1,6):
        title = "aid_utf8/4.mt/aidall"+str(i)+"(u8s,eng,all,mt).csv"
        data = pd.read_csv(title,encoding='utf-8-sig')
        aid_all_total = pd.concat([aid_all_total,data],axis=0)
        
    aid_all_total.reset_index(drop=True,inplace=True)
    aid_all_total.reset_index(drop=False,inplace=True)
    aid_all_total.rename(columns ={'index':'PK'},inplace=True)
    return aid_all_total
#%%
aid_all_total = setaidall()


#%%
# PK와 컬럼 데이터프레임 합치고 저장하는 함수
def savesetdf(df1,df2,csvName='temp.csv'):
    result_df = pd.concat([df1,df2], axis=1)
    result_df.to_csv(csvName,encoding='utf-8-sig',index=False)
    return result_df
    
    
#%%
pk_df = aid_all_total['PK'].to_frame()
print(type(pk_df))

#%%
# 기존 MOVE_TM 오류 발견후 제거 함수
def notequymd(list):
    STATMNT_YMD = list[0].split('-')
    HOMING_YMD = list[1].split('-')
    STATMNT_TM = list[2].split(':')
    ONSPOT_ARVL_TM = list[3].split(':')
    HOMING_TM = list[4].split(':')
    MOVE_TM = list[5]
    ST_H = int(STATMNT_TM[0])
    ST_M = int(STATMNT_TM[1])
    OT_H = int(ONSPOT_ARVL_TM[0])
    OT_M = int(ONSPOT_ARVL_TM[1])
    HT_H = int(HOMING_TM[0])
    HT_M = int(HOMING_TM[1])
    st_y = int(STATMNT_YMD[0])
    st_m = int(STATMNT_YMD[1])
    st_d = int(STATMNT_YMD[2])
    ho_y = int(HOMING_YMD[0])
    ho_m = int(HOMING_YMD[1])
    ho_d = int(HOMING_YMD[2])   
    
    if(st_y == ho_y):
        if(st_m == ho_m):
            if(st_d == ho_d):
                if(ST_H>OT_H):
                    st = -1
                elif(OT_H>HT_H):
                    st = -1
                else:
                    st = MOVE_TM
            elif(st_d + 1 < ho_d):
                st = -1
            else:
                st = MOVE_TM
        else :
            st = MOVE_TM
    else :
        st = MOVE_TM
        
    if (st<0):
    	result = None
    else :
    	result = st
        
    return result
#%%
siri = aid_all_total.loc[:,'STATMNT_YMD':'MOVE_TM'].apply(notequymd,axis=1, result_type='reduce').to_frame()
siri.columns=['MOVE_TM']
aid_pk_mt = pd.concat([pk_df,siri], axis=1)
print(aid_pk_mt)

#%%
# 레이블 인코딩
#%%
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
le.fit(aid_pk_jur.JURISD_DIV_NM)
#%%
print(le.classes_)
JURISD_DIV_NM = le.transform(aid_pk_jur.JURISD_DIV_NM)
#%%
# 원 핫 인코딩
temp = pd.get_dummies(aid_pk_jur.JURISD_DIV_NM)
temp.columns=['JURISD_DIV_NM_CNT','JURISD_DIV_NM_OUT']
temp = pd.concat([aid_pk_jur['PK'].to_frame(),temp],axis=1)

#%%
# 이상치 제거
moveTm = pd.read_csv('aid_utf8/6.set/aid(pk,mt)03291921.csv',encoding='utf-8-sig')
#%%
moveTm.dropna(axis=0,inplace=True)
print(moveTm.isnull().sum())
#%%
print(moveTm['PK'])
print(moveTm['MOVE_TM'])
#%%
print(moveTm.head(50))
#%%
plt.figure(figsize=(12,8))
sns.boxplot(data = moveTm['MOVE_TM'], color='red')
plt.show()
#%%
def get_outlier(df=None, column=None, weight=1.5):
    q_25 = np.percentile(df[column].values, 25)
    q_75 = np.percentile(df[column].values, 75)
    
    IQR = q_75 - q_25
    IQR_weight = IQR*weight
    
    lowest = q_25 - IQR_weight
    highest = q_75 + IQR_weight
    
    outlier_idx = df[column][(df[column] < lowest) | (df[column] > highest)].index
    return outlier_idx
#%%
outlier_idx = get_outlier(df=moveTm, column='MOVE_TM', weight=1.5)
moveTm.drop(outlier_idx, axis=0, inplace=True)

#%%
print(moveTm.info())
#%%
print(moveTm[moveTm['MOVE_TM']>14])

#%%
moveTm.to_csv('aid(pk,mt)04012219.csv',encoding='utf-8-sig', index=False)
#%%
# 이상치 제거
od_df = pd.read_csv('aid_utf8/6.set/aid(pk,od)03291533.csv',encoding='utf-8-sig')
#%%
od_df.dropna(axis=0,inplace=True)
print(od_df.isnull().sum())
#%%
print(od_df['PK'])
print(od_df['ONSPOT_DSTN'])
#%%
print(od_df.head(50))
#%%
plt.figure(figsize=(12,8))
sns.boxplot(data = od_df['ONSPOT_DSTN'], color='red')
plt.show()
#%%
def get_outlier(df=None, column=None, weight=1.5):
    q_25 = np.percentile(df[column].values, 25)
    q_75 = np.percentile(df[column].values, 75)
    
    IQR = q_75 - q_25
    IQR_weight = IQR*weight
    
    lowest = q_25 - IQR_weight
    highest = q_75 + IQR_weight
    
    outlier_idx = df[column][(df[column] < lowest) | (df[column] > highest)].index
    return outlier_idx
#%%
outlier_idx = get_outlier(df=od_df, column='ONSPOT_DSTN', weight=1.5)
od_df.drop(outlier_idx, axis=0, inplace=True)

#%%
print(od_df.info())
#%%
print(od_df[od_df['ONSPOT_DSTN']>10])

#%%
od_df.to_csv('aid(pk,od)04021140.csv',encoding='utf-8-sig', index=False)
#%%
