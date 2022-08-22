# -*- coding: utf-8 -*-
"""
Created on Fri Mar 26 10:49:46 2021

@author: COM
"""

# 모델 머신러닝 파트
#%%
#모듈 선언
import pandas as pd
import numpy as np
import missingno
import re
from tqdm import tqdm
from scipy.stats import kurtosis, skew
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_val_score
import math
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import r2_score
from scipy import stats
from sklearn.decomposition import PCA
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.svm import SVR

from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import VotingRegressor, VotingClassifier






pd.options.display.max_columns= 50
pd.options.display.max_rows= 1000


#%% 필요 데이터 가져와서 프레임 만들기
def setaiddata(list_csvNames):
    dirtstr = 'aid_utf8/6.set/'
    csvNames = list_csvNames # 확장자 제외하고 쓸것
    pktitle = dirtstr+'aid(pk)03291407.csv' # pk파일명 넣을것
    aid_data = pd.read_csv(pktitle,encoding='utf-8-sig')    
    for csvName in csvNames:
        title = dirtstr+ csvName +".csv"
        print(title)
        data = pd.read_csv(title,encoding='utf-8-sig')
        aid_data = pd.merge(aid_data, data, on='PK')
    print(aid_data.info())
    return aid_data

#%%
# 피어슨 상관계수
from scipy.stats import pearsonr
print(pearsonr(aid_set_data.STATMNT_TM, aid_set_data.EMRLF_EMD_NM))

#%%
# 스피어만 상관계수 
from scipy.stats.stats import spearmanr
print(spearmanr(aid_set_data.EMRLF_EMD_NM, aid_set_data.ONSPOT_DSTN))




#%%
# 선형회귀 모델 평가 함수
# 사용시 주의사항. df(데이터프레임)의 종속변수는 항상 맨 마지막 열에 위치 할 것.

#%%
# 선형회귀 모델 평가 함수
# 사용시 주의사항. df(데이터프레임)의 종속변수는 항상 맨 마지막 열에 위치 할 것.
def modelscore(model,df):
    # df의 독립변수와 종속변수 분리    
    X = df.iloc[:,:-1]
    print('사용한 독립변수들 :\n', X.columns)
    y = df.iloc[:,-1]
    print('종속변수 : ',y.name)
        
    # X,y의 학습데이터, 테스트데이터 분리
    print('데이터 분리중...')
    train_X, test_X, train_y, test_y = train_test_split(X,y,test_size=0.3)


    # 모델학습
    print('모델 학습중...')
    model.fit(train_X,train_y)
        
    #예측하기
    print('모델 예측중...')
    pred_train_y = model.predict(train_X)
    pred_test_y = model.predict(test_X)
    
    # 훈련데이터 셋을 이용한 평가
    print('---훈련데이터 셋을 이용한 평가---')
    print('결정계수 r2 : ',r2_score(train_y, pred_train_y))
    print('RMSE : ', math.sqrt(mean_squared_error(train_y, pred_train_y)))
    print('MAE : ', mean_absolute_error(train_y, pred_train_y))
    Rsquared = 1 - (1-model.score(train_X,train_y))*(len(train_y)-1)/(len(train_y)-train_X.shape[1]-1)
    print('수정된 결정계수(R-squared)를 위한 summary :\n',Rsquared) 
    
    # 테스트데이터 셋을 이용한 평가
    print('---테스트데이터 셋을 이용한 평가---')
    print('종속변수 예측평균 : ',pred_test_y.mean())
    print('종속변수 실제평균 : ',test_y.mean())
    print('결정계수 r2 : ',r2_score(test_y,pred_test_y))
    print('RMSE : ', math.sqrt(mean_squared_error(test_y, pred_test_y)))
    print('MAE : ',mean_absolute_error(test_y, pred_test_y))
    Rsquared = 1 - (1-model.score(test_X,test_y))*(len(test_y)-1)/(len(test_y)-test_X.shape[1]-1)
    print('수정된 결정계수(R-squared)를 위한 summary :\n',Rsquared)
    
    # 잔차의 정규성 확인
    print('띄워진 그래프는 잔차의 정규성을 확이하기 위한 그래프임.')
    fig= plt.figure()
    plt.scatter(test_y,pred_test_y)
    plt.xlabel("Target y")
    plt.xlabel("Predicted y")
    plt.title("Prediction vs. Actual")
    plt.show()
    
    return model

#%%
# 선형회귀 모델 평가 함수 (주성분 포함)
# 사용시 주의사항. df(데이터프레임)의 종속변수는 항상 맨 마지막 열에 위치 할 것.
# 'com'매게변수에는 주성분 수 입력.
def modelscore2(model,df, com=5):
    # df의 독립변수와 종속변수 분리    
    X = df.iloc[:,:-1]
    print('사용한 독립변수들 :\n', X.columns)
    y = df.iloc[:,-1]
    print('종속변수 : ',y.name)
    
    # X,y의 학습데이터, 테스트데이터 분리
    print('데이터 분리중...')
    train_X, test_X, train_y, test_y = train_test_split(X,y,test_size=0.3)
    
    # 주성분 분석
    print('주성분 분석중...')
    pca = PCA(n_components=com)
    train_X_pca = pca.fit_transform(train_X)
    test_X_pca = pca.fit_transform(test_X)    
    
    # 모델학습
    print('모델 학습중...')
    model.fit(train_X_pca,train_y)
    
    #예측하기
    print('모델 예측중...')
    pred_train_y = model.predict(train_X_pca)
    pred_test_y = model.predict(test_X_pca)
    
    # 훈련데이터 셋을 이용한 평가
    print('---훈련데이터 셋을 이용한 평가---')
    print('결정계수 r2 : ',r2_score(train_y, pred_train_y))
    print('RMSE : ', math.sqrt(mean_squared_error(train_y, pred_train_y)))
    print('MAE : ', mean_absolute_error(train_y, pred_train_y))
    Rsquared = 1 - (1-model.score(train_X_pca,train_y))*(len(train_y)-1)/(len(train_y)-train_X_pca.shape[1]-1)
    print('수정된 결정계수(R-squared)를 위한 summary :\n',Rsquared) 
    
    # 테스트데이터 셋을 이용한 평가
    print('---테스트데이터 셋을 이용한 평가---')
    print('종속변수 예측평균 : ',pred_test_y.mean())
    print('종속변수 실제평균 : ',test_y.mean())
    print('결정계수 r2 : ',r2_score(test_y,pred_test_y))
    print('RMSE : ', math.sqrt(mean_squared_error(test_y, pred_test_y)))
    print('MAE : ',mean_absolute_error(test_y, pred_test_y))
    Rsquared = 1 - (1-model.score(test_X_pca,test_y))*(len(test_y)-1)/(len(test_y)-test_X_pca.shape[1]-1)
    print('수정된 결정계수(R-squared)를 위한 summary :\n',Rsquared)
    
    # 잔차의 정규성 확인
    print('띄워진 그래프는 잔차의 정규성을 확이하기 위한 그래프임.')
    fig= plt.figure()
    plt.scatter(test_y,pred_test_y)
    plt.xlabel("Target y")
    plt.xlabel("Predicted y")
    plt.title("Prediction vs. Actual")
    plt.show()
    
    return model, pca.components_

#%%
# 선형회귀 모델 평가 함수 (특이값분해)
# 사용시 주의사항. df(데이터프레임)의 종속변수는 항상 맨 마지막 열에 위치 할 것.
from sklearn.utils.extmath import randomized_svd
def modelscore3(model,df,com=6,column_list=['A','B','C','D','E']):
    # df의 독립변수와 종속변수 분리    
    X = df.iloc[:,:-1]
    print('사용한 독립변수들 :\n', X.columns)
    y = df.iloc[:,-1]
    print('종속변수 : ',y.name)
    
    #특이값 분해
    print('특이값 분해중...')
    U, s, VT = randomized_svd(X.to_numpy(), n_components=com, n_iter='auto', random_state=None)    
    S = np.zeros((com,com))        
    for i in range(com):
        S[i][i] = s[i]    
    X_svd = U @ S @ VT[:,0:com]
    X_svd_df = pd.DataFrame(X_svd, columns=column_list)
    
    
    # X,y의 학습데이터, 테스트데이터 분리
    print('데이터 분리중...')
    train_X, test_X, train_y, test_y = train_test_split(X_svd_df,y,test_size=0.3)
    
    # 모델학습
    print('모델 학습중...')
    model.fit(train_X,train_y)
        
    #예측하기
    print('모델 예측중...')
    pred_train_y = model.predict(train_X)
    pred_test_y = model.predict(test_X)
    
    # 훈련데이터 셋을 이용한 평가
    print('---훈련데이터 셋을 이용한 평가---')
    print('결정계수 r2 : ',r2_score(train_y, pred_train_y))
    print('RMSE : ', math.sqrt(mean_squared_error(train_y, pred_train_y)))
    print('MAE : ', mean_absolute_error(train_y, pred_train_y))
    Rsquared = 1 - (1-model.score(train_X,train_y))*(len(train_y)-1)/(len(train_y)-train_X.shape[1]-1)
    print('수정된 결정계수(R-squared)를 위한 summary :\n',Rsquared) 
    
    # 테스트데이터 셋을 이용한 평가
    print('---테스트데이터 셋을 이용한 평가---')
    print('종속변수 예측평균 : ',pred_test_y.mean())
    print('종속변수 실제평균 : ',test_y.mean())
    print('결정계수 r2 : ',r2_score(test_y,pred_test_y))
    print('RMSE : ', math.sqrt(mean_squared_error(test_y, pred_test_y)))
    print('MAE : ',mean_absolute_error(test_y, pred_test_y))
    Rsquared = 1 - (1-model.score(test_X,test_y))*(len(test_y)-1)/(len(test_y)-test_X.shape[1]-1)
    print('수정된 결정계수(R-squared)를 위한 summary :\n',Rsquared)
    
    # 잔차의 정규성 확인
    print('띄워진 그래프는 잔차의 정규성을 확이하기 위한 그래프임.')
    fig= plt.figure()
    plt.scatter(test_y,pred_test_y)
    plt.xlabel("Target y")
    plt.xlabel("Predicted y")
    plt.title("Prediction vs. Actual")
    plt.show()
    
    return model

#%%
def useGridSearchCV(model, df, N_OPTIONS_list, EAT_OPTIONS_list):
    # df의 독립변수와 종속변수 분리    
    X = df.iloc[:,:-1]
    print('사용한 독립변수들 :\n', X.columns)
    y = df.iloc[:,-1]
    print('종속변수 : ',y.name)
         
    pipe = Pipeline([
        # the reduce_dim stage is populated by the param_grid
        ('reduce_dim', None),
        ('Regression',model)
        ])
    N_FEATURES_OPTIONS = N_OPTIONS_list
    EAT_OPTIONS = EAT_OPTIONS_list
    param_grid = [
        {
            'reduce_dim':[PCA(iterated_power=7)],
            'reduce_dim__n_components': N_FEATURES_OPTIONS,
            'Regression__EAT': EAT_OPTIONS
        },
        {
            'reduce_dim':[SelectKBest(chi2)],
            'reduce_dim__k': N_FEATURES_OPTIONS,
            'Regression__EAT': EAT_OPTIONS
         },
        ]
    reducer_labels = ['PCA', 'KBest(chi2)']
    grid = GridSearchCV(pipe, cv=5, n_jobs=-1, param_grid=param_grid)
    print('GridSearchCV 학습중...')
    grid.fit(X,y)
    print('최고 점수를 낸 파라미터의 모형 : ', grid.best_estimator_)
    print('최적 파라미터 정보 : ', grid.best_params_)
    print('최적화 된 모형의 score(r2)값 : ', grid.score(X,y))
    mean_scores = np.array(grid.cv_results_['mean_test_score'])
    mean_scores = mean_scores.reshape(len(EAT_OPTIONS), -1, len(N_FEATURES_OPTIONS))
    mean_scores = mean_scores.max(axis=0)
    bar_offsets = (np.arange(len(N_FEATURES_OPTIONS))*
                  (len(reducer_labels)+1)+.5)
    
    print('그래프 그리는 중...')
    plt.figure()
    COLORS = 'bgrcmyk'
    for i, (label, reducer_scores) in enumerate(zip(reducer_labels, mean_scores)):
        plt.bar(bar_offsets + i, reducer_scores, label=label, color=COLORS[i])
    
    plt.title("Comparing feature reduction techniques")
    plt.xlabel('Reduced number of features')
    plt.xticks(bar_offsets + len(reducer_labels) / 2, N_FEATURES_OPTIONS)
    plt.ylabel('Aid Regression R2')
    plt.ylim((0,1))
    plt.legend(loc='upper left')
    plt.show()
    print('최고 점수를 낸 파라미터의 모형 : ', grid.best_estimator_)
    print('최적 파라미터 정보 : ', grid.best_params_)
    print('최적화 된 모형의 score(r2)값 : ', grid.score(X,y))
    
    return grid



#%%
# 1번모델을 위한 데이터셋 불러오기
aid_set_data = pd.read_csv('aid_utf8/7.com/soyoung_0407.csv',encoding='utf-8-sig')

#%%
# 분석을 위해 pk값 제거
aid_set_data.drop("PK",axis=1,inplace=True)
print(aid_set_data.head())

#%%
# 2번모델을 위한 데이터셋 만들기 (1단계 종속변수 뽑기)
aid_onspot = aid_set_data["ONSPOT_DSTN"].to_frame()
print(aid_onspot)
#%%
# 2번모델을 위한 데이터셋 만들기 (2단계 불필요 변수 제거)
aid_set_data_onspot = aid_set_data.drop(["ONSPOT_DSTN","EMRLF_EMD_NM", "RELIF_OCCURPLC_TYPE"],axis=1)
print(aid_set_data_onspot)

#%%
# 2번모델을 위한 데이터셋 만들기 (3단계 독립변수와 종속변수 더한 프레임 만들기)
aid_set_data_onspot_set = pd.concat([aid_set_data_onspot,aid_onspot],axis=1)

#%%
from xgboost import XGBRegressor

#%%
# 1번모델 최적 파라미터 탐색
useGridSearchCV(XGBRegressor(),aid_set_data,[4,5,6,7],[0])

#%%
# 2번모델 최적 파라미터 탐색
useGridSearchCVdata = useGridSearchCV(XGBRegressor(),aid_set_data_onspot_set,[4,5,6,7],[0])


#%%
# XGBRegressor 모델 (최종모델생성코드)
from xgboost import XGBRegressor
model_aid_xgb2 = XGBRegressor(EAT=0, base_score=0.5, booster='gbtree',
                              colsample_bylevel=1, colsample_bynode=1,
                              colsample_bytree=1, gamma=0, gpu_id=-1,
                              importance_type='gain',
                              interaction_constraints='',
                              learning_rate=0.300000012, max_delta_step=0,
                              max_depth=6, min_child_weight=1, missing=None,
                              monotone_constraints='()', n_estimators=100,
                              n_jobs=8, num_parallel_tree=1, random_state=0,
                              reg_alpha=0, reg_lambda=1, scale_pos_weight=1,
                              subsample=1, tree_method='exact',
                              validate_parameters=1, verbosity=None)
#%%
#  탐색파라미터를 적용한 1번모델 생성
aid_set_data_endfit2 = modelscore(model_aid_xgb2, aid_set_data)

#%%
#  탐색파라미터를 적용한 2번모델 생성
aid_set_data_onspot_set_endfit2 = modelscore2(model_aid_xgb2, aid_set_data_onspot_set, 5)


#%%
import pickle
#  모델 피클링 추출
with open("aid_predict_time.pkl", "wb") as f:
    pickle.dump(aid_set_data_endfit2, f)
#%%
with open("aid_predict_dstn.pkl", "wb") as f:
    pickle.dump(aid_set_data_onspot_set_endfit2, f)





#%%
# 미사용 모델


#%%
# SVM - SVR 모델
from sklearn.svm import SVR
modelscore(SVR(),aid_set_data)

#%%
# LinearRegression 모델
model_aid = LinearRegression()
modelscore2(model_aid,aid_set_data)

#%%
# RandomForestRegressor 모델
from sklearn.ensemble import RandomForestRegressor
aid_RFR = RandomForestRegressor(n_estimators=50,
                    max_depth=None, max_features=4,
                    oob_score=False, random_state=1)
aid_RFR_endfit = modelscore(aid_RFR,aid_set_data)

#%%
# Adaboost regressor 모델
from sklearn.ensemble import AdaBoostRegressor
aid_ABR = AdaBoostRegressor()
aid_ABR_endfit = modelscore(aid_ABR,aid_set_data)

#%%
#lightgbm 모델
import lightgbm as lgb
ds_train = lgb.Dataset(train_X, label=train_y) #데이터셋생성
params = {}
params['learning_rate'] = 0.003
params['boosting_type'] = 'gbdt'
params['objective'] = 'regression'
params['metric'] = 'mse'
params['sub_feature'] = 0.5
params['num_leaves'] = 144
params['max_depth'] = 30
clf = lgb.train(params, ds_train, 1000)
y_pred=clf.predict(test_X)
print('MAE : ',mean_absolute_error(test_y, y_pred))
#%%
#VotingRegressor 확인
reg_dt = DecisionTreeRegressor()
reg_lr = LinearRegression(normalize=True)
reg_svm = SVR()
estimators = [('dt', reg_dt), ('lr', reg_lr), ('svm', reg_svm)]
#예측모형
reg_movetm = VotingRegressor(estimators)
reg_movetm.fit(train_X, train_y)
reg_pred = reg_movetm.predict(test_X)
# 예측모형 성능
print('MAE : ',mean_absolute_error(test_y, reg_pred))
#%%
aid_set_mt = aid_set_data['MOVE_TM']
print(aid_set_mt)

#%%
print(aid_set_mt[aid_set_mt > 1000])