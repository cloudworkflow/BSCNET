#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
import numpy as np  
import matplotlib.pyplot as plt 
import xgboost
from numpy import loadtxt
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn import preprocessing
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--dataset',default=1)
parser.add_argument('--task_type',default="M1.0")
opt = parser.parse_args()


# In[ ]:


if opt.dataset == 1:
    if opt.task_type == 'M0.3':
        y_path = 'y.csv'
        data_path = '/data1/M0.3.csv'
    elif opt.task_type == 'M1.0':
        y_path = 'y.csv'
        data_path = '/data1/M1.0.csv'
    else:
        y_path = 'y.csv'
        data_path = '/data1/E1.csv'
else:
    if opt.task_type == 'M0.3':
        y_path = 'y.csv'
        data_path = '/data2/M0.3.csv'
    elif opt.task_type == 'M1.0':
        y_path = 'y.csv'
        data_path = '/data2/M1.0.csv'
    else:
        y_path = 'y.csv'
        data_path = '/data2/E1.csv'


# In[4]:


def get_mape(y_true, y_pred):    
    return np.mean(np.abs((y_pred - y_true) / y_true))


# In[5]:


train_size = 6


# In[164]:


##1.0best
all_y = pd.read_csv(y_path,index_col=0)
df = pd.read_csv(data_path,index_col=0)
y = np.array(df["y"])
x = np.array(df.iloc[:,:-1])#18:-1
# 载入数据集
min_max_scaler=preprocessing.MinMaxScaler()
X_minMax=min_max_scaler.fit_transform(x.T)#x.T

# split data into X and y
X = X_minMax.T
y = y
seed = 100

#6,10,8
test_size = 0.3
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=seed)
#x_train, x_test = X[:train_size,],X[-(21-train_size):,]
#y_train,y_test = y[:train_size],y[-(21-train_size):]

#使用sklearn中提供的网格搜索进行测试--找出最好参数，并作为默认训练参数
import xgboost  as xgb
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import cross_validate
from sklearn.model_selection import KFold
from sklearn.model_selection import validation_curve
from sklearn.metrics import accuracy_score
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
from matplotlib import pyplot


model = xgb.XGBRegressor(max_depth=3, learning_rate=0.17, n_estimators=10, #6700
                        objective='reg:squarederror',min_child_weight=3,#6
                         booster='gbtree',subsample = 0.8,colsample_bylevel=0.1,
                         reg_alpha = 1, reg_lambda = 0.95, random_state=0)#0.15713346092237
#model = xgb.train(params,xgb.DMatrix(f_train, l_train))#0.18

print(np.shape(x_train))
model.fit(x_train, y_train)
preds =model.predict(x_test)

# MSE
mse_predict = mean_squared_error(y_test, preds)
print('MSE: ',mse_predict)

# MAE
mae_predict = mean_absolute_error(y_test, preds)
print('MAE: ',mae_predict)

# MAPE
mape_predict=get_mape(y_test, preds)
print('MAPE: ',mape_predict)
#MAPE:  0.2096967618888361
print()
#1.0
from xgboost import plot_importance #显示特征重要性
im=pd.DataFrame({'importance':model.feature_importances_,'var':df.columns[:-1]})
im=im.sort_values(by='importance',ascending=False)
print(im)
