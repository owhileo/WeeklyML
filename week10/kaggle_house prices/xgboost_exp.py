from matplotlib.pyplot import sca
import xgboost as xgb
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error,accuracy_score
import numpy as np
from sklearn.preprocessing import MinMaxScaler

from hyperopt import hp
from hyperopt_template import hyperopt_test


data=pd.read_csv('train.csv')
data_x=data.iloc[:,1:-1]
data_y=data.iloc[:,-1]

data_x=pd.get_dummies(data_x)

scaler=MinMaxScaler()

X_train, X_test, y_train, y_test = train_test_split(data_x, data_y, test_size=0.2, random_state=42)
X_train=scaler.fit_transform(X_train)
X_test=scaler.transform(X_test)
dtrain = xgb.DMatrix(X_train,label=y_train)
dtest = xgb.DMatrix(X_test)
dtest_ = xgb.DMatrix(X_test,y_test)

params_ = {
    'booster':'gbtree',#'dart'
    'eval_metric': 'rmse',    # 性能度量
    # 'eval_metric': 'merror',    # 性能度量
    # 'num_class': 3,             # 多分类类别

}

def train(params):
    model = xgb.train(params.update(params_), dtrain, params['num_rounds'])
    pred = model.predict(dtest)
    pred=np.round(pred)
    return -np.sqrt(mean_squared_error(pred, y_test))
    # return accuracy_score(pred, y_test)

space={
    'objective': ['reg:squarederror', 'reg:squaredlogerror','reg:pseudohubererror'],  # 损失函数
    # 'objective': ['multi:softmax'],  # 损失函数

    'eta': np.exp(np.arange(-3,5,.2)),             # shrinkage 学习率
    # 'eta': np.arange(.1,1,.05),             # shrinkage 学习率
    'gamma':  np.exp(np.arange(-3,5,.2)),           # 分裂的 最小损失减少，值越大，树越保守
    'max_depth':  range(5,30,2),         # 树最大深度
    'min_child_weight': range(1,20),  # 叶结点最小权重。当权重相同时，表示叶结点最少样本数
    'colsample_bytree': np.arange(.2,.9,.1),  # 构建树的时候随机挑选特征比例

    'subsample': np.arange(0,1.,.1),
    'lambda': np.exp(np.arange(-3,5,.2)),
    'grow_policy': ['depthwise', 'lossguide'],

    'num_rounds':  range(10,200,10),


    # 'rate_drop': np.arange(0,1.,.1),
    # 'one_drop': np.arange(0,1.,.1),
    # 'skip_drop': np.arange(0,1.,.1),
}

hyperopt_test(train,space,'xgboost_.png',max_evals=400)
