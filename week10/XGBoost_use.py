from sklearn.datasets import load_iris, load_diabetes
import xgboost as xgb
from xgboost import plot_importance
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, mean_squared_error

# 数据集准备
iris = load_iris()
X = iris.data
y = iris.target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)
dtrain = xgb.DMatrix(X_train, y_train)
dtest = xgb.DMatrix(X_test)  # , feature_names=['12','1d2','1f2','1g2']

# 参数设置[学习任务，学习器]
params = {
    'booster': 'gbtree',        # 基学习器
    'objective': 'multi:softmax',  # 损失函数
    'eval_metric': 'merror',    # 性能度量
    'num_class': 3,             # 多分类类别

    'eta': 0.1,             # shrinkage 学习率
    'gamma': 0.1,           # 分裂的 最小损失减少，值越大，树越保守
    'max_depth': 6,         # 树最大深度
    'min_child_weight': 3,  # 叶结点最小权重。当权重相同时，表示叶结点最少样本数
    'lambda': 2,            # 正则化系数
    'colsample_bytree': 0.7,  # 构建树的时候随机挑选特征比例
}
num_rounds = 20  # boosting 迭代次数

# 训练
model = xgb.train(params, dtrain, num_rounds)
# model.dump_model('1.txt')

# 对测试集进行预测
pred = model.predict(dtest)
print(pred, y_test)

# 计算准确率
print(accuracy_score(pred, y_test))

# ===================== 特征重要性 =====================
score = model.get_score(importance_type='weight')  # 分裂次数
print(score)
score = model.get_score(importance_type='gain')  # 增益
print(score)
score = model.get_score(importance_type='cover')  # 覆盖样本数
print(score)

# plot_importance(model)
# plt.show()

# ===================== 自定义loss =====================
# import numpy as np
# from typing import Tuple
#
#
# def gradient(predt: np.ndarray, dtrain: xgb.DMatrix) -> np.ndarray:
#     """Compute the gradient squared log error."""
#     y = dtrain.get_label()
#     return (np.log1p(predt) - np.log1p(y)) / (predt + 1)
#
#
# def hessian(predt: np.ndarray, dtrain: xgb.DMatrix) -> np.ndarray:
#     """Compute the hessian for squared log error."""
#     y = dtrain.get_label()
#     return ((-np.log1p(predt) + np.log1p(y) + 1) /
#             np.power(predt + 1, 2))
#
#
# def squared_log(predt: np.ndarray, dtrain: xgb.DMatrix) -> Tuple[np.ndarray, np.ndarray]:
#     predt[predt < -1] = -1 + 1e-6
#     grad = gradient(predt, dtrain)
#     hess = hessian(predt, dtrain)
#     return grad, hess
#
#
