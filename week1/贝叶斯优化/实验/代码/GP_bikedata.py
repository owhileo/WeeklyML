import numpy as np
from data_process import *
from matplotlib import pyplot as plt
from BO import *
import sklearn
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C
from matplotlib import gridspec
import warnings
warnings.filterwarnings('ignore')

train_X,train_Y,test_X,test_Y = data_process()

###手动实现GP1
gp = GP()
gp.fit(train_X,train_Y)
pred_Y1, sigma1 = gp.predict(test_X)##此处输出sigma为协方差矩阵
sigma1 = sigma1.diagonal()##获取对角线元素

###sklearn GP2
kernel = C(1.0, (1e-3, 1e3)) * RBF(10, (1e-2, 1e2))
gp = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=9)
gp.fit(train_X, train_Y)
pred_Y2, sigma2 = gp.predict(test_X, return_std=True)

mse1 = sklearn.metrics.mean_squared_error(test_Y, pred_Y1,sample_weight=None, multioutput='uniform_average')/test_Y.shape[0]
mse2 = sklearn.metrics.mean_squared_error(test_Y, pred_Y2,sample_weight=None, multioutput='uniform_average')/test_Y.shape[0]
