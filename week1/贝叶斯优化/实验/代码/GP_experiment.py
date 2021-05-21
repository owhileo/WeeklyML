import numpy as np
from matplotlib import pyplot as plt
from BO import *
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C
from matplotlib import gridspec
import warnings
warnings.filterwarnings('ignore')


np.random.seed(1)
def f(x):
    """The function to predict."""
    return x * np.sin(x)

train_X = np.atleast_2d(np.linspace(0, 10, 100)).T ###设定训练集大小
train_Y = f(train_X).ravel()
plt_X = np.atleast_2d(np.linspace(0, 10, 1000)).T

###手动实现GP1
gp = GP()
gp.fit(train_X,train_Y)
plt_Y, sigma1 = gp.predict(plt_X)##此处输出sigma为协方差矩阵
sigma1 = sigma1.diagonal()##获取对角线元素

###sklearn GP2
kernel = C(1.0, (1e-3, 1e3)) * RBF(10, (1e-2, 1e2))
gp = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=9)
gp.fit(train_X, train_Y)
y_pred, sigma2 = gp.predict(plt_X, return_std=True)

####绘图
fig = plt.figure(figsize=(16, 10))

gs=gridspec.GridSpec(2,2)#分为2行2列 
GP1 = plt.subplot(gs[:,0])
GP2 = plt.subplot(gs[:,1])

GP1.plot(plt_X, f(plt_X), 'r:', label=r'$f(x) = x\,\sin(x)$')
GP1.plot(train_X, train_Y, 'r.', markersize=10, label='Observations')
GP1.plot(plt_X, plt_Y, 'b-', label='Prediction')
GP1.fill(np.concatenate([plt_X, plt_X[::-1]]),
         np.concatenate([plt_Y - 1.9600 * sigma1,
                        (plt_Y + 1.9600 * sigma1)[::-1]]),
         alpha=.5, fc='b', ec='None', label='95% confidence interval')
GP1.set_xlabel('$x$')
GP1.set_ylabel('$f(x)$')
GP1.set_ylim(-10, 20)
GP1.legend(loc='upper left')

GP2.plot(plt_X, f(plt_X), 'r:', label=r'$f(x) = x\,\sin(x)$')
GP2.plot(train_X, train_Y, 'r.', markersize=10, label='Observations')
GP2.plot(plt_X, y_pred, 'b-', label='Prediction')
GP2.fill(np.concatenate([plt_X, plt_X[::-1]]),
         np.concatenate([y_pred - 1.9600 * sigma2,
                        (y_pred + 1.9600 * sigma2)[::-1]]),
         alpha=.5, fc='b', ec='None', label='95% confidence interval')
GP2.set_xlabel('$x$')
GP2.set_ylabel('$f(x)$')
GP2.set_ylim(-10, 20)
GP2.legend(loc='upper left')

plt.show()
