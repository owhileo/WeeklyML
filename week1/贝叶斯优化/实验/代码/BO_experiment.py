import numpy as np
from matplotlib import pyplot as plt
from BO import *
from matplotlib import gridspec
import warnings
warnings.filterwarnings('ignore')

np.random.seed(1)
def f1(x):
    """The function to predict."""
    return x * np.sin(x)

def f2(x):
    return x*np.cos(x)

###########################################BO one dim
###初始训练集
X1 = np.atleast_2d(np.linspace(1, 10, 7)).T
Y1 = f1(X1)

bo1 = BO()
for i in range(10):
    bo1.Create_GP(X1,Y1,np.array([[1,10]]))
    Xcand = bo1.acquire()
    X1 = np.concatenate((X1, [Xcand]))
    Y1 = np.concatenate((Y1, [f1(Xcand)]))
plt_X = np.atleast_2d(np.linspace(1, 10, 1000)).T
plt_Y, sigma1 = bo1.models[0].predict(plt_X)##此处输出sigma为协方差矩阵
sigma1 = sigma1.diagonal()##获取对角线元素
######################################################


###########################################BO two dim
###初始训练集
X2 = np.atleast_2d(np.linspace(1, 10, 6)).T
Y2 = np.zeros((6,2))
Y2[:,0] = f1(X2).ravel()
Y2[:,1] = f2(X2).ravel()

bo2 = BO()
for i in range(10):
    bo2.Create_GP(X2,Y2,np.array([[1,10]]))
    Xcand = bo2.acquire()
    X2 = np.concatenate((X2, [Xcand]))
    Y2 = np.concatenate((Y2, np.array([f1(Xcand), f2(Xcand)]).reshape(1,2)))
arr = bo2.fun_plot()
######################################################

####绘图
fig = plt.figure(figsize=(16, 10))

gs=gridspec.GridSpec(2,2)#分为2行2列 
BO1 = plt.subplot(gs[:,0])
BO2 = plt.subplot(gs[:,1])

BO1.plot(plt_X, f1(plt_X), 'r:', label=r'$f(x) = x\,\sin(x)$')
BO1.plot(X1, Y1, 'r.', markersize=10, label='Observations')
BO1.plot(plt_X, plt_Y, 'b-', label='Prediction')
BO1.fill(np.concatenate([plt_X, plt_X[::-1]]),
         np.concatenate([plt_Y - 1.9600 * sigma1,
                        (plt_Y + 1.9600 * sigma1)[::-1]]),
         alpha=.5, fc='b', ec='None', label='95% confidence interval')
BO1.set_xlabel('$x$')
BO1.set_ylabel('$f1(x)$')
BO1.set_ylim(-10, 15)
BO1.legend(loc='upper left')

# print(bo2.pf)
BO2.scatter(bo2.pf[:,0], bo2.pf[:,1], c='0')
BO2.plot(arr[:,0], arr[:,1],color='r',label='Pareto front')

BO2.set_xlabel('$f1(x)$')
BO2.set_ylabel('$f2(x)$')
BO2.set_ylim(-10, 15)
BO2.legend(loc='upper left')

plt.show()
