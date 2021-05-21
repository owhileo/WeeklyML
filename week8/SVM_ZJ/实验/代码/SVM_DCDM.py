import random

import numpy as np
from sklearn import datasets

class LSVM:
    def __init__(self,X,Y,eps=1e-3,C = 1.0,max_iter = 10000):
        self.alpha = None  ###初始化为0
        self.G = None      ###初始化为-1
        self.Q = None
        self.L = None
        self.alpha_Y = None
        self.b = None
        self.eps = eps
        self.Y = Y
        self.X = X
        self.C = C
        self.max_iter = max_iter

    ###模型初始化
    def Initial(self):
        self.L = self.Y.shape[0]
        self.alpha = np.zeros(self.L)
        self.G = np.zeros(self.L)
        self.Q = np.zeros((self.L,self.L))
        for i in range(self.L):
            self.G[i] -= 1
            for j in range(self.L):
                self.Q[i][j] =self.Y[i]*self.Y[j]*np.sum(self.X[i] * self.X[j])

    ###DCDM中随机打乱index
    def permutation(self):
        idx = np.array([i for i in range(self.L)])
        random.shuffle(idx)
        return idx

    ###alpha更新
    def update(self):
        for i in range(self.max_iter):
            old_alpha = np.zeros((1,self.L))
            old_alpha[0] = self.alpha
            Idx_per = self.permutation()
            for j in range(self.L):
                idx = Idx_per[j]
                if self.alpha[idx] == 0:
                    PG = min(self.G[idx],0)
                elif self.alpha[idx] == self.C:
                    PG = max(self.G[idx],0)
                else: ##self.alpha[idx] > 0 and self.alpha[idx] < self.C
                    PG = self.G[idx]
                if abs(PG) != 0:
                    alp = max( self.alpha[idx] - self.G[idx]/self.Q[idx][idx] , 0)
                    self.alpha[idx] = min(alp,self.C)

                    delta_alpha = self.alpha[idx] - old_alpha[0][idx]

                    for i in range(self.L):
                        self.G[i] += self.Q[idx][i] * delta_alpha

            if (self.alpha == old_alpha[0]).all():
                    return

                
    ####计算y_i*alpha_i 和 b值
    def calculate(self):
        self.alpha_Y = self.alpha*self.Y

    ###模型训练
    def fit(self):
        self.Initial()
        self.update()
        self.calculate()

    ###模型预测
    def predict(self,x):
        len = x.shape[0]
        value = np.zeros(len)
        for i in range(len):
            for j in range(self.L):
                value[i] += self.alpha_Y[j] * np.sum(x[i] * self.X[j])

        for i in range(len):
            if value[i] > 0:
                value[i] = 1
            else:
                value[i] = -1
        return value

# iris = datasets.load_iris()
# X = iris["data"][:, (2,3)]
# Y = (iris["target"] == 2).astype( np.float64 )
# for i in range(len(Y)):
#     if Y[i] == 0:
#         Y[i] = -1
# S1 = LSVM(X,Y)
# S1.fit()
# res = S1.predict(np.array([[5.5, 1.7]]))
# print(res)
