import numpy as np
from matplotlib import pyplot as plt

class GP:

    def __init__(self, optimize=True):
        self.f = False  ##一个flag，标明GPR超参数是否优化
        self.train_X, self.train_Y = None, None
        self.params = {"l": 0.5, "sig": 0.2} ##超参数，依据核函数而定

    # RBF核函数
    def RBF_kernel(self,x1,x2,l=1.0, sig=1.0):
        m, n = x1.shape[0], x2.shape[0]
        kernel_matrix = np.zeros((m, n), dtype=float)
        for i in range(m):
            for j in range(n):
                kernel_matrix[i][j] = sig ** 2 * np.exp(- 1 / (2 * l ** 2) * np.sum((x1[i] - x2[j]) ** 2))
        return kernel_matrix

    ##matern52核函数
    # def matern52_kernel(x1, x2, l=1.0):
    #     m, n = x1.shape[0], x2.shape[0]
    #     kernel_matrix = np.zeros((m, n), dtype=float)
    #     for i in range(m):
    #         for j in range(n):
    #             kernel_matrix[i][j] = (1 + 5**0.5*(np.sum((x1[i] - x2[j]) ** 2))**0.5/l + 5*np.sum((x1[i] - x2[j]) ** 2)/(3*l**2))*np.exp(-5**0.5*(np.sum((x1[i] - x2[j]) ** 2))**0.5/l)
    #     return kernel_matrix

    # 使用拟牛顿法进行优化只需要梯度函数即可
    # def nllloss(params):
    #     self.params["l"], self.params["sig"] = params[0], params[1]
    #     KNN = self.RBF_kernel(self.train_X, self.train_X)
    #     loss = 0.5 * self.train_y.T.dot(np.linalg.inv(KNN)).dot(self.train_y) + \
    #            0.5 * np.linalg.slogdet(KNN)[1] + \
    #            0.5 * len(self.train_X) * np.log(2 * np.pi)
    #     return loss.ravel()

    # 一阶梯度函数
    def grad(self,hypp):
        KNN = self.RBF_kernel(self.train_X,self.train_X,hypp[0],hypp[1])
        m = train_X.shape[0]  ##数据个数
        dl = np.zeros((m, m), dtype=float)
        for i in range(m):
            for j in range(m):
                dl[i][j] = np.sum((train_X[i] - train_X[j]) ** 2 / hypp[0] ** 3)
        # print(1111111)
        # print(KNN)        
        # g_l = -0.5 * self.train_Y.T.dot(np.linalg.inv(KNN)).dot(KNN * dl).dot(np.linalg.inv(KNN)).dot(self.train_Y) + \
        #       0.5 * np.trace(np.linalg.inv(KNN).dot(KNN * dl))
        #
        # g_sig = -0.5 * self.train_Y.T.dot(np.linalg.inv(KNN)).dot(KNN * 2 / hypp[1]).dot(np.linalg.inv(KNN)).dot(self.train_Y) + \
        #     0.5 * np.trace(np.linalg.inv(KNN).dot(KNN * 2 / hypp[1]))

        g_l = -0.5 * self.train_Y.T.dot(np.linalg.pinv(KNN)).dot(KNN * dl).dot(np.linalg.pinv(KNN)).dot(self.train_Y) + \
              0.5 * np.trace(np.linalg.pinv(KNN).dot(KNN * dl))

        g_sig = -0.5 * self.train_Y.T.dot(np.linalg.pinv(KNN)).dot(KNN * 2 / hypp[1]).dot(np.linalg.pinv(KNN)).dot(self.train_Y) + \
            0.5 * np.trace(np.linalg.pinv(KNN).dot(KNN * 2 / hypp[1]))
        return np.asarray([g_l, g_sig]).ravel()

    def fit(self, X, Y):
        #获取训练数据
        self.train_X = np.asarray(X)
        self.train_Y = np.asarray(Y)
        # 基于BFGS的超参数优化
        eps = 1e-2
        k = 0
        B_k = np.eye(2) ##初始单位阵
        hypp_k = np.array([self.params["l"], self.params["sig"]])##初始超参数
        while True:
            g_k = self.grad(hypp_k)  ##梯度
            d_k = -np.linalg.inv(B_k).dot(g_k)  ##搜索方向
            ##确定步长lmd
            lmd = 0.1
            s_k = lmd * d_k
            hypp_k = hypp_k + s_k
            # print(hypp_k)
            if abs(np.sum(self.grad(hypp_k) ** 2)) < eps:
                self.params["l"], self.params["sig"] = hypp_k[0], hypp_k[1]
                break
            y_k = self.grad(hypp_k) - g_k
            y_k = y_k.reshape(-1, 1)#行向量转列向量
            s_k = s_k.reshape(-1, 1)#行向量转列向量
            B_k = B_k + \
                  y_k.dot(y_k.T)/(y_k.T.dot(s_k)) - \
                  B_k.dot(s_k).dot(s_k.T).dot(B_k)/(s_k.T.dot(B_k).dot(s_k))
        self.f = True

##GPR拟合后预测，参数X为预测样本点
    def predict(self, X):
        if not self.f:
            print("GP未拟合无法进行预测.") ##GPR未拟合则模型输出固定超参数的贝叶斯过程
            return
        # print(X)
        X = np.asarray(X)
        KNN = self.RBF_kernel(self.train_X, self.train_X)  # (N, N)
        Knn = self.RBF_kernel(X, X)  # (n+1, n+1)
        KnN = self.RBF_kernel(self.train_X, X)  #(N, n+1)
        KNN_pinv = np.linalg.pinv(KNN)  # (N, N)

        mu = KnN.T.dot(KNN_pinv).dot(self.train_Y)
        cov = Knn - KnN.T.dot(KNN_pinv).dot(KnN)
        return mu, cov

# train_X = np.array([3, 1, 4, 5, 9]).reshape(-1, 1)
# train_Y = np.array([-1, 0.5, -0.6, 0.2, -0.9]).reshape(-1, 1)
# gpr = GP()
# gpr.fit(train_X, train_Y)
# mu,cov = gpr.predict([3.5])
# print(gpr.params["l"], gpr.params["sig"])
# print(mu,cov)




np.random.seed(1)
def f(x):
    """The function to predict."""
    return x * np.sin(x)

train_X = np.atleast_2d([1., 3., 5., 6., 7., 8.]).T
# print(train_X)

# Observations
train_Y = f(train_X).ravel()

# Mesh the input space for evaluations of the real function, the prediction and
# its MSE
plt_X = np.atleast_2d(np.linspace(0, 10, 1000)).T


gp = GP()
gp.fit(train_X,train_Y)
plt_Y, sigma = gp.predict(plt_X)##此处输出sigma为协方差矩阵
sigma = sigma.diagonal()##获取对角线元素
# print(sigma)
# print(plt_Y)
plt.figure()
plt.plot(plt_X, f(plt_X), 'r:', label=r'$f(x) = x\,\sin(x)$')
plt.plot(train_X, train_Y, 'r.', markersize=10, label='Observations')
plt.plot(plt_X, plt_Y, 'b-', label='Prediction')
plt.fill(np.concatenate([plt_X, plt_X[::-1]]),
         np.concatenate([plt_Y - 1.9600 * sigma,
                        (plt_Y + 1.9600 * sigma)[::-1]]),
         alpha=.5, fc='b', ec='None', label='95% confidence interval')
plt.xlabel('$x$')
plt.ylabel('$f(x)$')
plt.ylim(-10, 20)
plt.legend(loc='upper left')
plt.show()
