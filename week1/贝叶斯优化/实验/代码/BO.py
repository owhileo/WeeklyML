import numpy as np


###Gaussian progress
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
    # def matern52_kernel(self,x1, x2, l=1.0):
    #     m, n = x1.shape[0], x2.shape[0]
    #     kernel_matrix = np.zeros((m, n), dtype=float)
    #     for i in range(m):
    #         for j in range(n):
    #             kernel_matrix[i][j] = (1 + 5**0.5*(np.sum((x1[i] - x2[j]) ** 2))**0.5/l + 5*np.sum((x1[i] - x2[j]) ** 2)/(3*l**2))*np.exp(-5**0.5*(np.sum((x1[i] - x2[j]) ** 2))**0.5/l)
    #     return kernel_matrix

    # 使用拟牛顿法进行优化只需要梯度函数即可
    # def nllloss(self,params):
    #     self.params["l"], self.params["sig"] = params[0], params[1]
    #     KNN = self.RBF_kernel(self.train_X, self.train_X)
    #     loss = 0.5 * self.train_y.T.dot(np.linalg.inv(KNN)).dot(self.train_y) + \
    #            0.5 * np.linalg.slogdet(KNN)[1] + \
    #            0.5 * len(self.train_X) * np.log(2 * np.pi)
    #     return loss.ravel()

    # 一阶梯度函数
    def grad(self,hypp):
        KNN = self.RBF_kernel(self.train_X,self.train_X,hypp[0],hypp[1])
        m = self.train_X.shape[0]  ##数据个数
        dl = np.zeros((m, m), dtype=float)
        for i in range(m):
            for j in range(m):
                dl[i][j] = np.sum((self.train_X[i] - self.train_X[j]) ** 2 / hypp[0] ** 3)
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
            #d_k = -np.linalg.pinv(B_k).dot(g_k)  ##搜索方向
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

##acquisition progress
class AC():
    ##数据去重

    def unique(self,arr):
        n = len(arr[:])
        i = 0
        while (i < n):
            for j in range(0, n):
                if i == j:
                    continue
                if np.sum((arr[j] - arr[i])!= 0) == 0:
                    arr = np.delete(arr, j, 0)
                    n = n - 1
                    i = i - 1
                    break
            i = i + 1
        return arr

    ##构造二维pareto前沿，x为x轴上界，y为y轴上界
    def Append(self,arr,x,y):
        arr = self.bubble_sort(arr,1)
        n = len(arr[:])
        for k in range(0, n):
            if k == n - 1:
                if k == 0:
                    arr = np.concatenate((arr, [[arr[0][0], y]]))  # 需调整边界Y
                    arr = np.concatenate((arr, [[x, arr[k][1]]]))  # 需调整边界X
                    # print(arr,n-1)
                    # input()
                    break
                else:
                    arr = np.concatenate((arr, [[x, arr[k][1]]]))  # 需调整边界X
                    # print(arr,n-1)
                    # input()
                    break
            if arr[k][0] == arr[k + 1][0]:
                continue
            if k == 0:
                arr = np.concatenate((arr, [[arr[0][0], y]]))  # 需调整边界Y
                # print(arr,0)
                # input()
            arr = np.concatenate((arr, [[arr[k + 1][0], arr[k][1]]]))
            # print(arr,3)
            # input()
        return arr

    def pareto(self,arr):
        dim = arr.shape[1]  # 计算目标值维度
        if dim <= 1:
            return
        arr=self.bubble_sort(arr)
        n = len(arr[:])
        i = 0
        while (i < n):
            for j in range(i+1, n):
                if np.sum((arr[j] - arr[i]) > 0) <= 0:
                    arr = np.delete(arr, i, 0)
                    n = n - 1
                    i = i - 1
                    break
            i = i + 1
        return arr

    ##从第i维开始排序,稳定排序
    def bubble_sort(self,arr,i=0):
        dim = arr.shape[1]  # 计算目标值维度
        if i > dim - 1:
            print("排序维度超出数据维度")
            return
        t = np.zeros((1,dim))
        n = len(arr[:])
        while (i <= dim - 1):
            for j in range(0, n - 1):
                for k in range(0, n - 1 - j):
                    if arr[k][i] < arr[k + 1][i]:
                        t[0] = arr[k]
                        arr[k] = arr[k + 1]
                        arr[k + 1] = t[0]
            i = i + 1
        arr = self.unique(arr)  # 数据去重
        return arr

    ###一维，AC函数
    ###UCB AC函数
    def UCB(self,kappa=1.0):
        Ycand = self.means + kappa * np.sqrt(self.var)
        index = np.argmax(Ycand)
        Xcand = self.X[index]
        return Xcand

    ###>=2维时，AC函数
    ###maxmin子函数，min
    def estimate(self,pf, Point):
        # pf = pareto(Y_train)
        mi = 0
        ma = float("-inf")
        for p in pf:
            mi = np.min(Point - p)
            if mi > ma:
                ma = mi
        if ma < 0:
            return -ma
        else:
            return 0

    ###maxmin AC函数，参数为X数组，及其对应的means和var数组
    def mami(self):
        # AC部分
        dim = self.means.shape[0]  # 计算维度
        n = np.shape(self.X)[0]
        # pf = pareto(Y_train)  #######################Y_train

        Ycand = np.zeros((n, 1))
        conv = np.zeros((dim, dim))
        i = 0
        while i < n:
            # print(i)
            Sum = 0
            mean = np.array([self.means[k][i] for k in range(dim)])  # 均值
            ###方差计算
            for k in range(dim):
                conv[k][k] = self.var[k][i]
            Yrandom = np.random.multivariate_normal(mean=mean, cov=conv, size=20)
            for Point in Yrandom:
                Sum = + self.estimate(self.pf, Point)
            Ycand[i] = Sum
            i = i + 1
        index = np.argmax(Ycand) ##max
        Xcand = self.X[index]
        return Xcand

###bayesian optimization
class BO(AC):
    ###输入参数皆为二维数组
    ###参数domain为X取值的范围，返回一个X数组，及其对应的means和var数组
    def __init__(self):
        self.dim = 0
        self.models = None
        self.X = None
        self.means = None
        self.var = None
        self.pf = None
        
    def Create_GP(self,X_train, Y_train, domain):
        ##创建GP模型
        self.dim = Y_train.shape[1]  # 计算维度
        self.models = [GP() for i in range(self.dim)]
        
        for i in range(self.dim):
            self.models[i].fit(X_train,Y_train[:,i])
        dx = X_train.shape[1]
        n = 200
        self.X = np.random.uniform(low=domain[:, 0], high=domain[:, 1], size=(n, dx))
        self.means = np.zeros((self.dim, n))
        self.var = np.zeros((self.dim, n))
        for i in range(self.dim):
            mean, sigma = self.models[i].predict(self.X)  ##此处输出sigma为协方差矩阵
            self.means[i] = mean
            self.var[i] = sigma.diagonal()
            
        self.pf = AC.pareto(self,Y_train)
        return self.X,self.means,self.var
        
    def acquire(self):
        if self.dim >= 2:
            return AC.mami(self)
        else:
            return AC.UCB(self)

    ##输出用于绘制二维空间pareto前沿点
    def fun_plot(self):
        arr = AC.Append(self,self.pf,15,15)
        arr = AC.bubble_sort(self,arr,1)
        return arr


# np.random.seed(1)
#
# def f1(x):
#     """The function to predict."""
#     return x * np.sin(x)
#
# def f2(x):
#     return x*np.cos(x)
#
# X = np.random.uniform(low=1, high=10, size=(10, 1))
# Y = np.zeros((10,2))
# Y[:,0] = f1(X).ravel()
# Y[:,1] = f2(X).ravel()
#
# bo = BO()
# print(bo)
# X, means, var = bo.Create_GP(X,Y,np.array([[1,10]]))
# Xcand = bo.acquire()
