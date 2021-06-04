import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets.samples_generator import make_classification


class LDA():
    def Train(self, X, y):
        ####X为训练数据集，y为label 
        X0 = np.array([X[i] for i in range(len(X)) if y[i] == 0])
        X1 = np.array([X[i] for i in range(len(X)) if y[i] == 1])

        # 求两个类的均值向量
        u0 = np.mean(X0, axis=0)
        u1 = np.mean(X1, axis=0)

        # dot(a, b, out=None) 计算类内散度矩阵
        cov0 = np.dot((X0 - u0).T, (X0 - u0))  ##类0散度矩阵
        cov1 = np.dot((X1 - u1).T, (X1 - u1))  ##类1散度矩阵
        Sw = cov0 + cov1

        # 计算投影向量w
        w = np.dot(np.mat(Sw).I, (u0 - u1).reshape((len(u0), 1)))

        # 记录训练结果
        self.u0 = u0  # 第0类的分类中心
        self.cov0 = cov0
        self.u1 = u1  # 第1类的分类中心
        self.cov1 = cov1
        self.Sw = Sw  # 类内散度矩阵
        self.w = w  # 判别权重矩阵

    def Test(self, X, y):
        ####X为测试数据集，y为label

        # 分类结果
        y_new = np.dot((X), self.w)

        # 计算fisher线性判别式
        nums = len(y)
        c0 = np.dot((self.u0 - self.u1).reshape(1, (len(self.u0))), np.mat(self.Sw).I)
        c1 = np.dot(c0, (self.u0 + self.u1).reshape((len(self.u0), 1)))
        c = 1/2 * c1  # 2个分类的中心
        h = y_new - c

        # 判别
        y_hat = []
        for i in range(nums):
            if h[i] >= 0:
                y_hat.append(0)
            else:
                y_hat.append(1)

        # 计算分类精度
        count = 0
        for i in range(nums):
            if y_hat[i] == y[i]:
                count += 1
        precise = count / nums

        # 显示信息
        print("Numbers of test samples:", nums)
        print("Numbers of predict correct samples:", count)
        print("Test precise:", precise)

        return precise


if '__main__' == __name__:
    # 产生分类数据
    n_samples = 500
    X, y = make_classification(n_samples=n_samples, n_features=2, n_redundant=0, n_classes=2,
                               n_informative=1, n_clusters_per_class=1, class_sep=0.5, random_state=10)

    # LDA线性判别分析(二分类)
    lda = LDA()
    # 60% 用作训练，40%用作测试
    Xtrain = X[:299, :]
    Ytrain = y[:299]
    Xtest = X[300:, :]
    Ytest = y[300:]
    lda.Train(Xtrain, Ytrain)
    precise = lda.Test(Xtest, Ytest)

    # 原始数据
    plt.scatter(X[:, 0], X[:, 1], marker='o', c=y)
    plt.xlabel("x1")
    plt.ylabel("x2")
    plt.title("Test precise:" + str(precise))
    plt.show()
