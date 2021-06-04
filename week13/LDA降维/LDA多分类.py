import numpy as np
from sklearn.datasets import load_iris
import matplotlib.pyplot as plt
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis


def LDA_dimensionality(X, y, k):
    ###X为数据集，y为label，k为目标维数
    
    label_ = list(set(y))

    X_classify = {}

    for label in label_:
        Xi = np.array([X[i] for i in range(len(X)) if y[i] == label])
        X_classify[label] = Xi

    u = np.mean(X, axis=0)
    u_classify = {}

    for label in label_:
        ui = np.mean(X_classify[label], axis=0)
        u_classify[label] = ui

    # 计算类内散度矩阵
    # Sw = np.dot((X - u).T, X - u)
    Sw = np.zeros((len(u), len(u)))
    for i in label_:
        Sw += np.dot((X_classify[i] - u_classify[i]).T,
                     X_classify[i] - u_classify[i])

    # 计算类间散度矩阵
    # Sb=St-Sw
    Sb = np.zeros((len(u), len(u)))
    for i in label_:
        Sb += len(X_classify[i]) * np.dot((u_classify[i] - u).reshape(
            (len(u), 1)), (u_classify[i] - u).reshape((1, len(u))))

    eig_vals, eig_vecs = np.linalg.eig(
        np.linalg.inv(Sw).dot(Sb))  # 计算Sw-1*Sb的特征值和特征矩阵

    # 按从小到大排序，输出排序指示值
    sorted_indices = np.argsort(eig_vals)
    # 反转
    sorted_indices = sorted_indices[::-1]
    # 提取前k个特征向量
    topk_eig_vecs = eig_vecs[:, sorted_indices[0:k:1]]
    
    """s[i:j:k]，i起始位置,j终止位置，k表示步长，默认为1
    s[::-1]是从最后一个元素到第一个元素复制一遍（反向）
    """
    ####OR
    # # 按从小到大排序，输出排序指示值
    # sorted_indices = np.argsort(eig_vals)
    # # 提取前k个特征向量
    # topk_eig_vecs = eig_vecs[:, sorted_indices[:-k - 1:-1]]

    return topk_eig_vecs


if '__main__' == __name__:

    iris = load_iris()
    X = iris.data
    y = iris.target

    W = LDA_dimensionality(X, y, 2)
    X_new = np.dot((X), W)  # 估计值

    plt.figure(1)
    plt.scatter(X_new[:, 0], X_new[:, 1], marker='o', c=y)
    plt.title("LDA reducing dimension - our method")
    plt.xlabel("x1")
    plt.ylabel("x2")

    # 与sklearn中的LDA函数对比
    lda = LinearDiscriminantAnalysis(n_components=2)
    lda.fit(X, y)
    X_new = lda.transform(X)
    X_new = - X_new  # 为了对比方便，取个相反数，并不影响分类结果
    print(X_new)
    plt.figure(2)
    plt.scatter(X_new[:, 0], X_new[:, 1], marker='o', c=y)
    plt.title("LDA reducing dimension - sklearn method")
    plt.xlabel("x1")
    plt.ylabel("x2")

    plt.show()
