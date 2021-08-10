# 导入程序依赖包
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import time

def load_datasets_mnist():
    dataset_train = pd.read_csv('mnist_train.csv',header=None)  # 读取csv文件，数据类型为DataFrame
    train_y = dataset_train.loc[:, 0]
    train_y = np.array(train_y)  # 转换为array
    train_y = np.eye(10)[train_y].reshape(dataset_train.shape[0], 10).T  # 将array转换为one-hot
    train_x = dataset_train.iloc[:, 1:]  # 将特征数据分割出来
    train_x = np.array(train_x).T.astype(float) / 255.0 * 0.99 + 0.01  # 转换为array

    dataset_test = pd.read_csv('mnist_test.csv',header=None)  # 读取csv文件，数据类型为DataFrame
    test_y = dataset_test.loc[:, 0]
    test_y = np.array(test_y)  # 转换为array
    test_y = np.eye(10)[test_y].reshape(dataset_test.shape[0], 10).T  # 将array转换为one-hot
    test_x = dataset_test.iloc[:, 1:]  # 将特征数据分割出来
    test_x = np.array(test_x).T.astype(float) / 255.0 * 0.99 + 0.01  # 转换为array
    
    return train_x, train_y,  test_x,  test_y

def load_datasets_bankdata():
    data=pd.read_csv('bank-additional-full.csv',sep=';')
    data_x=data.iloc[:,:-1]
    data_y=data.iloc[:,-1]
    data_x=pd.get_dummies(data_x)
    data_y=pd.get_dummies(data_y)
    data_y=data_y.iloc[:,0]

    scaler=MinMaxScaler()
    x_train, x_test, y_train, y_test = train_test_split(data_x, data_y, test_size=0.2, random_state=42)

    x_train = scaler.fit_transform(x_train).T
    y_train = np.eye(2)[y_train].reshape(len(y_train), 2).T.astype(float)
    
    x_test = scaler.transform(x_test).T
    y_test = np.eye(2)[y_test].reshape(len(y_test), 2).T.astype(float)    
    return x_train, y_train,  x_test,  y_test

def relu(Z):
    A = np.maximum(0, Z)  # 将Z中所有负数置0
    cache = Z  # 将Z存入缓存区
    return A, cache


def relu_backward(dA, cache):
    Z = cache  # 从缓存区中取出Z
    dZ = np.array(dA, copy=True)  # 将dA进行复制，并转换为array赋值给dZ
    dZ[Z <= 0] = 0  # 将dZ中负数部分置0
    return dZ

def softmax(Z):
    A = np.exp(Z) / (np.sum(np.exp(Z), axis=0))  # 计算激活值
    cache = Z  # 将Z存入缓存区
    return A, cache

def initialize_parameters(layers_dims, seed):
    seed = max(seed, 1)  # 随机种子要大于0
    np.random.seed(seed)  # 为了实验能够复现，固定随机种子
    parameters_init = {}  # 初始化参数字典
    L = len(layers_dims)  # 网络层数
    for l in range(1, L):
        parameters_init["W" + str(l)] = np.random.randn(layers_dims[l], layers_dims[l - 1]) / np.sqrt(
            layers_dims[l - 1])  # 对网络权重进行抑梯度异常初始化
        parameters_init["b" + str(l)] = np.zeros((layers_dims[l], 1))  # 对网络偏置进行零初始化
    return parameters_init


def linear_forward(A, W, b):
    Z = np.dot(W, A) + b  # 进行矩阵乘法
    cache = (A, W, b)  # 将输入参数存入缓冲区
    return Z, cache


def linear_activation_forward(A_prev, W, b):
    Z, linear_cache = linear_forward(A_prev, W, b)  # 本层神经元的线性运算
    A, activation_cache = relu(Z)  # 本层神经元的激活函数运算
    cache = (linear_cache, activation_cache)  # 将本层的相关参数存入缓存区
    return A, cache


def model_forward(X, parameters, keep_prob):
    caches = []  # 开辟数据缓存区
    # np.random.seed()  # 取消随机种子效果，实现dropout，其实也可以注释掉，同样有正则化的效果，并且结果能够复现
    L = len(parameters) // 2  # 计算神经网络层数
    # 神经网络第一层，输入层的激活值为输入样本
    A = X
    # 神经网络的隐藏层[1,L-1]为Liner + relu
    for l in range(1, L):
        A_prev = A  # 本层的输入值为上一层的激活值
        A, cache = linear_activation_forward(A_prev, parameters['W' + str(l)], parameters['b' + str(l)])  # 计算一层网络的前向传播
        # dropout正则化，防止网络过拟合
        D = np.random.rand(A.shape[0], A.shape[1])  # 按照激活值的大小随机初始化矩阵dropout矩阵D
        D = D < keep_prob  # 使​​用keep_prob作为阈值，将D矩阵中大于keep_prob的元素值置0
        A = A * D  # 舍弃A的一些节点(D中为0的节点将被舍弃)
        A = A / keep_prob  # 缩放未舍弃的节点(不为0)的值
        D_cache = (D, cache)  # 保存dropout矩阵D
        caches.append(D_cache)  # 将dropout矩阵D和每一层的数据缓存区存入前向缓存区
    # 神经网络的输出层为Liner + softmax
    Z, linear_cache = linear_forward(A, parameters['W' + str(L)], parameters['b' + str(L)])  # 输出层的线性计算
    AL, activation_cache = softmax(Z)  # 输出层的softmax激活函数计算
    cache = (linear_cache, activation_cache)  # 保存输出层的中间参数
    caches.append(cache)  # 将输出层的中间参数存入前向缓存区
    return AL, caches


def compute_cost(AL, Y):
    m = Y.shape[1]  # 样本数
    cost = -np.sum(np.multiply(np.log(AL), Y)) / m  # 计算交叉熵
    cost = np.squeeze(cost)  # 数据降维，将array变为标量
    return cost


def linear_backward(dZ, cache):
    A_prev, W, b = cache  # 从缓存区中取得数据
    m = A_prev.shape[1]  # 上一层网络的神经元个数
    dW = np.dot(dZ, A_prev.T) / m  # 计算权值梯度
    db = np.sum(dZ, axis=1, keepdims=True) / m  # 计算偏置梯度
    dA_prev = np.dot(W.T, dZ)  # 计算反向传播时上一层网络激活函数的输入
    return dA_prev, dW, db


def linear_activation_backward(dA, cache):
    linear_cache, activation_cache = cache  # 取得相关数据
    dZ = relu_backward(dA, activation_cache)  # 激活函数反向传播
    dA_prev, dW, db = linear_backward(dZ, linear_cache)  # 线性部分反向传播
    return dA_prev, dW, db


def model_backward(AL, Y, caches, keep_prob):
    grads = {}  # 初始化梯度字典
    L = len(caches)  # 计算网络层数
    Y = Y.reshape(AL.shape)  # 匹配预测矩阵和真实标签的维度
    # 输出层linear + softmax的反向传播
    current_cache = caches[L - 1]  # 取得输出层缓冲区
    linear_cache, activation_cache = current_cache  # 取得输出层的前向数据
    dZ = AL - Y  # 损失函数对softmax输入的导数
    A_prev, W, b = linear_cache  # 取得输出层的前向数据
    m = A_prev.shape[1]  # 前一层神经元个数
    dW = np.dot(dZ, A_prev.T) / m  # 输出层的权重梯度
    db = np.sum(dZ, axis=1, keepdims=True) / m  # 输出层的偏置梯度
    dA_prev = np.dot(W.T, dZ)  # 计算前一层的网络激活函数的输入
    grads["dA" + str(L)], grads["dW" + str(L)], grads["db" + str(L)] = dA_prev, dW, db  # 将梯度存入字典
    # 隐藏层到输入层[L-1,0]linear + relu的反向传播
    for l in reversed(range(L - 1)):
        D, current_cache = caches[l]  # 取出数据
        grades = grads["dA" + str(l + 2)]  # 传递输出层的梯度
        # dropout正则化的反向传播
        grades = grades * D  # 使用正向传播期间相同的节点，舍弃那些关闭的节点
        grades = grades / keep_prob  # 缩放未舍弃的节点(不为0)的值
        dA_prev_temp, dW_temp, db_temp = linear_activation_backward(grades, current_cache)  # 当前层的反向传播
        grads["dA" + str(l + 1)] = dA_prev_temp  # 存入当前层的激活值梯度
        grads["dW" + str(l + 1)] = dW_temp  # 存入当前层的权值梯度
        grads["db" + str(l + 1)] = db_temp  # 存入当前层的偏置梯度
    return grads


def update_parameters(parameters, grads, learning_rate):
    L = len(parameters) // 2  # 神经网络层数
    for l in range(L):
        parameters["W" + str(l + 1)] = parameters["W" + str(l + 1)] - learning_rate * grads["dW" + str(l + 1)]  # 更新权值
        parameters["b" + str(l + 1)] = parameters["b" + str(l + 1)] - learning_rate * grads["db" + str(l + 1)]  # 更新偏置
    return parameters


def predict(X, Y, parameters):
    L = len(parameters) // 2  # 神经网络层数
    A = X  # 输入层的激活值
    # 隐藏层的前向传播[1，L-1]，linear + relu
    for l in range(1, L):
        A_prev = A  # 本层神经元的激活值为下一层神经元的输入
        A, cache = linear_activation_forward(A_prev, parameters['W' + str(l)], parameters['b' + str(l)])  # 本层神经元的前向传播
    # 输出层的前向传播，linear + softmax
    Z, linear_cache = linear_forward(A, parameters['W' + str(L)], parameters['b' + str(L)])  # 输出层的前向传播线性部分
    Y_pre, caches = softmax(Z)  # softmax激活函数
    Y_pre = list(map(lambda x: x == max(x), Y_pre.T)) * np.ones(shape=Y_pre.T.shape)  # 将预测矩阵转换为one-hot，m*4矩阵
    Y_pre = Y_pre.T  # 将预测矩阵转置为4*m矩阵
    acc = np.sum(Y * Y_pre) / Y.shape[1]  # 计算预测准确率
    return acc, Y_pre.T


def model(train_features, train_labels, test_features, test_labels, layers_dims, seed, learning_rate,
          keep_prob):
    check = 1000
    iters = 1  # 记录迭代次数
    acc_best = 0  # 记录测试集准确率
    parameters = initialize_parameters(layers_dims, seed)  # 抑梯度异常随机初始化

    while True:
        AL, caches = model_forward(train_features, parameters, keep_prob)  # 前向传播
        cost = compute_cost(AL, train_labels)  # 计算交叉熵损失
        grads = model_backward(AL, train_labels, caches, keep_prob)  # 误差反向传播
        parameters = update_parameters(parameters, grads, learning_rate)  # 更新参数
        if iters % check == 0:
            acc_testing, label_pre = predict(test_features, test_labels, parameters)
            if acc_testing - acc_best <= 0.0001:
                break
            acc_best = acc_testing
            print(acc_best)
        iters += 1
    iters = iters - check
    return acc_best,iters


if __name__ == '__main__':
    #####################################数据
    train_x, train_y, test_x, test_y = load_datasets_mnist()

    seed = 3  # 程序随机种子
    parms = [[[784, 512, 10],0.1,0.5],
             [[784, 256, 10],0.01,0.5],
             [[784, 256, 10],0.1,0.3],
             [[784, 256, 10],0.1,0.7],
             [[784, 256, 10],0.1,1]]

##########################参数##########################
    # layers = [784 , 200 ,10]  # 设置神经网络结构
    # learning_rate=0.1  # 设置学习率
    # keep_prob = 1  # dropout率
##########################参数##########################
    for i in range(len(parms)):
        start = time.time()  # 记录当前时间
        # 搭建神经网络，利用训练数据进行训练，在训练过程中利用测试数据进行模型评估
        acc_best,iters = model(train_x, train_y, test_x, test_y, layers_dims = parms[i][0], seed=seed,
                           learning_rate=parms[i][1], keep_prob=parms[i][2])

        # 对测试集进行预测，返回测试集准确率和预测类别
        print("############# 网络结构为" + str(parms[i][0]),"  学习率 = " + str(parms[i][1]),"  dropout率 = " + str(parms[i][2]),"#############")
        print("测试集准确率为" + str(acc_best))
        print("迭代次数为" + str(iters))
        print("程序运行时间为：" + str(time.time() - start))
