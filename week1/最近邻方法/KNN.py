from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.neighbors import KDTree
from sklearn.model_selection import train_test_split
import data_process
import numpy as np
from kd_tree import MyKDTree
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
from sklearn.metrics import classification_report


class LinearTable:
    def __init__(self, X, metric='euclidean'):
        self.X = X
        self.metric = metric

    def query(self, X, k=1):
        if k > len(self.X):
            k = len(self.X)
        dist = np.zeros(len(self.X))
        indices = []
        distance = []
        for x in X:
            if callable(self.metric):
                for i, self_x in enumerate(self.X):
                    dist[i] = self.metric(x, self_x)
            else:
                dist = np.sqrt(((self.X - x) ** 2).sum(axis=1))
            index = np.argsort(dist)[:k]
            distance.append(dist[index])
            indices.append(index)

        return np.array(distance), np.array(indices)


class KNN:
    def __init__(self, n_neighbors=5,
                 weights='uniform', algorithm='brute', metric='euclidean'):
        self.n_neighbors = n_neighbors
        self.weights = weights
        self.algorithm = algorithm
        self.metric = metric

    def fit(self, X, y):
        self.X = X
        self.y = y
        if self.algorithm == 'brute':
            self.table = LinearTable(X, metric=self.metric)
        elif self.algorithm == 'MyKDTree':
            self.table = MyKDTree(X)
        else:
            self.table = KDTree(X)


class KNNClassifier(KNN):
    def __init__(self, n_neighbors=5,
                 weights='uniform', algorithm='brute', metric='euclidean'):
        super().__init__(n_neighbors, weights, algorithm, metric)

    def predict(self, X):
        ret = []
        dist, neighbors_index = self.table.query(X, k=self.n_neighbors)
        neighbors_label = self.y[neighbors_index].tolist()
        if self.weights == 'uniform':
            for i in neighbors_label:
                ret.append(max(i, key=i.count))
        elif self.weights == 'distance':
            for d, i in zip(dist, neighbors_label):
                labels = np.unique(i)
                weights = []
                for label in labels:
                    weights.append(np.sum(1 / (d[np.array(i) == label] + 1)))
                ret.append(labels[np.argmax(weights)])
        else:
            pass
        return np.array(ret)


class KNNRegressor(KNN):
    def __init__(self, n_neighbors=5,
                 weights='uniform', algorithm='brute', metric='euclidean'):
        super().__init__(n_neighbors, weights, algorithm, metric)

    def predict(self, X):
        ret = []
        dist, neighbors_index = self.table.query(X, k=self.n_neighbors)
        neighbors_y = self.y[neighbors_index].tolist()
        if self.weights == 'uniform':
            ret = np.mean(neighbors_y, axis=1)
        elif self.weights == 'distance':
            ret = np.average(neighbors_y, axis=1, weights=1 / (dist+1e-9))
        else:
            pass
        return np.array(ret)


if __name__ == '__main__':
    # ============== 鸢尾花 : 分类 ==============
    iris_data = data_process.load_iris()
    X = iris_data.iloc[:, :-1].values
    y = iris_data.iloc[:, -1].values
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.40, shuffle=True,random_state=1)  # 划分训练集测试集
    # 属性标准化
    scaler = StandardScaler()
    scaler.fit(X_train)
    X_train = scaler.transform(X_train)
    X_test = scaler.transform(X_test)
    # 分别使用库函数和自己实现的程序进行学习和预测
    knn = KNeighborsClassifier()
    knn.fit(X_train, y_train)
    pred_y = knn.predict(X_test)
    print(pred_y.tolist())

    my_knn = KNNClassifier(algorithm='brute')
    my_knn.fit(X_train, y_train)
    my_pred_y = my_knn.predict(X_test)
    print(pred_y.tolist())

    print('结果是否相同:',pred_y.tolist() == my_pred_y.tolist())
    # 性能度量
    print(classification_report(my_pred_y, y_test))

    # ============== forest fire : 回归 ==============
    forestfires_data = data_process.load_forestfires()
    X = forestfires_data.iloc[:, :-1].values
    y = forestfires_data.iloc[:, -1].values
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.40, shuffle=True, random_state=1)  # 划分训练集测试集
    # # 属性标准化
    # scaler = StandardScaler()
    # scaler.fit(X_train)
    # X_train = scaler.transform(X_train)
    # X_test = scaler.transform(X_test)

    my_knn = KNNRegressor()
    my_knn.fit(X_train, y_train)
    my_pred = my_knn.predict(X_test)
    # print(pred.tolist())
    knn = KNeighborsRegressor()
    knn.fit(X_train, y_train)
    pred = knn.predict(X_test)
    # print(pred.tolist())
    print(mean_squared_error(y_test, my_pred))

    # ============== adult : 分类 ==============
    X_train, X_test, y_train, y_test = data_process.load_adult()
    # 分别使用库函数和自己实现的程序进行学习和预测
    knn = KNeighborsClassifier()
    knn.fit(X_train, y_train)
    pred_y = knn.predict(X_test)
    # print(pred_y.tolist())

    my_knn = KNNClassifier(algorithm='KDTree')
    my_knn.fit(X_train, y_train)
    my_pred_y = my_knn.predict(X_test)
    # print(pred_y.tolist())

    # print('结果是否相同:',pred_y.tolist() == my_pred_y.tolist())
    # 性能度量
    from sklearn.metrics import classification_report
    # print(classification_report(pred_y, y_test))
    print('准确率：', np.sum(pred_y == y_test)/len(y_test))
