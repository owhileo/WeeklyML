import numpy as np
import pandas as pd
import collections
from sklearn.model_selection import train_test_split
def data_process():
    iris_raw = pd.read_csv('../iris_data/iris.data',names=[i for i in range(5)])
    length = len(iris_raw)
    iris1 = np.array(iris_raw)
    iris2 = np.array(iris_raw)
    iris3 = np.array(iris_raw)
    
    d = collections.Counter(iris1[:,4])
    
    iris1[0:d['Iris-setosa'],4] = -1
    iris1[d['Iris-setosa']:d['Iris-setosa']+d['Iris-versicolor'],4] = 1
    iris1 = iris1.tolist()
    del iris1[d['Iris-setosa']+d['Iris-versicolor']:length]
    iris1 = np.array(iris1)

    iris2[0:d['Iris-setosa'],4] = -1
    iris2[d['Iris-setosa']+d['Iris-versicolor']:length,4] = 1
    iris2 = iris2.tolist()
    del iris2[d['Iris-setosa']:d['Iris-setosa']+d['Iris-versicolor']]
    iris2 = np.array(iris2)

    iris3[d['Iris-setosa']:d['Iris-setosa']+d['Iris-versicolor'],4] = -1
    iris3[d['Iris-setosa']+d['Iris-versicolor']:length,4] = 1
    iris3 = iris3.tolist()
    del iris3[0:d['Iris-setosa']]
    iris3 = np.array(iris3)

    iris1_X = iris1[:,:-1]
    iris1_Y = iris1[:,-1]
    X1_train, X1_test, Y1_train, Y1_test = train_test_split(iris1_X, iris1_Y, test_size=0.3,random_state=5)

    iris2_X = iris2[:,:-1]
    iris2_Y = iris2[:,-1]
    X2_train, X2_test, Y2_train, Y2_test = train_test_split(iris2_X, iris2_Y, test_size=0.3,random_state=5)

    iris3_X = iris3[:,:-1]
    iris3_Y = iris3[:,-1]
    X3_train, X3_test, Y3_train, Y3_test = train_test_split(iris3_X, iris3_Y, test_size=0.3,random_state=8)
    return X1_train, X1_test, Y1_train, Y1_test,\
           X2_train, X2_test, Y2_train, Y2_test,\
           X3_train, X3_test, Y3_train, Y3_test


