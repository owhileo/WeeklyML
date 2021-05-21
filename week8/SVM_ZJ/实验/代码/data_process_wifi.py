import numpy as np
import pandas as pd
import collections
from sklearn.model_selection import train_test_split
def data_process():
    wifi = np.loadtxt('../wifi_data/wifi_localization.txt')
    length = wifi.shape[0]
    dim = wifi.shape[1]
    d = collections.Counter(wifi[:,dim-1])

    wifi[-d[4]:,dim-1]  = 1
    wifi[d[1]:d[1]+d[2]+d[3],dim-1] = -1

    wifi_X = wifi[:,:dim-1]
    wifi_Y = wifi[:,dim-1]

    X_train, X_test, Y_train, Y_test = train_test_split(wifi_X, wifi_Y, test_size=0.3,random_state=6)

    return X_train, X_test, Y_train, Y_test


