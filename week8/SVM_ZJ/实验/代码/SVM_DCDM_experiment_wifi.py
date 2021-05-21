from SVM_DCDM import *
import numpy as np
from data_process_wifi import *
from sklearn import metrics
from sklearn import svm

X_train, X_test, Y_train, Y_test = data_process()

# SVM1 = LSVM(X_train,Y_train)
# SVM1.fit()
# Y_pred = SVM1.predict(X_test)

SVM2 = svm.SVC()
SVM2.fit(X_train,Y_train)
y_pred = SVM2.predict(X_test)

# acc1 = metrics.accuracy_score(Y_test, Y_pred)
# r1 = metrics.recall_score(Y_test, Y_pred)
# p1 = metrics.precision_score(Y_test, Y_pred)
# f1 = metrics.f1_score(Y_test, Y_pred)

acc2 = metrics.accuracy_score(Y_test, y_pred)
r2 = metrics.recall_score(Y_test, y_pred)
p2 = metrics.precision_score(Y_test, y_pred)
f2 = metrics.f1_score(Y_test, y_pred)
###6
