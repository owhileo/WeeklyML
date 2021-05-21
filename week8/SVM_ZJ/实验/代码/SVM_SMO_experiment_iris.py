from SVM_SMO import *
import numpy as np
from data_process_iris import *
from sklearn import metrics

X1_train, X1_test, Y1_train, Y1_test,\
X2_train, X2_test, Y2_train, Y2_test,\
X3_train, X3_test, Y3_train, Y3_test = data_process()

SVM1 = SVM(X1_train,Y1_train)
SVM1.fit()
Y1_pred = SVM1.predict(X1_test)
acc1 = metrics.accuracy_score(Y1_test, Y1_pred)
r1 = metrics.recall_score(Y1_test, Y1_pred)
p1 = metrics.precision_score(Y1_test, Y1_pred)
f1 = metrics.f1_score(Y1_test, Y1_pred)

SVM2 = SVM(X2_train,Y2_train)
SVM2.fit()
Y2_pred = SVM2.predict(X2_test)
acc2 = metrics.accuracy_score(Y2_test, Y2_pred)
r2 = metrics.recall_score(Y2_test, Y2_pred)
p2 = metrics.precision_score(Y2_test, Y2_pred)
f2 = metrics.f1_score(Y2_test, Y2_pred)

SVM3 = SVM(X3_train,Y3_train)###调参gamma = 5，acc = 0.9333
SVM3.fit()
Y3_pred = SVM3.predict(X3_test)
acc3 = metrics.accuracy_score(Y3_test, Y3_pred)
r3 = metrics.recall_score(Y3_test, Y3_pred)
p3 = metrics.precision_score(Y3_test, Y3_pred)
f3 = metrics.f1_score(Y3_test, Y3_pred)
