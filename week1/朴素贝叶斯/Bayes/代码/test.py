# from bayes import Bayes
# import pandas as pd


# if __name__ == '__main__':
#     #对数据的处理
#     classify=[' <=50K',' >50K']
#     cla_data=[] ##为每个classify建立对应的样本存储
#     f=pd.read_table("train_adult.txt",sep=",")
#     for i in range(len(classify)):
#         temp=f.loc[f["classify"] == classify[i]]
#         cla_data.append(temp)
#     attribute=[1,0,1,0,1,0,0,0,0,0,1,1,1,0]
#     adult=Bayes("test_adult.txt",classify,cla_data,attribute)
#     mean,var=adult.Mean_Var()
#     rate=adult.class_pro(len(f))
#     adult.forecast(rate,mean,var)
# #对数据的处理（iris数据集）
#     classify=['Iris-setosa','Iris-versicolor','Iris-virginica']
#     cla_data=[] ##为每个classify建立对应的样本存储
#     f=pd.read_table("train_iris.txt",sep=",")
#     for i in range(len(classify)):
#         temp=f.loc[f["classify"] == classify[i]]
#         cla_data.append(temp)
#     attribute=[1,1,1,1]
#     iris=Bayes("test_iris.txt",classify,cla_data,attribute)
#     mean,var=iris.Mean_Var()
#     rate=iris.class_pro(len(f))
#     iris.forecast(rate,mean,var)

#sklearn 实现贝叶斯 ---
from sklearn import datasets
iris=datasets.load_iris()
from sklearn.naive_bayes import GaussianNB
gnb=GaussianNB()
y_pred=gnb.fit(iris.data,iris.target).predict(iris.data)
print(iris.target==y_pred)