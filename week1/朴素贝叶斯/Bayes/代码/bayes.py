import numpy as np
import math
import pandas as pd


class Bayes:
    def __init__(self,test,classify,cla_data,attribute):
            super().__init__()
            self.cla_data=cla_data
            self.classify=classify
            self.test=test
            self.attribute=attribute#0/1值 ，记录连续/离散属性
        
    # 求均值
    def Mean_Var(self):
        mean=[]
        var=[]
        for i in range(len(self.cla_data)):
            temp_mean=[]
            temp_var=[]
            for col in self.cla_data[i].columns:
                if(col!="classify"):
                  if(self.attribute[int(col)]==1):
                    temp_mean.append(self.cla_data[i][col].mean())
                    temp_var.append(self.cla_data[i][col].var())
            mean.append(temp_mean)
            var.append(temp_var)
        return mean ,var

    #计算条件概率(连续)
    def condition_pro(self,x,mean,var):
        pro=(1/(math.sqrt(2*math.pi)*var))*math.exp(-(math.pow(x-mean,2))/(2*math.pow(var,2)))
        return pro
    #计算条件概率(离散)
    def condition_dispe(self,x,x_num,classify_num): #x_num 代表第几个属性   
        sum=len(self.cla_data[classify_num])
        # print( self.cla_data[classify_num] )
        num=len(self.cla_data[classify_num][self.cla_data[classify_num][str(x_num)]==x])
        return num/sum

    #计算每个类的概率
    def class_pro(self,sum):
        rate=[]
        for i in range(len(self.classify)):
            rate.append(len(self.cla_data[i])/sum)
        return rate


    #预测(attribute:0/1值 1：连续 0：离散；mean:二维数组，每个类每个属性的mean)
    def forecast(self,rate,mean,var):
      yes=0
      sum=0
      with open(self.test) as f:
        for line in f.readlines():
           sum+=1
           line=line.strip()
           line=line.split(',')
           current=0#记录当前概率最大值
           cur_classify=""# 记录对应的预测分类结果
           
           dispersed=0
           for i in range(len(self.classify)):#每个classify预测
                temp=rate[i]
                continuity=0 #记录第几个连续/离散数据
                for j in range(len(self.attribute)):#每个属性
                   if(self.attribute[j]==1):#是连续的
                       temp*=self.condition_pro(float(line[j]),mean[i][continuity],var[i][continuity])
                       continuity+=1
                   else:#是离散的
                       temp*=self.condition_dispe(line[j],j,i)
                if temp>current:
                    current=temp
                    cur_classify=self.classify[i]
           if(cur_classify.strip()==line[(len(line)-1)].strip()):
                yes+=1
      print(yes/sum)
    

if __name__ == '__main__':
    #对数据的处理(adult 数据集)
    classify=[' <=50K',' >50K']
    cla_data=[] ##为每个classify建立对应的样本存储
    f=pd.read_table("train_adult.txt",sep=",")
    for i in range(len(classify)):
        temp=f.loc[f["classify"] == classify[i]]
        cla_data.append(temp)
    attribute=[1,0,1,0,1,0,0,0,0,0,1,1,1,0]
    adult=Bayes("test_adult.txt",classify,cla_data,attribute)
    mean,var=adult.Mean_Var()
    rate=adult.class_pro(len(f))
    adult.forecast(rate,mean,var)


    # #对数据的处理（iris数据集）
    # classify=['Iris-setosa','Iris-versicolor','Iris-virginica']
    # cla_data=[] ##为每个classify建立对应的样本存储
    # f=pd.read_table("train_iris.txt",sep=",")
    # for i in range(len(classify)):
    #     temp=f.loc[f["classify"] == classify[i]]
    #     cla_data.append(temp)
    # attribute=[1,1,1,1]
    # iris=Bayes("test_iris.txt",classify,cla_data,attribute)
    # mean,var=iris.Mean_Var()
    # rate=iris.class_pro(len(f))
    # iris.forecast(rate,mean,var)



