# -*- coding: utf-8 -*-
"""
Created on Tue Mar 16 16:52:26 2021

@author: nijie
"""
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 16 13:45:29 2021

@author: nijie
"""

# -*- coding: utf-8 -*-
"""
Created on Fri Mar 12 15:56:24 2021
@author: nijie
"""
from math import log
import numpy as np
import pandas as pd
import random
import operator
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.model_selection  import cross_val_score
import math
import sklearn.tree as st

#属性和标签的数据结构
class MyDataset:
    def __init__(self, X,Y):
        self.x=X
        self.y=Y
    def get_label(self):
        return self.y
    
    def get_atrribute(self):
        return self.x
    def get_len(self):
        return len(self.y)

#决策树分类器
class MyDecisionTreeClassifier:
    def __init__(self,min_sample_size=0,max_depth=100000,pruning=False):
        self.Tree=None
        self.min_sample_size=min_sample_size
        self.max_depth=max_depth
        self.prune_state=pruning
        
        
    #计算信息熵的函数
    def entropyT(self,dataSet):
        label=dataSet.get_label()
        numEntries = len(label)  
        labelCounts = {}   
        for currentLabel in label: 
            #currentLabel = featVec[-1]  
            if currentLabel not in labelCounts.keys(): labelCounts[currentLabel] = 0
            labelCounts[currentLabel] += 1
        shannonEnt = 0.0  
        for key in labelCounts:
            prob = float(labelCounts[key])/numEntries
            shannonEnt -= prob * log(prob,2) 
        return shannonEnt
    
    
    #根据给定的标签和值划分左右子树
    def splitDataSetRT(self,dataSet, axis, value): 
        X=dataSet.get_atrribute()
        Y=dataSet.get_label()
        col=X[:,axis]
        index=np.where(col>=value)
        #index=(col>=value)
        X=X[index]
        Y=Y[index]
        return MyDataset(X,Y)
    
    
    def splitDataSetLT(self,dataSet, axis, value): 
        X=dataSet.get_atrribute()
        Y=dataSet.get_label()
        col=X[:,axis]
        index=np.where(col<value)
        #index=(col>=value)
        X=X[index]
        Y=Y[index]
        return MyDataset(X,Y)

    #在当前数据集中选择最优的划分属性和划分值
    def chooseBestFeatureToSplitMT(self,dataSet):
        X=dataSet.get_atrribute()
        Y=dataSet.get_label()
        numFeatures = len(X[0])    
        baseEntropy = self.entropyT(dataSet) 
        bestInfoGain = 0.0; bestFeature = -1   
        splitValueA=-100
        for i in range(numFeatures):       
            featList = X[:,i]
            featList=np.unique(featList)
            uniqueVals=np.sort(featList)       
            newEntropy = 0.0
            minEntropy=100000
            splitValue=-100
            for value in uniqueVals: 
                subDataSetR = self.splitDataSetRT(dataSet, i, value)
                subDataSetL = self.splitDataSetLT(dataSet, i, value)
                probL = subDataSetL.get_len()/float(dataSet.get_len())
                probR = subDataSetR.get_len()/float(dataSet.get_len())
                newEntropy = probL * self.entropyT(subDataSetL)+probR * self.entropyT(subDataSetR)
                if(newEntropy<minEntropy):
                    minEntropy=newEntropy
                    splitValue=value
            infoGain = baseEntropy - minEntropy     
            if (infoGain > bestInfoGain):      
                bestInfoGain = infoGain       
                bestFeature = i
                splitValueA=splitValue
        crit=[bestFeature,splitValueA]
        return crit
    
    #返回当前出现次数最多的标签
    def majorityCnt(self,classList):
        classCount={}
        for vote in classList:
            if vote not in classCount.keys(): classCount[vote] = 0
            classCount[vote] += 1
        sortedClassCount = sorted(classCount.items(), key=operator.itemgetter(1), reverse=True)
        return sortedClassCount[0][0]
    
    
    #从深度0开始建树
    def createTreeMT(self,dataSet,depth=0):
        dp=depth
        X=dataSet.get_atrribute()
        Y=dataSet.get_label()
        classList =Y
        if sum(classList==classList[0]) == len(classList):
            return classList[0]
        #所有标签都相同,返回这个标签
        my_array=X
        c=0
        for array in my_array:
            if(not np.array_equal(array,my_array[0])):
                c=1
        if(c==0):
            return self.majorityCnt(classList)
        #所有属性值都相同，返回出现频率最高的标记
        if len(classList)<self.min_sample_size:
            return self.majorityCnt(classList)
        #样本数目小于min_sample_size,返回出现频率最高的标记
        if dp>self.max_depth:
            return self.majorityCnt(classList)
        #树的深度大于max_depth,返回出现频率最高的标记
        if self.prune_state==True:
            train_X,test_X,train_Y,test_Y = train_test_split(X,Y,test_size=0.3,random_state=10)
            label=self.majorityCnt(train_Y)
            count=sum(test_Y==label)
            train_data=MyDataset(train_X, train_Y)
            c = self.chooseBestFeatureToSplitMT(train_data)
            L_prune=self.splitDataSetLT(train_data,c[0],c[1])
            L_count=len(L_prune.get_label())
            R_prune=self.splitDataSetRT(train_data,c[0],c[1])
            R_count=len(R_prune.get_label())
            if L_count>=2:
                leftlabel=self.majorityCnt(L_prune.get_label())
            else:
                leftlabel=label
            if R_count>=2:    
                rightlabel=self.majorityCnt(R_prune.get_label())
            else:
                rightlabel=label
            label_prune=[]
            for currentX in test_X:
                if currentX[c[0]]<c[1]:
                    currentlabel=leftlabel
                else:
                    currentlabel=rightlabel
            label_prune.append(currentlabel)
            count_prune=sum(test_Y==label_prune)
            if(count_prune<count):
                return self.majorityCnt(classList)
        crit = self.chooseBestFeatureToSplitMT(dataSet) 
        L=self.splitDataSetLT(dataSet,crit[0],crit[1])
        R=self.splitDataSetRT(dataSet,crit[0],crit[1])
        result=[crit,self.createTreeMT(L,dp+1),self.createTreeMT(R,dp+1)]
        return result
    
    
    #根据数据集训练模型
    def fit(self,X,Y):
        dataSet=MyDataset(X,Y)
        #self.x=X
        #self.y=Y
        self.Tree=self.createTreeMT(dataSet)  
    
        
    #预测结果
    def predict(self,test):
        myTree=self.Tree
        label=[]
        for meta in test:
            Tree=myTree
            while(not isinstance(Tree,str)):    
                index=Tree[0][0]
                value=Tree[0][1]
                L=Tree[1]
                R=Tree[2]
                if(meta[index]>=value):
                    Tree=R
                else:
                    Tree=L
            metalabel=Tree
            label.append(metalabel)
        return label



if __name__ == '__main__':
    data=pd.read_csv('C://py_doc//iris.csv',header=None)
    #data=pd.read_csv('C://py_doc//wine_quality.csv',header=0)
    
    Y=data.iloc[:,-1]
    coloum_size=data.shape[1]
    X=data.iloc[:,:coloum_size-1]
    X=np.array(X)
    Y=np.array(Y,dtype=str)
    
    #data= np.delete(data, -1, axis=1)
    #label=np.array([label]).T
    #data=np.append(data,label,axis=1)
    #random.shuffle(data)
    
    fold_num=4 #k折交叉验证,手动实现
    myDecisionTree=MyDecisionTreeClassifier(pruning=True,max_depth=10,min_sample_size=10)
    KF=KFold(n_splits=fold_num, shuffle=True, random_state=2021)
    accuracy_kold=[]
    for train_index,test_index in KF.split(X):
        train_X,test_X=X[train_index],X[test_index]
        train_Y,test_Y=Y[train_index],Y[test_index]
        myDecisionTree.fit(train_X,train_Y)
        result=myDecisionTree.predict(test_X)
        count=sum(result==test_Y)
        accuracy=count/len(result)
        accuracy_kold.append(accuracy)
    accuracy_mean=np.array(accuracy_kold).mean()
    print(accuracy_mean)
    
    '''
    myDecisionTree=MyDecisionTreeClassifier(pruning=True,max_depth=10,min_sample_size=10)
    train_X,test_X,train_Y,test_Y = train_test_split(X,Y,test_size=0.2,random_state=2022)
    myDecisionTree.fit(train_X,train_Y)
    result=myDecisionTree.predict(test_X)
    count=sum(result==test_Y)
    accurary=count/len(result)
    print(accurary)
    '''
    
    #x,y=us.shuffle(x,y,random_state=7)
    
    #sklearn直接调包训练
    dt=st.DecisionTreeClassifier(criterion='entropy',max_depth=10,min_samples_split=10)
    scores = cross_val_score(dt, X, Y, cv=fold_num, scoring='accuracy')
    scores_mean=scores.mean()
    print(scores_mean)


    
    








