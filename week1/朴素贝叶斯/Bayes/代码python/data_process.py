import random
import os
import pandas as pd
# train =[]
# test=[]
# # 拆分成训练集和测试集
# with open("iris.txt") as f:
#     num=-1
#     for line in f.readlines():
#         num+=1
#         temp=random.random()
#         if(temp>0.7):
#                 test.append(line)
#         else:
#                 train.append((line))

# for i in range(len(train)):
#         with open('train.txt','a') as f:
#                 f.write(train[i])   

# for i in range(len(test)):
#         with open('test.txt','a') as f:
#                 f.write(test[i])      
# print(len(train))
# print(len(test))

f=pd.read_csv("test_adult.txt",sep=",")
f=f.replace(" >50K.",">50K")
f=f.replace(" <=50K.","<=50K")
f.to_csv('test_adult.txt',index=False,sep=",")
print(f)


