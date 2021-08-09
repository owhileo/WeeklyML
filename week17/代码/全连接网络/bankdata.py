import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import torch
from torch.utils.data import DataLoader,TensorDataset
from torch import nn, optim
import numpy as np
import matplotlib.pyplot as plt
import itertools


data=pd.read_csv('./data/bank-additional-full.csv',sep=';')

data_x=data.iloc[:,:-1]
data_y=data.iloc[:,-1]

data_x=pd.get_dummies(data_x)#变成onehot形式
data_y=pd.get_dummies(data_y)
data_y=data_y.iloc[:,0]

scaler=MinMaxScaler()

X_train, X_test, y_train, y_test = train_test_split(data_x, data_y, test_size=0.2, random_state=42)

X_train=scaler.fit_transform(X_train)
X_test=scaler.transform(X_test)#test数据集采用和训练集一样的fit，所以只需要用transform（）就可以了 

deal_dataset=TensorDataset(torch.tensor(X_train,dtype=torch.float32),torch.tensor(y_train.values,dtype=torch.long))
test_dataset=TensorDataset(torch.tensor(X_test,dtype=torch.float32),torch.tensor(y_test.values,dtype=torch.long))
train_loader=DataLoader(deal_dataset,32,True)
test_loader=DataLoader(test_dataset,32,False)


#网络训练------------------------------------------------------------------------------------------------------------------


# 定义model
class CF(nn.Module):
    # 多层
    def __init__(self, in_dim, hidden_dim, out_dim):
        super(CF, self).__init__()
        #不能写成list 的模式 否则会报错（在测试阶段 model.eval()函数不起作用 不能抑制BN）
        self.hidden_dim = hidden_dim
        self.layer0 = nn.Sequential(
            nn.Linear(in_dim, hidden_dim[0]), nn.BatchNorm1d(hidden_dim[0]), nn.ReLU(True)
        )
        self.test = nn.Sequential()
        for i in range(len(hidden_dim)-1):
            layer_i = nn.Sequential(nn.Linear(hidden_dim[i], hidden_dim[i+1]), nn.BatchNorm1d(hidden_dim[i+1]), nn.ReLU(True))
            self.test.add_module("layer{}".format(i+1), layer_i)
        self.out = nn.Linear(hidden_dim[len(hidden_dim)-1], out_dim)

    def forward(self, x):
        y = self.layer0(x)
        y = self.test(y)
        y = self.out(y)
        return y


# 主函数入口
if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # 把所有需要改动的变量都写到一起 方便调参
    # learning_rate=1e-2
    # batch_size = 64
    num_eopches = 30
    # 加载数据
    train_set, test_set = train_loader,test_loader
    # 每层网络的节点个数
    num_point = [32,128]
    #设置网络的层数
    hidden_list=[]
    for i in range(1,11):
        hidden_list.append(itertools.product(num_point, repeat=i))

    
    for hidden_num,hidden_iter in enumerate(hidden_list):#不同层数的神经网络
        x_test = []#绘制图片
        y_test = []
        for num, iter in enumerate(hidden_iter):#不同节点设置
            iter = list(iter)
            # print(iter)
            model_ = CF(63, iter, 2)
            model_.to(device=device)
            model_.train(mode=True)
            criterion = nn.CrossEntropyLoss()
            # optimizer=optim.SGD(model.parameters(),lr=learning_rate)
            optimizer = optim.Adam(model_.parameters(),lr=1e-2, betas=(0.9, 0.999), eps=1e-08)

            # 训练
            for i in range(num_eopches):
                for img, label in train_set:
                    img = img.view(img.size(0), -1)
                    img = img.to(device=device)
                    label = label.to(device=device)
                # print(img.size())
                    predict_y = model_(img)
                    loss = criterion(predict_y, label)

                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()


            # 测试
            model_.eval()
            count = 0
            total=0
            for data in test_set:
                img, label = data
            # img.size(0)是指有几张图片,需要将其压缩成一个二维的数据，其中每一行是代表一张图片（全连接的时候都需要这一步）
                img = img.view(img.size(0), -1)
                img = img.to(device=device)
                out = model_(img)
                _, predict_y = torch.max(out, 1)
                total+=label.size(0)
                count+=(label==predict_y).sum().item()
            print("acc={}".format(count/total))

            # 绘图
            x_test.append(num)
            y_test.append(count/total)
            plt.plot(x_test, y_test)
            plt.xlabel('iter')
            plt.ylabel('auc')
            plt.title('batch_size=64  iter={} Adam'.format(iter))
            plt.savefig('./pic/'+'layer{}_'.format(hidden_num+3)+str(num)+'.png')
            print("layer={},iter={},loss is {}".format(hidden_num,iter, loss))
