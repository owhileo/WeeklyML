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


# 构建模型，三层神经网络
class batch_net(nn.Module):
    def __init__(self, in_dim, hidden1_dim, hidden2_dim,hidden3_dim,hidden4_dim ,out_dim):
        super(batch_net, self).__init__()
        self.layer1 = nn.Sequential(nn.Linear(in_dim, hidden1_dim),nn.BatchNorm1d(hidden1_dim), nn.ReLU())
        self.layer2 = nn.Sequential(nn.Linear(hidden1_dim, hidden2_dim), nn.BatchNorm1d(hidden2_dim), nn.ReLU())
        self.layer3 = nn.Sequential(nn.Linear(hidden2_dim, hidden3_dim),nn.BatchNorm1d(hidden3_dim), nn.ReLU())
        self.layer4 = nn.Sequential(nn.Linear(hidden3_dim, hidden4_dim),nn.BatchNorm1d(hidden4_dim), nn.ReLU())
        self.layer5 = nn.Sequential(nn.Linear(hidden4_dim, out_dim))
    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.layer5(x)
        return x

#定义一个初始化权重的方法
def weight_init(net):
    for op in net.modules():
        if isinstance(op,nn.Linear):
            #设置成100，50 acc=0.25  10，50 acc=0.38 0.1 0.1 acc=0.39
            # nn.init.constant_(op.weight.data,val=0.1)
            # nn.init.constant_(op.bias.data,val=0.1)    

            
            # nn.init.normal_(op.weight.data)
            # nn.init.normal_(op.bias.data)

            #   Xavier在tanh中表现的很好，但在Relu激活函数中表现的很差(Xavier用于初始化w)
            # nn.init.xavier_normal_(op.weight.data,gain=nn.init.calculate_gain('relu'))
            # nn.init.normal_(op.bias.data)

            # #效果   kaiming 方法针对xavier方法在relu上效果不好的改进方法
            nn.init.kaiming_normal_(op.weight.data,nonlinearity='relu')
            nn.init.normal_(op.bias.data)

if __name__ == "__main__":
    # 超参数配置
    learning_rate = 1e-2#1e-2
    num_epoches = 25
    # 加载数据集
    train_dataset, test_dataset = train_loader,test_loader
    # 导入网络，并定义损失函数和优化器 
    model = batch_net(63, 100, 64,32,16,2)#0.918 (relu)
    #初始化权重
    weight_init(model)
    if torch.cuda.is_available():
        model = model.cuda()
    criterion = nn.CrossEntropyLoss()
    # opitimizer = optim.SGD(model.parameters(), lr=learning_rate)
    opitimizer = optim.Adam(model.parameters(),lr=learning_rate, betas=(0.9, 0.999), eps=1e-08)#weight_decay l2正则化
    # optim.RMSprop
   
    #pytorch自带的自动调参工具 ，lr更新是每个batch更新，不是每个epoch更新
    #这个效果最好 acc=0.9872 初始lr=1e-2
    #给定一个metric，当metric停止优化时减小学习率。 factor（float）：lr减小的乘法因子，默认为0.1；patience（int）：在metric停止优化patience个epoch后减小lr
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(opitimizer, mode='min', factor=0.1, patience=2)
   
    # 按照论文里面的方法去寻找base/max lr 第一次出现拐点的两个值，这个方法用adam 貌似效果一般  #0.9831  #sgd 0.8974
    # 这里蓝色最好啊，推荐Nesterov Momentum+CLR？默默地改了自己的优化器
    # scheduler=CyclicLR(optimizer=opitimizer,base_lr=2.e-05,max_lr=0.0001,cycle_momentum=False)
    # 开始训练
    x=[]#记录学习率的变化
    y=[]
    
    num=0
    for i in range(num_epoches):
        num+=1
        for img, label in train_dataset:
            img = img.view(img.size(0), -1)
            # img = Variable(img)
            # print(img.size())
            # label = Variable(label)
            # forward
            out = model(img)
            loss = criterion(out, label)
            # backward
            opitimizer.zero_grad()
            loss.backward()
            # #梯度截断
            # nn.utils.clip_grad_norm_(model.parameters(), max_norm=20, norm_type=2)
            opitimizer.step()
            # # #CyclicLR 学习率的调整
            # scheduler.step()
            # x.append(num)
            # y.append(scheduler.get_lr())
        #为CLR 确定base/max learining rate
        # x.append(i)
        # y.append(loss.detach().numpy())
        # print(loss.detach().numpy())
        # opitimizer.param_groups[0]['lr']*=1.5
        # print(opitimizer.param_groups[0]['lr'])
        # #ReduceLROnPlateau学习率的调整
        scheduler.step(loss)
        # x.append(i)
        # y.append(opitimizer.param_groups[0]['lr'])
    # plt.plot(x,y)
    # plt.xlabel("iter")
    # plt.ylabel("learning rate")
    # plt.savefig("./pic/lr_CLR_pro")
    
    # plt.close()      
            # 打印
            # print("epoches= {},loss is {}".format(i, loss))
    # 测试
        model.eval()
        count = 0
        total=0
        for data in test_dataset:
            img, label = data
            img = img.view(img.size(0), -1)
            # img = Variable(img, volatile=True)
            #label = Variable(label, volatile=True)
            out = model(img)
            _, predict = torch.max(out, 1)
            total += label.size(0)
            count+= (predict == label).sum().item()
        print("acc = {}".format(count/total))
        x.append(num)
        y.append(count/total)
    
    plt.plot(x,y)
    plt.xlabel("ietr")
    plt.ylabel("test acc")
    plt.savefig("./pic2/test_acc.png")
    plt.close()
