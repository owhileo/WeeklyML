import torch
from torch import optim, nn
from torch.autograd import Variable
from torch.optim import optimizer
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import numpy as np
from torch.optim.lr_scheduler import CyclicLR
import matplotlib.pyplot as plt
# 加载数据集
def get_data():
    # 定义数据预处理操作, transforms.Compose将各种预处理操作组合在一起
    data_tf = transforms.Compose([transforms.ToTensor(), transforms.Normalize([0.5], [0.5])])
    train_dataset = datasets.MNIST(root='./data', train=True, transform=data_tf, download=True)
    test_dataset = datasets.MNIST(root='./data', train=False, transform=data_tf, download=True)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    #test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, drop_last=True)
    return train_loader, test_dataset

# 构建模型，三层神经网络
class batch_net(nn.Module):
    def __init__(self, in_dim, hidden1_dim, hidden2_dim, hidden3_dim,out_dim):
        super(batch_net, self).__init__()
        self.layer1 = nn.Sequential(nn.Linear(in_dim, hidden1_dim), nn.BatchNorm1d(hidden1_dim), nn.RReLU())
        self.layer2 = nn.Sequential(nn.Linear(hidden1_dim, hidden2_dim), nn.BatchNorm1d(hidden2_dim), nn.RReLU())
        self.layer3 = nn.Sequential(nn.Linear(hidden2_dim, hidden3_dim), nn.BatchNorm1d(hidden3_dim), nn.RReLU())
        self.layer4 = nn.Sequential(nn.Linear(hidden3_dim, out_dim))
    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        return x

#定义一个初始化权重的方法
def weight_init(net):
    for op in net.modules():
        if isinstance(op,nn.Linear):
            nn.init.kaiming_normal_(op.weight.data,nonlinearity='relu')
            nn.init.normal_(op.bias.data)

if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # 超参数配置
    batch_size = 64 
    learning_rate = 1e-2#1e-2
    num_epoches = 50
    # 加载数据集
    train_dataset, test_dataset = get_data()
    # 导入网络，并定义损失函数和优化器
    model = batch_net(28*28, 512, 512,128, 10)#acc = 0.9862
    model.to(device=device)
    #初始化权重
    weight_init(model)
    criterion = nn.CrossEntropyLoss()
    opitimizer = optim.Adam(model.parameters(),lr=0.001, betas=(0.9, 0.999), eps=1e-08,weight_decay=0.01)
   
    #pytorch自带的自动调参工具 ，lr更新是每个batch更新，不是每个epoch更新
    #这个效果最好 acc=0.9872 初始lr=1e-2
    #给定一个metric，当metric停止优化时减小学习率。 factor（float）：lr减小的乘法因子，默认为0.1；patience（int）：在metric停止优化patience个epoch后减小lr
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(opitimizer, mode='min', factor=0.1, patience=2)

    # 开始训练
    x=[]#记录学习率的变化
    y=[]
    
    num=0
    for i in range(num_epoches):
        num+=1
        model.train()
        for img, label in train_dataset:
            img = img.view(img.size(0), -1)
            img = img.to(device=device)
            label = label.to(device=device)
            out = model(img)
            loss = criterion(out, label)
            # backward
            opitimizer.zero_grad()
            loss.backward()
            opitimizer.step()
        scheduler.step(loss)
    
        model.eval()
        count = 0
        for data in test_dataset:
            img, label = data
            img = img.view(img.size(0), -1)
            img = img.to(device=device)
            # img = Variable(img, volatile=True)
            #label = Variable(label, volatile=True)
            out = model(img)
            _, predict = torch.max(out, 1)
            if predict == label:
                count += 1
        print("acc = {}".format(count/len(test_dataset)))