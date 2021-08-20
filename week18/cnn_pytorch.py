import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torchvision
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import numpy as np
import time
import torch.nn.functional as F

# 设定超参数及常数
learning_rate = 0.0005       #学习率
batch_size = 32            #批处理量
epochs_num = 10            #训练迭代次数
DROPOUT_VALUE = 0          #dropout率
dataset = 0                #数据集载入 0：mnist   1：intelimg

download = False           #数据集加载方式
use_gpu = 1                #CUDA GPU加速  1:使用  0:禁用
is_train = 1               #训练模型  1:重新训练     0:加载现有模型

if dataset == 0:
# 载入MNIST训练集#####################################################################
    train_dataset = datasets.MNIST(root='.',                      # 数据集目录
                                   train=True,                    # 训练集标记
                                   transform=transforms.ToTensor(),  # 转为Tensor变量
                                   download=download)

    train_loader = DataLoader(dataset=train_dataset,  # 数据集加载
                              shuffle=True,           # 随机打乱数据
                              batch_size=batch_size)  # 批处理量
# MNIST测试集准确率测试
    test_dataset = datasets.MNIST(root='.',
                                  train=False,
                                  transform=transforms.ToTensor(),
                                  download=download)

    test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                              shuffle=True,
                                              batch_size=batch_size)
# 载入intelimg数据集########################################################################################
else:
    transformtrain = transforms.Compose([
        transforms.Resize((150, 150)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((.5, .5, .5), (.5, .5, .5))
    ])
    transformtest = transforms.Compose([
        transforms.Resize((150, 150)),
        transforms.ToTensor(),
        transforms.Normalize((.5, .5, .5), (.5, .5, .5))
    ])
    trainds = datasets.ImageFolder('./archive/seg_train/seg_train', transform=transformtrain)
    testds = datasets.ImageFolder('./archive/seg_test/seg_test', transform=transformtest)
    train_loader = DataLoader(trainds, batch_size=256, shuffle=True)
    test_loader = DataLoader(testds, batch_size=64, shuffle=False)


class Lenet(nn.Module):
    def __init__(self):
        super(Lenet,self).__init__()
        layer1 = nn.Sequential()
        layer1.add_module('conv1',nn.Conv2d(1,6,3,padding=1))
        layer1.add_module('pool1',nn.MaxPool2d(2,2))
        self.layer1 = layer1
        
        layer2 = nn.Sequential()
        layer2.add_module('conv2',nn.Conv2d(6,16,3))
        layer2.add_module('conv3',nn.Conv2d(16,16,3))
        layer2.add_module('pool2',nn.MaxPool2d(2,2))
        self.layer2 = layer2
        
        layer3 = nn.Sequential()
        layer3.add_module('fc1',nn.Linear(400,120))
        layer3.add_module('fc2',nn.Linear(120,84))
        layer3.add_module('fc3',nn.Linear(84,10))
        
        self.layer3 = layer3
        
    def forward(self,x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = x.view(x.size(0),-1)
        x = self.layer3(x)
        
        return x

class LRN(nn.Module):
    def __init__(self, local_size=1, alpha=1.0, beta=0.75, ACROSS_CHANNELS=False):
        super(LRN, self).__init__()
        self.ACROSS_CHANNELS = ACROSS_CHANNELS
        if self.ACROSS_CHANNELS:
            self.average=nn.AvgPool3d(kernel_size=(local_size, 1, 1), #0.2.0_4会报错，需要在最新的分支上AvgPool3d才有padding参数
                    stride=1,
                    padding=(int((local_size-1.0)/2), 0, 0)) 
        else:
            self.average=nn.AvgPool2d(kernel_size=local_size,
                    stride=1,
                    padding=int((local_size-1.0)/2))
        self.alpha = alpha
        self.beta = beta
    
    
    def forward(self, x):
        if self.ACROSS_CHANNELS:
            div = x.pow(2).unsqueeze(1)
            div = self.average(div).squeeze(1)
            div = div.mul(self.alpha).add(1.0).pow(self.beta)#这里的1.0即为bias
        else:
            div = x.pow(2)
            div = self.average(div)
            div = div.mul(self.alpha).add(1.0).pow(self.beta)
        x = x.div(div)
        return x

class AlexNet(nn.Module):
    def __init__(self, num_classes = 10):#imagenet数量
        super().__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=96,kernel_size=3, stride=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            LRN(local_size=5, alpha=1e-4, beta=0.75, ACROSS_CHANNELS=True)
        )
        
        self.layer2 = nn.Sequential(
            nn.Conv2d(in_channels=96, out_channels=96, kernel_size=3, groups=2, padding=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=96, out_channels=256, kernel_size=3, groups=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            LRN(local_size=5, alpha=1e-4, beta=0.75, ACROSS_CHANNELS=True)
        )
        
        self.layer3 = nn.Sequential(
            nn.Conv2d(in_channels=256, out_channels=384, padding=1, kernel_size=3),
            nn.ReLU(inplace=True)
        )
        self.layer4 = nn.Sequential(
            nn.Conv2d(in_channels=384, out_channels=384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Dropout()
        )
        
        self.layer5 = nn.Sequential(
            nn.Conv2d(in_channels=384, out_channels=256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.MaxPool2d(kernel_size=3, stride=2)
        )
        
         #需要针对上一层改变view
        self.layer6 = nn.Sequential(
            nn.Linear(in_features=2*2*256, out_features=4096),
            nn.ReLU(inplace=True),
            #nn.Dropout()
        )
        self.layer7 = nn.Sequential(
            nn.Linear(in_features=4096, out_features=4096),
            nn.ReLU(inplace=True),
            #nn.Dropout()
        )
        
        self.layer8 = nn.Linear(in_features=4096, out_features=num_classes)
        
    def forward(self, x):
        x = self.layer5(self.layer4(self.layer3(self.layer2(self.layer1(x)))))
        x = x.view(x.size()[0], 256*2*2)
        x = self.layer8(self.layer7(self.layer6(x)))
        return x
    
########图片大小不断变小（卷积、池化），卷积和池化深度有限制################
class VGG(nn.Module):
    def __init__(self,num_classes = 10):
        super(VGG,self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1,64,kernel_size=3,padding=197),
            nn.ReLU(True),
            
            nn.Conv2d(64,64,kernel_size=3,padding=1),
            nn.ReLU(True),
            nn.MaxPool2d(kernel_size=2,stride=2),
            
            nn.Conv2d(64,128,kernel_size=3,padding=1),
            nn.ReLU(True),

            nn.Conv2d(128,128,kernel_size=3,padding=1),
            nn.ReLU(True),
            nn.MaxPool2d(kernel_size=2,stride=2),
            
            nn.Conv2d(128,256,kernel_size=3,padding=1),
            nn.ReLU(True),
            
            nn.Conv2d(256,256,kernel_size=3,padding=1),
            nn.ReLU(True),

            nn.Conv2d(256,256,kernel_size=3,padding=1),
            nn.ReLU(True),
            nn.MaxPool2d(kernel_size=2,stride=2),
            
            nn.Conv2d(256,512,kernel_size=3,padding=1),
            nn.ReLU(True),
            
            nn.Conv2d(512,512,kernel_size=3,padding=1),
            nn.ReLU(True),

            nn.Conv2d(512,512,kernel_size=3,padding=1),
            nn.ReLU(True),
            nn.MaxPool2d(kernel_size=2,stride=2),
            
            nn.Conv2d(512,512,kernel_size=3,padding=1),
            nn.ReLU(True),
            
            nn.Conv2d(512,512,kernel_size=3,padding=1),
            nn.ReLU(True),

            nn.Conv2d(512,512,kernel_size=3,padding=1),
            nn.ReLU(True),
            nn.MaxPool2d(kernel_size=2,stride=2))
        
        self.classifier = nn.Sequential(
            nn.Linear(input_nodes,4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096,4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096,num_classes))
        
    def forward(self,x):
        x = self.features(x)
        print(x.size())
        x = x.view(x.size(0),-1)
        x = self.classifier(x)
        return x

# 初始化卷积神经网络
class Network(nn.Module):
    def __init__(self):
        super(Network, self).__init__()

        # Input Block
        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=3, padding=0), 
            nn.ReLU(),
            #nn.Dropout(DROPOUT_VALUE),
            #nn.MaxPool2d(2,stride=2)
            # nn.BatchNorm2d(out_channels)
        )
        
        # CONVOLUTION BLOCK 1
        self.conv2 = nn.Sequential(
            nn.Conv2d(64, 256, kernel_size=3, padding=0),
            nn.ReLU(),
            #nn.Dropout(DROPOUT_VALUE),
            #nn.MaxPool2d(2,stride=2) 
            # nn.BatchNorm2d(out_channels)
        )

        self.conv3 = nn.Sequential(
            nn.Conv2d(256, 512, kernel_size=3, padding=0),
            nn.ReLU(),
            #nn.Dropout(DROPOUT_VALUE),
            nn.MaxPool2d(2,stride=2) 
            # nn.BatchNorm2d(out_channels)
        )

        self.conv4 = nn.Sequential(
            nn.Conv2d(512, 128, kernel_size=3, padding=0),
            nn.ReLU(),
            #nn.Dropout(DROPOUT_VALUE),
            nn.MaxPool2d(2,stride=2) 
            # nn.BatchNorm2d(out_channels)
        )
                
        # FC BLOCK 1
        self.fc1 = nn.Sequential(
            nn.Linear(147456,1024),                       # 全连接层
            nn.ReLU()                                     # 激活函数ReLU
        )

        # output BLOCK 2
        self.output = nn.Sequential(
            nn.Linear(1024,10),                       # 全连接层
#使用了交叉熵损失，就无需在输出结果用一个softmax，否则loss就一直在一个范围内波动。
            # nn.Softmax(dim=1)
        )

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        
        x = x.view(x.size()[0], -1)
        #print(x.size())
        x = self.fc1(x)
        
        x = self.output(x)
        return x


# 初始化神经网络
net = Network()
if use_gpu:           #CUDA GPU加速
    net = net.cuda()

start = time.time()
if is_train:
    criterion = nn.CrossEntropyLoss()  # 交叉熵损失函数
    # optimizer = torch.optim.SGD(net.parameters(), lr=learning_rate, momentum=0.8)  # 使用SGD算法进行训练
    optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate)  # 使用adam算法进行训练
    counter = []
    loss_history = []
    correct_history = []
    correct_cnt = 0
    counter_temp = 0
    record_interval = 100
    # 多次迭代训练网络
    for epoch in range(0, epochs_num):
        for i, data in enumerate(train_loader, 0):
            img, label = data
            if use_gpu:   #CUDA GPU加速
                img, label = img.cuda(), label.cuda()

            optimizer.zero_grad()            # 清除网络状态
            output = net(img)                # 前向传播
            loss = criterion(output, label)  # 计算损失函数
            loss.backward()                  # 反向传播
            optimizer.step()                 # 参数更新

            _, predict = torch.max(output, 1)
            correct_cnt += (predict == label).sum()  # 预测值与实际值比较

            # 存储损失值与精度
            if i%record_interval == record_interval-1:
                counter_temp += record_interval * batch_size
                counter.append(counter_temp)
                loss_history.append(loss.item())
                correct_history.append(correct_cnt.float().item()/(record_interval*batch_size))
                correct_cnt = 0
        print("迭代次数 {}\n 当前损失函数值 {}\n".format(epoch, loss.item()))

    # 存储模型参数
    state = {'net':net.state_dict()}
    torch.save(net.state_dict(),'.\modelpara.pth')
end = time.time()

# 加载模型参数
if use_gpu:
    net.load_state_dict(torch.load('.\modelpara.pth'))
else:
    net.load_state_dict(torch.load('.\modelpara.pth', map_location='cpu'))


# 训练集预测测试

correct = 0
for i,data in enumerate(test_loader, 0):
    img,label = data
    if use_gpu:             #CUDA GPU加速
        img, label = img.cuda(), label.cuda()
    output = net(img)       # 前向传播
    _,predict = torch.max(output,1)
    correct += (predict==label).sum()  # 预测值与实际值比较


# 输出测试准确率及时间
print('MNIST测试集识别准确率= {:.2f}'.format(correct.cpu().numpy()/len(test_dataset)*100)+'%')
print ('时间= {:.3f}'.format(end-start)+' s')
