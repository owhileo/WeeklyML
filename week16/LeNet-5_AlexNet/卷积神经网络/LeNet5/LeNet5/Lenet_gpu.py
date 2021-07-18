import torch
import torch.nn as nn
import torch.utils.data as Data
import torchvision
import time

# torch.set_printoptions(profile="full")
 
USE_GPU = False
 
EPOCH = 10
BATCH_SIZE = 50
LR = 0.01
DOWNLOAD_MNIST = True
 
train_data = torchvision.datasets.MNIST(root='./mnist/', train=True, transform=torchvision.transforms.ToTensor(),
                                        download=DOWNLOAD_MNIST, )
train_loader = Data.DataLoader(dataset=train_data, batch_size=BATCH_SIZE, shuffle=True)
test_data = torchvision.datasets.MNIST(root='./mnist/', train=False)
 
if USE_GPU:
    test_x = torch.unsqueeze(test_data.test_data, dim=1).type(torch.FloatTensor)[:2000].cuda()/255.
    test_y = test_data.test_labels[:2000].cuda()
else:
    test_x = torch.unsqueeze(test_data.test_data, dim=1).type(torch.FloatTensor)[:2000] / 255.
    # print("hhhhhhhhhhhhhhhhhh")
    # print(torch.unsqueeze(test_data.test_data, dim=1).type(torch.FloatTensor)[:2000]/255)
    test_y = test_data.test_labels[:2000]
 
 
# 搭建网络
class _LeNet(nn.Module):
    def __init__(self):
        super(_LeNet, self).__init__()  # 输入是28*28*1
 
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=16, kernel_size=5, stride=1, padding=2),  # 28*28*16
            nn.MaxPool2d(kernel_size=2),  # 14*14*16
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=5, stride=1, padding=2),  # 14*14*32
            nn.MaxPool2d(kernel_size=2),  # 7*7*32
        )
        self.linear1 = nn.Linear(7 * 7 * 32, 120)
        self.linear2 = nn.Linear(120, 84)
        self.out = nn.Linear(84, 10)
 
    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = x.view(x.size(0), -1)
        x = self.linear1(x)
        x = self.linear2(x)
        output = self.out(x)
        return output
 
 
cnn = _LeNet()
if USE_GPU:
    cnn.cuda()
print(cnn)
optimizer = torch.optim.Adam(cnn.parameters(), lr=LR)
loss_func = nn.CrossEntropyLoss()
time_start=time.time()
for epoch in range(EPOCH):
    for step, (x, y) in enumerate(train_loader):
        if USE_GPU:
            c_x=x.cuda()
            c_y=y.cuda()
        else:    
            c_x = x
            c_y = y
        output = cnn(c_x)
        print(output)
        print("hh")
        print(c_y)
        loss = loss_func(output, c_y)
        print(loss)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        #########训练到此结束##########
 
        if True:
            test_out = cnn(test_x)
            if USE_GPU:
                pred_y = torch.max(test_out, 1)[1].cuda().data
            else:    
                pred_y = torch.max(test_out, 1)[1].data
            accuracy = torch.sum(pred_y == test_y).type(torch.FloatTensor) / test_y.size(0)
            if USE_GPU:
                print('Epoch: ', epoch, '| train loss: %.4f' % loss.data.cpu().numpy(), '| test accuracy: %.2f' % accuracy)
            else:    
                print('Epoch: ', epoch, '| train loss: %.4f' % loss.data.numpy(), '| test accuracy: %.2f' % accuracy)
time_end=time.time()
 
test_output = cnn(test_x[:10])
if USE_GPU:
    pred_y = torch.max(test_output, 1)[1].cuda().data
else:    
    pred_y = torch.max(test_output, 1)[1].data
print(pred_y, 'prediction numbe')
print(test_y[:10], 'real number')
print('time cost',time_end-time_start,'s')
