from operator import mod
import pandas as pd
from sklearn import utils
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, accuracy_score
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import torch
from torch._C import dtype
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, TensorDataset
import torch.nn.functional as F
import torchvision.datasets as dst
from torchvision import transforms

from sklearn.metrics import balanced_accuracy_score,accuracy_score

import matplotlib.pyplot as plt


def load_bank_data(BATCH_SIZE=32):
    data = pd.read_csv('bank-additional/bank-additional-full.csv', sep=';')
    data_x = data.iloc[:, 1:-1]
    data_y = data.iloc[:, -1]
    data_x = pd.get_dummies(data_x)
    data_y = pd.get_dummies(data_y)
    # print(data_y.groupby(data_y.columns[0]).count())
    data_y = data_y.iloc[:, 0]

    scaler = MinMaxScaler()

    X_train, X_test, y_train, y_test = train_test_split(
        data_x, data_y, test_size=0.2, random_state=42)
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    deal_dataset = TensorDataset(torch.tensor(
        X_train, dtype=torch.float32), torch.tensor(y_train.values, dtype=torch.long))
    test_dataset = TensorDataset(torch.tensor(
        X_test, dtype=torch.float32), torch.tensor(y_test.values, dtype=torch.long))
    train_loader = DataLoader(deal_dataset, BATCH_SIZE, True)
    test_loader = DataLoader(test_dataset, BATCH_SIZE, False)
    return train_loader, test_loader

def load_MNIST_data(BATCH_SIZE):
    transform = transforms.Compose([transforms.ToTensor()])
    data_train = dst.MNIST('MNIST_data/', train=True,
                           transform=transform, download=False)
    data_test = dst.MNIST('MNIST_data/', train=False, transform=transform)
    train_loader = DataLoader(
        dataset=data_train,  batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(
        dataset=data_test, batch_size=BATCH_SIZE, shuffle=False)
    return train_loader, test_loader

def compute_kl_loss(p, q, pad_mask=None):
    
    p_loss = F.kl_div(F.log_softmax(p, dim=-1), F.softmax(q, dim=-1), reduction='none')
    q_loss = F.kl_div(F.log_softmax(q, dim=-1), F.softmax(p, dim=-1), reduction='none')
    
    # pad_mask is for seq-level tasks
    if pad_mask is not None:
        p_loss.masked_fill_(pad_mask, 0.)
        q_loss.masked_fill_(pad_mask, 0.)

    # You can choose whether to use function "sum" and "mean" depending on your task
    p_loss = p_loss.sum()
    q_loss = q_loss.sum()

    loss = (p_loss + q_loss) / 2
    return loss

class NNModel(nn.Module):
    def __init__(self,output=2):
        super(NNModel, self).__init__()
        self.model = nn.Sequential(
            nn.BatchNorm1d(62,momentum=0.3),
            nn.Linear(62, 256),
            nn.Dropout(0.1),
            nn.LeakyReLU(),
            nn.Linear(256, 512),
            nn.Dropout(0.1),
            nn.LeakyReLU(),
            nn.Linear(512, 128),
            nn.Dropout(0.1),
            nn.LeakyReLU(),
            nn.Linear(128, output),
        )

    def forward(self, x):
        x = self.model(x)
        return x

class NNModel2(nn.Module):
    def __init__(self,output=10):
        super(NNModel2, self).__init__()
        self.model = nn.Sequential(
            nn.BatchNorm1d(784,momentum=0.3),
            nn.Linear(784, 128),
            nn.Dropout(0.4),
            nn.ReLU(),
            nn.Linear(128, 256),
            nn.Dropout(0.4),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.Dropout(0.4),
            nn.ReLU(),
            nn.Linear(128, output),
        )

    def forward(self, x):
        x=x.view([-1,784])
        x = self.model(x)
        return x

if __name__=='__main__':
    BANK=True
    if BANK:
        BATCH_SIZE=128
        EPOCH = 80
        alpha = 0.1
        R_DROPOUT=False
        lr=1e-5
        # lr=1e-4
        model_name='model1.m'
    
        train_loader, test_loader=load_bank_data(BATCH_SIZE) 
        model = NNModel()
        # model = torch.load('model1.m')
        optimizer = optim.Adam(model.parameters(), lr=lr,
        betas=(0.9,0.999),weight_decay=0)
        scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.9999)
        weight=torch.Tensor([1.,1.]).cuda()
        loss = nn.CrossEntropyLoss(weight=weight)
    else:
        BATCH_SIZE=128
        EPOCH = 80
        alpha = 0.01
        R_DROPOUT=False
        lr=1e-4
        model_name='model2.m'
        train_loader, test_loader=load_MNIST_data(BATCH_SIZE) 
        model = NNModel2()
        optimizer = optim.Adam(model.parameters(), lr=lr)
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10,gamma=0.9)
        loss = nn.CrossEntropyLoss()
    # model = torch.load('model1.m')

    
    # optimizer = optim.Adadelta(model.parameters(), lr=lr)
    # optimizer = optim.ASGD(model.parameters(), lr=lr)

    model.cuda()
    history_train = []
    history_test = []
    history_acc_test = []
    history_acc_train = []
    for epoch in range(EPOCH):
        loss_ = 0
        acc_ = []
        model.train()
        for i, data in enumerate(train_loader):
            input, label = data
            input = input.cuda()
            label = label.cuda()
            optimizer.zero_grad()
            output = model(input)

            if R_DROPOUT:
                output2 = model(input)
                ce_loss = 0.5 * (loss(output, label) + loss(output2, label))
                kl_loss = compute_kl_loss(output, output2)
                l = ce_loss + alpha * kl_loss
            else:
                l = loss(output, label)
            l.backward()
            optimizer.step()
            loss_ += l.item()
            _, indices = torch.max(output, dim=1)
            acc=balanced_accuracy_score(label.cpu(),indices.cpu())
            # correct = torch.sum(indices == label)
            # acc = correct.item()/len(label)
            acc_.append(acc)
        scheduler.step()
        history_train.append(loss_)
        history_acc_train.append(np.mean(acc_))
        print("training loss for %d epoch: %.2lf" % (epoch, loss_))
        loss_ = 0
        model.eval()
        acc_ = []
        acc_2 = []
        acc_3 = []
        for i, data in enumerate(test_loader):
            input, label = data
            input = input.cuda()
            label = label.cuda()
            output = model(input)

            if R_DROPOUT:
                output2 = model(input)
                ce_loss = 0.5 * (loss(output, label) + loss(output2, label))
                kl_loss = compute_kl_loss(output, output2)
                l = ce_loss + alpha * kl_loss
            else:
                l = loss(output, label)
            loss_ += l.item()

            _, indices = torch.max(output, dim=1)
            acc=balanced_accuracy_score(label.cpu(),indices.cpu())
            acc2=accuracy_score(label.cpu(),indices.cpu())
            correct = torch.sum(indices == label)
            acc3 = correct.item()/len(label)
            acc_.append(acc)
            acc_2.append(acc2)
            acc_3.append(acc3)
        history_test.append(loss_)
        history_acc_test.append(np.mean(acc_))
        print("testing  bal. acc for %d epoch: %.2lf%%" % (epoch, 100*history_acc_test[-1]))
        print("testing  acc for %d epoch: %.2lf%%" % (epoch, 100*np.mean(acc_2)))
        print("testing  acc2 for %d epoch: %.2lf%%" % (epoch, 100*np.mean(acc_3)))
        print("testing  loss for %d epoch: %.2lf" % (epoch, loss_))

    torch.save(model,model_name)
    print(min(history_test))
    print(max(history_acc_test)*100)
    plt.subplot(121)
    plt.plot(range(EPOCH), history_train, label='train loss')
    plt.plot(range(EPOCH), history_test, label='test loss')
    # plt.title('Without DropOut')
    plt.legend()
    plt.subplot(122)
    plt.plot(range(EPOCH), history_acc_train, label='train acc')
    plt.plot(range(EPOCH), history_acc_test, label='test acc')
    plt.legend()
    plt.savefig('res.png')
    # plt.show()
