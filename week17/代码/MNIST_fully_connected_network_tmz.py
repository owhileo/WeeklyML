import os
import time

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torchvision.transforms as transforms
import torchvision.datasets as datasets


hidden_size2 = 1024
hidden_size3 = 1024
hidden_size4 = 512
class NeuralNet4(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(NeuralNet4, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, hidden_size2)
        self.fc3 = nn.Linear(hidden_size2, hidden_size3)
        self.fc4 = nn.Linear(hidden_size3, hidden_size4)
        self.fc5 = nn.Linear(hidden_size4, num_classes)
        self.dropout = nn.Dropout(p=0.2)
        self.bn1 = nn.BatchNorm1d(hidden_size)
        self.bn2 = nn.BatchNorm1d(hidden_size2)
        self.bn3 = nn.BatchNorm1d(hidden_size3)
        self.bn4 = nn.BatchNorm1d(hidden_size4)

    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.bn1(out)
        out = self.dropout(out)
        out = self.relu(self.fc2(out))
        out = self.bn2(out)
        out = self.dropout(out)
        out = self.fc3(out)
        out = self.relu(out)
        out = self.bn3(out)
        out = self.dropout(out)
        out = self.fc4(out)
        out = self.relu(out)
        out = self.bn4(out)
        out = self.dropout(out)
        out = self.fc5(out)
        return out

# Hyper-parameters
input_size = 784
hidden_size = 2048

num_classes = 10
num_epochs = 150
batch_size = 128
learning_rate = 0.001

print_freq = 100

def init_weights(m):
    if isinstance(m, nn.Linear):
        # torch.nn.init.xavier_uniform(m.weight)
        torch.nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        m.bias.data.fill_(0.01)



def main():

    model = NeuralNet4(input_size, hidden_size, num_classes).cuda()
    # model.apply(init_weights)
    cudnn.benchmark = True

    normalize = transforms.Normalize((0.1307,), (0.3081,))

    train_loader = torch.utils.data.DataLoader(
        datasets.MNIST(root='./data', train=True, transform=transforms.Compose([
            transforms.ToTensor(),
            # normalize,
        ]), download=True),
        batch_size=batch_size, shuffle=True,
        num_workers=4, pin_memory=True)
    # print(train_loader.dataset.data[0])
    val_loader = torch.utils.data.DataLoader(
        datasets.MNIST(root='./data', train=False, transform=transforms.Compose([
            transforms.ToTensor(),
            # normalize,
        ])),
        batch_size=batch_size, shuffle=False,
        num_workers=4, pin_memory=True)

    # define loss function (criterion) and optimizer
    criterion = nn.CrossEntropyLoss().cuda()

    # optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate,
    #                             momentum=0,
    #                             weight_decay=0)
    optimizer = torch.optim.Adam(model.parameters(), lr = learning_rate)

    l = [50, 90, 120]
    lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer,gamma=0.1,
                                                        milestones=l, last_epoch=-1)

    for epoch in range(0, num_epochs):

        # train for one epoch
        print('current lr {:.5e}'.format(optimizer.param_groups[0]['lr']))
        train(train_loader, model, criterion, optimizer, epoch)
        lr_scheduler.step()

        # evaluate on validation set
        prec1 = validate(val_loader, model, criterion)

        # # remember best prec@1 and save checkpoint
        # is_best = prec1 > best_prec1
        # best_prec1 = max(prec1, best_prec1)


def train(train_loader, model, criterion, optimizer, epoch):
    """
        Run one train epoch
    """
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()

    # switch to train mode
    model.train()

    end = time.time()
    for i, (input, target) in enumerate(train_loader):

        target = target.cuda()
        input_var = input.reshape(-1, 28*28).cuda()
        target_var = target

        # measure data loading time
        data_time.update(time.time() - end)
        # compute output
        output = model(input_var)

        loss = criterion(output, target_var)

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        output = output.float()
        loss = loss.float()
        # measure accuracy and record loss
        prec1 = accuracy(output.data, target)[0]
        losses.update(loss.item(), input.size(0))
        top1.update(prec1.item(), input.size(0))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if (i + 1) % print_freq == 0 or (i + 1) == len(train_loader):
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Prec@1 {top1.val:.3f} ({top1.avg:.3f})'.format(
                epoch, i + 1, len(train_loader), batch_time=batch_time,
                data_time=data_time, loss=losses, top1=top1))


def validate(val_loader, model, criterion):
    """
    Run evaluation
    """
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()

    # switch to evaluate mode
    model.eval()

    end = time.time()
    with torch.no_grad():
        for i, (input, target) in enumerate(val_loader):
            target = target.cuda()
            input_var = input.reshape(-1, 28*28).cuda()
            target_var = target.cuda()

            # compute output
            output = model(input_var)
            loss = criterion(output, target_var)

            output = output.float()
            loss = loss.float()

            # measure accuracy and record loss
            prec1 = accuracy(output.data, target)[0]
            losses.update(loss.item(), input.size(0))
            top1.update(prec1.item(), input.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if (i + 1) % print_freq == 0 or (i + 1) == len(val_loader):
                print('Test: [{0}/{1}]\t'
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                      'Prec@1 {top1.val:.3f} ({top1.avg:.3f})'.format(
                    i + 1, len(val_loader), batch_time=batch_time, loss=losses,
                    top1=top1))

    print(' * Prec@1 {top1.avg:.3f}'
          .format(top1=top1))

    return top1.avg


def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    torch.save(state, filename)


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res


if __name__ == '__main__':
    main()
