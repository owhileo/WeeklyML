# !/usr/bin/env python3
# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable
from torchvision import transforms
import torchvision.datasets as dst
from torchvision.utils import save_image


EPOCH = 15
BATCH_SIZE = 64
n = 2   # num_workers
LATENT_CODE_NUM = 32
log_interval = 10




class VAE(nn.Module):
    def __init__(self):
        super(VAE, self).__init__()

        self.encoder = nn.Sequential(   # x: 1,28,28
            nn.Conv2d(1, 64, kernel_size=4, stride=2, padding=1),   # 64,14,14
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1), # 128,7,7
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1), # 128,7,7
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
        )

        self.fc11 = nn.Linear(128 * 7 * 7, LATENT_CODE_NUM)
        self.fc12 = nn.Linear(128 * 7 * 7, LATENT_CODE_NUM)
        self.fc2 = nn.Linear(LATENT_CODE_NUM, 128 * 7 * 7)

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(64, 1, kernel_size=4, stride=2, padding=1),
            nn.Sigmoid()
        )

    def reparameterize(self, mu, logvar):
        eps = Variable(torch.randn(mu.size(0), mu.size(1))).cuda()
        z = mu + eps * torch.exp(logvar/2)

        return z

    def forward(self, x):
        # x: 1,28,28
        out1, out2 = self.encoder(x), self.encoder(x)  # batch_s, 128, 7, 7
        mu = self.fc11(out1.view(out1.size(0), -1))     # batch_s, latent
        logvar = self.fc12(out2.view(out2.size(0), -1))  # batch_s, latent
        z = self.reparameterize(mu, logvar)      # batch_s, latent
        out3 = self.fc2(z).view(z.size(0), 128, 7, 7)    # batch_s, 8, 7, 7

        return self.decoder(out3), mu, logvar


def loss_func(recon_x, x, mu, logvar):
    BCE = F.binary_cross_entropy(recon_x, x,  size_average=False)
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

    return BCE+KLD

if __name__=='__main__':
    def train(epoch):
        vae.train()
        total_loss = 0
        for i, (data, _) in enumerate(train_loader, 0):
            data = Variable(data).cuda()
            optimizer.zero_grad()
            recon_x, mu, logvar = vae.forward(data)
            loss = loss_func(recon_x, data, mu, logvar)
            loss.backward()
            total_loss += loss.item()
            optimizer.step()

            if i % log_interval == 0:
                # sample = Variable(torch.randn(64, LATENT_CODE_NUM)).cuda()
                # sample = vae.decoder(vae.fc2(sample).view(64, 128, 7, 7)).cpu()
                # save_image(sample.data.view(64, 1, 28, 28),
                #         'result/sample_' + str(epoch) + '.png')
                print('Train Epoch:{} -- [{}/{} ({:.0f}%)] -- Loss:{:.6f}'.format(
                    epoch, i*len(data), len(train_loader.dataset),
                    100.*i/len(train_loader), loss.item()/len(data)))

        print('====> Epoch: {} Average loss: {:.4f}'.format(
            epoch, total_loss / len(train_loader.dataset)))
        vae.eval()
        for data in tdata:
            data = Variable(data).cuda()
            optimizer.zero_grad()
            recon_x, mu, logvar = vae.forward(data)
            loss = loss_func(recon_x, data, mu, logvar)
            total_loss += loss.item()
            save_image(recon_x.cpu().data.view(64, 1, 28, 28),
                    'result/output_' + str(epoch) + '.png')
            sample = Variable(torch.randn(64, LATENT_CODE_NUM)).cuda()
            sample = vae.decoder(vae.fc2(sample).view(64, 128, 7, 7)).cpu()
            save_image(sample.data.view(64, 1, 28, 28),
                    'result/sample_' + str(epoch) + '.png')
                # print('Train Epoch:{} -- [{}/{} ({:.0f}%)] -- Loss:{:.6f}'.format(
                #     epoch, i*len(data), len(train_loader.dataset),
                #     100.*i/len(test_loader), loss.item()/len(data)))


    transform = transforms.Compose([transforms.ToTensor()])
    data_train = dst.MNIST('MNIST_data/', train=True,
                        transform=transform, download=False)
    data_test = dst.MNIST('MNIST_data/', train=False, transform=transform)
    train_loader = torch.utils.data.DataLoader(
        dataset=data_train, num_workers=n, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = torch.utils.data.DataLoader(
        dataset=data_test, num_workers=n, batch_size=BATCH_SIZE, shuffle=False)

    print(len(data_train))
    print(len(data_test))

    vae = VAE().cuda()
    optimizer = optim.Adam(vae.parameters(), lr=0.001, betas=(0.9, 0.999), eps=1e-08, weight_decay=0)
    tdata=0
    for i, (data, _) in enumerate(test_loader, 0):
        tdata=[data]
        save_image(data.data.view(64, 1, 28, 28),
                    'result/sample' + '.png')
        break
    for epoch in range(1, EPOCH):
        train(epoch)
    
    torch.save(vae,'vae.m')


