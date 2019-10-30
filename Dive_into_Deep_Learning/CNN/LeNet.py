#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
  @ Time     : 2019/10/30 下午1:08
  @ Author   : Vodka
  @ File     : LeNet .py
  @ Software : PyCharm
"""
import sys
import time

import torch
import torch.nn as nn
import torchvision

batch_size = 256
lr = 0.001
num_epochs = 5
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# 和加载数据集有关的函数
def load_data_fashion_mnist(batch_size, resize=None, root='~/Desktop'):
    """Download the fashion mnist dataset and then load into memory."""
    # 将28*28的矩阵转换为一维的784的矩阵
    #
    trans = []
    if resize:
        trans.append(torchvision.transforms.Resize(size=resize))
    trans.append(torchvision.transforms.ToTensor())

    transform = torchvision.transforms.Compose(trans)
    mnist_train = torchvision.datasets.FashionMNIST(root=root, train=True, download=True, transform=transform)
    mnist_test = torchvision.datasets.FashionMNIST(root=root, train=False, download=True, transform=transform)
    if sys.platform.startswith('win'):
        num_workers = 0  # 0表示不用额外的进程来加速读取数据
    else:
        num_workers = 4
    train_iter = torch.utils.data.DataLoader(mnist_train, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    test_iter = torch.utils.data.DataLoader(mnist_test, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    # 分为235次读取，每次读256个数据
    return train_iter, test_iter


def evaluate_accuracy(data_iter, net):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    acc_sum, n = 0.0, 0
    with torch.no_grad():
        for X, y in data_iter:
            if isinstance(net, torch.nn.Module):
                net.eval()  # 评估模式, 这会关闭dropout
                acc_sum += (net(X.to(device)).argmax(dim=1) == y.to(device)).float().sum().cpu().item()
                net.train()  # 改回训练模式
            else:  # 自定义的模型, 3.13节之后不会用到, 不考虑GPU
                if ('is_training' in net.__code__.co_varnames):  # 如果有is_training这个参数
                    # 将is_training设置成False
                    acc_sum += (net(X, is_training=False).argmax(dim=1) == y).float().sum().item()
                else:
                    acc_sum += (net(X).argmax(dim=1) == y).float().sum().item()
            n += y.shape[0]
    return acc_sum / n


class LeNet(nn.Module):
    def __init__(self):
        super(LeNet, self).__init__()
        """
            (28*28)--Conv-->(24*24)--MaxPool2d-->(12*12)--Conv-->(8*8)--MaxPool2d-->(4*4)
        """
        self.conv = nn.Sequential(
            nn.Conv2d(1, 6, 5),  # in_channels, out_channels, kernel_size
            nn.Sigmoid(),
            nn.MaxPool2d(2, 2),  # kernel_size, stride
            nn.Conv2d(6, 16, 5),
            nn.Sigmoid(),
            nn.MaxPool2d(2, 2)
        )

        self.fc = nn.Sequential(
            nn.Linear(16 * 4 * 4, 120),
            nn.Sigmoid(),
            nn.Linear(120, 84),
            nn.Sigmoid(),
            nn.Linear(84, 10)
        )

    def forward(self, img):
        features = self.conv(img)
        # img.shape[0] 为 batch_size
        output = self.fc(features.view(img.shape[0], -1))
        return output


def train(net, train_iter, test_iter, batch_size, optimizer, device, num_epochs):
    net = net.to(device)
    loss = torch.nn.CrossEntropyLoss()
    cnt_batch = 0
    for epoch in range(num_epochs):
        train_loss_sum = 0.0  # 每次迭代的总损失
        train_acc_sum = 0.0  # 每次迭代的精度
        n = 0  # 每次迭代的样例个数
        for _index, (X, y) in enumerate(train_iter):
            X = X.to(device)
            y = y.to(device)  # y是标量
            predict_y = net(X)
            _loss = loss(predict_y, y)
            optimizer.zero_grad()
            _loss.backward()
            optimizer.step()
            train_loss_sum += _loss.cpu().item()
            train_acc_sum + (predict_y.argmax(dim=1) == y).sum().cpu().item()
            n += y.shape[0]
            cnt_batch += 1
            print('epoch: %d , batch: %d , train loss: %.3f' % (epoch + 1, _index, train_loss_sum / n))
        test_acc = evaluate_accuracy(test_iter, net)
        print('epoch %d, test acc %.3f'
              % (epoch + 1, test_acc))


net = LeNet()
train_iter, test_iter = load_data_fashion_mnist(batch_size=batch_size)
optimizer = torch.optim.Adam(net.parameters(), lr=lr)
train(net, train_iter, test_iter, batch_size, optimizer, device, num_epochs)
