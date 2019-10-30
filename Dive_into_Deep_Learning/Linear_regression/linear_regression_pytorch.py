#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
  @ Time     : 2019/9/29 下午3:45
  @ Author   : Vodka
  @ File     : linear_regression_pytorch .py
  @ Software : PyCharm
"""
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import torch.utils.data as Data

num_inputs = 2
num_examples = 1000
learning_rate = 0.03
batch_size = 10
num_epochs = 20


# 定义模型
class LinearNet(nn.Module):
    def __init__(self, n_feature):
        super(LinearNet, self).__init__()
        self.linear = nn.Linear(n_feature, 1)

    def forward(self, X):  # forward 定义前向传播
        y = self.linear(X)
        return y

    # 查看参数
    # net.linear.weight.detach().numpy()
    # net.linear.bias


if __name__ == '__main__':
    # 生成训练数据 y = 2x1-3.4x2+4.2
    true_w = [2, -3.4]
    true_b = 4.2
    features = torch.tensor(np.random.normal(0, 1, (num_examples, num_inputs)), dtype=torch.float)  # 1000×2
    labels = true_w[0] * features[:, 0] + true_w[1] * features[:, 1] + true_b
    labels += torch.tensor(np.random.normal(0, 0.01, size=labels.size()), dtype=torch.float)  # 1000×1
    dataset = Data.TensorDataset(features, labels)
    data = Data.DataLoader(dataset, batch_size, shuffle=True, num_workers=2)

    net = LinearNet(num_inputs)
    # 也可以用nn.Sequential来实现模型层的添加
    # net = nn.Sequential()
    # net.add_module('linear', nn.Linear(num_inputs, 1))
    # for param in net.parameters():
    #     print(param)
    # print(net)  # 使⽤print可以打印出⽹络的结构

    # 初始化模型
    loss = nn.MSELoss()
    optimizer = torch.optim.SGD(net.parameters(), lr=0.03)

    # 训练模型
    loss_dict = []
    for epoch in range(num_epochs):
        for X, y in data:
            output = net(X)
            l = loss(output, y.view(-1, 1))
            loss_dict.append(l.item())
            l.backward()
            optimizer.step()
            optimizer.zero_grad()  # 梯度清零，等价于net.zero_grad()
        if (epoch + 1) % 5 == 0:
            print('Epoch [{}/{}], Loss: {:.4f}'.format(epoch + 1, num_epochs, l.item()))
    plt.plot(loss_dict, label='loss')
    plt.legend()
    plt.show()
    print('真实函数y=%.2fx1%.2fx2+%.2f' % (true_w[0], true_w[1], true_b))
    print('拟合函数y=%.2fx1%.2fx2+%.2f' % (
        net.linear.weight.detach().numpy()[0][0], net.linear.weight.detach().numpy()[0][1], net.linear.bias))
