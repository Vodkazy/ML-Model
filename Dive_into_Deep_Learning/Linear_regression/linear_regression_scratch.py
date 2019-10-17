#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
  @ Time     : 2019/9/29 下午12:09
  @ Author   : Vodka
  @ File     : linear_regression_scratch .py
  @ Software : PyCharm
"""
import random
import torch
from matplotlib import pyplot as plt
import numpy as np


# 线性回归就是y=Xw+b

def show_train_data(features, labels):
    plt.scatter(features[:, 1].numpy(), labels.numpy())
    plt.show()


def read_data(batch_size, features, labels):
    # 随机小样本批处理
    _indexs = list(range(features.shape[0]))
    random.shuffle(_indexs)  # 打乱顺序
    for i in range(0, features.shape[0], batch_size):
        j = torch.LongTensor(_indexs[i:min(i + batch_size, features.shape[0])])
        yield features.index_select(0, j), labels.index_select(0, j)  # 学习一下关键字yield的用法


def predict(X, w, b):
    return torch.mm(X, w) + b


def loss_squared(y, target):
    return (target - y.view(target.size())) ** 2 / 2


def sgd(params, learningrate, batch_size):
    for param in params:
        param.data -= learningrate * param.grad / batch_size  # 注意是param.data，这样的话不会产生梯度


# 超参数
num_inputs = 2
num_examples = 1000
learning_rate = 0.03
batch_size = 10
num_epochs = 3

# 生成训练数据 y = 2x1-3.4x2+4.2
true_w = [2, -3.4]
true_b = 4.2
features = torch.from_numpy(np.random.normal(0, 1, (num_examples, num_inputs)))  # 1000×2
labels = true_w[0] * features[:, 0] + true_w[1] * features[:, 1] + true_b
labels += torch.from_numpy(np.random.normal(0, 0.01, size=labels.size()))  # 1000×1

# 初始化模型参数
w = torch.tensor(np.random.normal(0, 0.01, (num_inputs, 1)), dtype=torch.float64)
w.requires_grad_(requires_grad=True)
b = torch.zeros(1, dtype=torch.float64)
b.requires_grad_(requires_grad=True)

for epoch in range(num_epochs):
    # 训练模型⼀共需要num_epochs个迭代周期，在每⼀个迭代周期中，会使⽤训练数据集中所有样本⼀次
    for X, y in read_data(batch_size, features, labels):
        l = loss_squared(predict(X, w, b), y).sum()  # 注意这里要用sum()将向量转为标量
        l.backward()
        sgd([w, b], learning_rate, batch_size)
        w.grad.data.zero_()
        b.grad.data.zero_()
    train_l = loss_squared(predict(features, w, b), labels)
    print('epoch %d, loss %f' % (epoch + 1, train_l.mean().item()))
print('真实函数y=%.2fx1%.2fx2+%.2f' % (true_w[0], true_w[1], true_b))
print('拟合函数y=%.2fx1%.2fx2+%.2f' % (w[0], w[1], b))
