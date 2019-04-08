# !/usr/bin/env python
# -*- coding: utf-8 -*-
"""
  @ Time     : 2019/01/16 10:34
  @ Author   : Vodka
  @ File     : LogisticRegression.py
  @ Software : PyCharm
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from sklearn import datasets


class LogisticRegression(nn.Module):
    def __init__(self, input_size):
        super(LogisticRegression, self).__init__()
        self.linear = nn.Linear(in_features=input_size, out_features=1, bias=True)

    def forward(self, input):
        """
        :param input:
        :return:
        """
        self.output = F.sigmoid(self.linear(input))
        return self.output


def saveModel(model):
    """
    :param model:
    :return:
    """
    torch.save(model.state_dict(), 'model')
    print("模型保存成功！")


def load_data():
    """
    :return:
    """
    train_data = []  # 4个变量 1个函数值  这里只用target为0或者1的数据
    test_data = []
    dataset = datasets.load_iris()

    # 统计target为0或1的数据的个数
    cnt = 0
    for i in dataset.target:
        if i < 2:
            cnt += 1

    divide_pos1 = int(cnt * 2 / 5)
    divide_pos2 = int(cnt * 3 / 5)

    # 训练集测试集数据分块 8-2开
    _index = 0
    for line in dataset.data:
        if (_index < divide_pos1 or _index >= divide_pos2) and _index < cnt:
            train_data.append([line[0], line[1], line[2], line[3]])
        elif (divide_pos1 <= _index and _index < divide_pos2) and _index < cnt:
            test_data.append([line[0], line[1], line[2], line[3]])
        _index += 1

    _index = 0
    _index_train = 0
    _index_test = 0
    for line in dataset.target:
        if (_index < divide_pos1 or _index >= divide_pos2) and _index < cnt:
            train_data[_index_train].append(line)
            _index_train += 1
        elif (divide_pos1 <= _index and _index < divide_pos2) and _index < cnt:
            test_data[_index_test].append(line)
            _index_test += 1
        _index += 1
    # print(train_data,test_data)
    return train_data, test_data


def DivideXY(dataset):
    """
    :param dataset:
    :return:
    """
    x = []
    y = []
    for item in dataset:
        x.append([item[0], item[1], item[2], item[3]])
        y.append(item[4])
    return x, y


def sigmoid(x):
    """
    :param x:
    :return:
    """
    return 1.0 / (1 + np.exp(x))


if __name__ == '__main__':
    train_data, test_data = load_data()
    model = LogisticRegression(3)

    epoch = 10000
    mini_batch = 10
    cnt_update = int(80 / mini_batch)

    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters())
    # print(train_data)
    for i in range(epoch):
        for j in range(cnt_update):
            data = ((np.array(train_data[j * mini_batch:80])) if (j == cnt_update - 1) else(
                np.array(train_data[j * mini_batch:(j + 1) * mini_batch])))

            # 训练的每个数据都应该是个数组 整个训练集的每个元素也是个数组
            x = torch.Tensor(list(data[:, :3]))
            y = torch.Tensor(list(data[:, 4:]))

            optimizer.zero_grad()
            loss = criterion(model(x), y)
            loss.backward()
            optimizer.step()
            print("当前为第{}次训练， 本次训练已更新{}%  Loss: {}".format(i + 1, float((j + 1) / cnt_update) * 100.0, loss.item()))

    saveModel(model)

    # 测试
    test_data = ((np.array(test_data)))
    x = torch.Tensor(list(test_data[:, :3]))
    y = np.array(list(test_data[:, 4:]))
    # list的样子是[x,x,x,x,x,x]
    # numpy.ndarray的样子是[x x x x x x]
    predict = np.array(list(model(x).detach().numpy()))
    # detach().numpy() 将tensor转换为numpy.ndarray
    precision = np.sum((abs(y - predict) <= 0.01) * 1) / len(test_data)
    print("测试准确率: {}%".format(precision * 100))
