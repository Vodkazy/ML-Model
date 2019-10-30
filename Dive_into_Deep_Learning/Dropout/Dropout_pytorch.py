#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
  @ Time     : 2019/10/17 下午10:19
  @ Author   : Vodka
  @ File     : Dropout_pytorch .py
  @ Software : PyCharm
"""
import torch
import torchvision
import sys
import matplotlib.pyplot as plt
import torch.nn as nn

batch_size = 256
num_inputs = 784
num_hiddens1 = 256
num_hiddens2 = 256
num_outputs = 10
dropout_1, dropout_2 = 0.2, 0.5


# 和加载数据集有关的函数
def load_data_fashion_mnist(batch_size, resize=None, root='~/Desktop'):
    """Download the fashion mnist dataset and then load into memory."""
    # 将28*28的矩阵转换为一维的784的矩阵
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

    return train_iter, test_iter


# 获取标签
def get_fashion_mnist_labels(labels):
    text_labels = ['t-shirt', 'trouser', 'pullover', 'dress', 'coat',
                   'sandal', 'shirt', 'sneaker', 'bag', 'ankle boot']
    return [text_labels[int(i)] for i in labels]


# 展示mnist数据集图表
def show_fashion_mnist(images, labels):
    # 这里的_表示我们忽略（不使用）的变量
    _, figs = plt.subplots(1, len(images), figsize=(12, 12))
    for f, img, lbl in zip(figs, images, labels):
        f.imshow(img.view((28, 28)).numpy())
        f.set_title(lbl)
        f.axes.get_xaxis().set_visible(False)
        f.axes.get_yaxis().set_visible(False)
    # plt.show()


# 计算准确率
def evaluate_accuracy(data_iter, net):
    acc_sum, n = 0.0, 0
    for X, y in data_iter:
        acc_sum += (net(X).argmax(dim=1) == y).float().sum().item()
        n += y.shape[0]
    return acc_sum / n


# 定义优化器
def sgd(params, lr, batch_size):
    # 为了和原书保持一致，这里除以了batch_size，但是应该是不用除的，因为一般用PyTorch计算loss时就默认已经
    # 沿batch维求了平均了。
    for param in params:
        param.data -= lr * param.grad / batch_size  # 注意这里更改param时用的param.data


# 训练模型
def train(net, train_iter, test_iter, loss, num_epochs, batch_size,
          params=None, lr=None, optimizer=None):
    for epoch in range(num_epochs):
        train_l_sum, train_acc_sum, n = 0.0, 0.0, 0
        for X, y in train_iter:
            y_hat = net(X)
            l = loss(y_hat, y).sum()

            # 梯度清零
            if optimizer is not None:
                optimizer.zero_grad()
            elif params is not None and params[0].grad is not None:
                for param in params:
                    param.grad.data.zero_()

            l.backward()
            if optimizer is None:
                sgd(params, lr, batch_size)
            else:
                optimizer.step()

            train_l_sum += l.item()
            train_acc_sum += (y_hat.argmax(dim=1) == y).sum().item()
            n += y.shape[0]
        test_acc = evaluate_accuracy(test_iter, net)
        print('epoch %d, loss %.4f, train acc %.3f, test acc %.3f'
              % (epoch + 1, train_l_sum / n, train_acc_sum / n, test_acc))


class FlattenLayer(nn.Module):
    def __init__(self):
        super(FlattenLayer, self).__init__()

    def forward(self, x):  # x shape: (batch, *, *, ...)
        return x.view(x.shape[0], -1)


"""
    在PyTorch中，我们只需要在全连接层后添加Dropout层并指定丢弃概率。在训练模型时，Dropout
    层将以指定的丢弃概率随机丢弃上一层的输出元素；在测试模型时（即model.eval()后），
    Dropout层并不发挥作用。
"""

if __name__ == '__main__':
    # 获取和读取数据
    train_iter, test_iter = load_data_fashion_mnist(batch_size)
    # 定义和初始化模型
    # 输入层784 -> 隐藏层256 -> 隐藏层256 -> 输出层10
    net = nn.Sequential(
        FlattenLayer(),
        nn.Linear(num_inputs, num_hiddens1),
        nn.ReLU(),
        nn.Dropout(dropout_1),
        nn.Linear(num_hiddens1, num_hiddens2),
        nn.ReLU(),
        nn.Dropout(dropout_2),
        nn.Linear(num_hiddens2, num_outputs)
    )

    # CrossEntropyLoss()中已经封装好了包含softmax和交叉熵损失计算的函数
    # 损失函数 CrossEntropyLoss() 与 NLLLoss()类似, 唯一的不同是它为我们去做 softmax 并取对数
    # 可以理解为 CrossEntropyLoss() = log_softmax() + NLLLoss()（负对数似然损失函数）
    # 我们通常使用的cross entropy loss，几乎都可以称作softmax loss
    loss = nn.CrossEntropyLoss()

    # 定义优化算法
    optimizer = torch.optim.SGD(net.parameters(), lr=0.1)

    # 训练模型
    num_epochs = 5
    train(net, train_iter, test_iter, loss, num_epochs, batch_size, None, None, optimizer)
