#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
  @ Time     : 2019/10/17 下午3:55
  @ Author   : Vodka
  @ File     : MLP_scratch .py
  @ Software : PyCharm
"""
#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
  @ Time     : 2019/9/30 下午9:49
  @ Author   : Vodka
  @ File     : softmax_regression_scratch .py
  @ Software : PyCharm
"""
import torch
import torchvision
import numpy as np
import sys
import matplotlib.pyplot as plt
# Softmax就是对每个数据取个以e为底的指数，然后算完后除以所有数据的sum

# 和加载数据集有关的函数
def load_data_fashion_mnist(batch_size, resize=None, root='~/Desktop'):
    """Download the fashion mnist dataset and then load into memory."""
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


def get_fashion_mnist_labels(labels):
    text_labels = ['t-shirt', 'trouser', 'pullover', 'dress', 'coat',
                   'sandal', 'shirt', 'sneaker', 'bag', 'ankle boot']
    return [text_labels[int(i)] for i in labels]


def show_fashion_mnist(images, labels):
    # 这里的_表示我们忽略（不使用）的变量
    _, figs = plt.subplots(1, len(images), figsize=(12, 12))
    for f, img, lbl in zip(figs, images, labels):
        f.imshow(img.view((28, 28)).numpy())
        f.set_title(lbl)
        f.axes.get_xaxis().set_visible(False)
        f.axes.get_yaxis().set_visible(False)
    # plt.show()


# 定义优化器
def sgd(params, lr, batch_size):
    # 为了和原书保持一致，这里除以了batch_size，但是应该是不用除的，因为一般用PyTorch计算loss时就默认已经
    # 沿batch维求了平均了。
    for param in params:
        param.data -= lr * param.grad / batch_size  # 注意这里更改param时用的param.data


# 实现softmax
def softmax(X):
    X_exp = X.exp()
    partition = X_exp.sum(dim=1, keepdim=True)
    return X_exp / partition  # 这里应用了广播机制

# 实现ReLU
def ReLU(X):
    return torch.max(input=X, other=torch.tensor(0.0))

# 定义模型
def net(X):
    X = X.view((-1, num_inputs))
    H = ReLU(torch.mm(X,W1) +b1)
    return torch.mm(H,W2)+b2

# 计算准确率
def evaluate_accuracy(data_iter, net):
    acc_sum, n = 0.0, 0
    for X, y in data_iter:
        acc_sum += (net(X).argmax(dim=1) == y).float().sum().item()
        n += y.shape[0]
    return acc_sum / n


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


# 获取和读取数据
batch_size = 256
train_iter, test_iter = load_data_fashion_mnist(batch_size)

# 初始化模型参数
num_inputs = 784
num_hiddens = 256
num_outputs = 10
num_epochs, lr = 5, 100
W1 = torch.tensor(np.random.normal(0, 0.01, (num_inputs, num_hiddens)),dtype=torch.float)
b1 = torch.zeros(num_hiddens, dtype=torch.float)
W2 = torch.tensor(np.random.normal(0, 0.01, (num_hiddens, num_outputs)),dtype=torch.float)
b2 = torch.zeros(num_outputs, dtype=torch.float)
W1.requires_grad_(requires_grad=True)
b1.requires_grad_(requires_grad=True)
W2.requires_grad_(requires_grad=True)
b2.requires_grad_(requires_grad=True)

# 训练模型，这里直接使用PyTorch提供的包括softmax运算和交叉熵损失计算的函数
train(net, train_iter, test_iter, torch.nn.CrossEntropyLoss(), num_epochs, batch_size, [W1, b1, W2, b2], lr)

# 预测
X, y = iter(test_iter).next()
true_labels = get_fashion_mnist_labels(y.numpy())
pred_labels = get_fashion_mnist_labels(net(X).argmax(dim=1).numpy())
titles = [true + '\n' + pred for true, pred in zip(true_labels, pred_labels)]
show_fashion_mnist(X[0:9], titles[0:9])
