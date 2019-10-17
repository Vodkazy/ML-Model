# !/usr/bin/env python
# -*- coding: utf-8 -*-
"""
  @ Time     : 2019/01/17 21:25
  @ Author   : Vodka
  @ File     : one2all.py
  @ Software : PyCharm
"""
# load the datasets 导入手写字体数据集
from sklearn.datasets import load_digits
import numpy as np
import matplotlib.pyplot as plt
from scipy import optimize


def draw_digits(classes):
    """
    可视化训练集
    :param classes:
    :return:
    """
    num_classes = len(classes)
    samples_per_class = 5
    for y, cla in enumerate(classes):
        # enumerate()函数用于将一个可遍历的数据对象(如列表、元组或字符串)组合为一个索引序列，同时列出数据和数据下标，一般用在for循环当中
        idxs = np.flatnonzero(target == y)
        # flatnonzero()返回输入矩阵的flatten形式中非0元素的索引，等价于a.ravel().nonzero()[0]
        idxs = np.random.choice(idxs, samples_per_class, replace=False)
        for i, idx in enumerate(idxs):
            plt_idx = i * num_classes + y + 1
            plt.subplot(samples_per_class, num_classes, plt_idx)
            plt.imshow(digits.images[idx].astype('uint8'))
            plt.axis('off')
            if i == 0:
                plt.title(cla)
    plt.show()


def sigmoid(Z):
    """
    :param Z:
    :return:
    """
    return 1 / (1 + np.exp(-Z))


def h(theta, X):
    """
    :param theta:
    :param X:
    :return:
    """
    return sigmoid(X.dot(theta))


def gradient(theta, X, Y):
    """
    求解梯度
    :param theta:
    :param X:
    :param Y:
    :return:
    """
    m, n = X.shape
    theta = theta.reshape(-1, 1)  # 在使用了reshape（-1，1）之后，数据变成了一列
    H = h(theta, X)
    grad = np.zeros((X.shape[1], 1))
    theta_1 = theta[1:, :]
    grad = X.T.dot((H - Y)) / m
    grad[1:, :] += reg * theta_1 / m  # theta0 without reg
    g = grad.ravel()
    return g


def cost_function(theta, X, Y):
    """
    代价函数
    :param theta:
    :param X:
    :param Y:
    :return:
    """
    m = X.shape[0]
    J = 0
    theta = theta.reshape((X.shape[1], 1))  # 这步必须有
    grad = np.zeros((X.shape[1], 1))
    theta_1 = theta[1:, :]
    J = -1 * np.sum(Y * np.log(h(theta, X)) + (1 - Y) * np.log((1 - h(theta, X)))) / m + 0.5 * reg * np.sum(
        theta_1 * theta_1) / m
    return J


def one2all(X, Y, num_class):
    """
    一对多分类，训练多个分类器
    :param X:
    :param Y:
    :param num_class:
    :return:
    """
    m, n = X.shape
    thetas = np.zeros((n, num_class))
    # 每次只训练一个分类器（将数字认成i的能力），每次传进去的Y向量除了Y==i的地方是1，其他都是0
    # 相当于使用Y==i的数据作为positive data去拟合所有Y==i的X所在的函数曲线，其他Y!=i的数据都是negetive data
    for i in range(num_class):
        theta = np.zeros((X.shape[1], 1)).ravel()
        # 使用优化算法去训练
        res = optimize.fmin_cg(cost_function, x0=theta, fprime=gradient, args=(X, Y == i), maxiter=500)
        thetas[:, i] = res
    return thetas


def predict(thetas, X):
    """
    预测
    :param thetas:
    :param X:
    :return:
    """
    pred = np.argmax(h(thetas, X), axis=1)  # 选出数值最大的下标作为分类
    return pred


if __name__ == '__main__':
    digits = load_digits()
    print(digits.keys())
    data = digits.data
    target = digits.target
    reg = 1.0

    # Randomly select 50 data points to display
    classes = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
    draw_digits(classes)

    X = data
    Y = target
    Y = Y.reshape((-1, 1))
    # 标准化
    X_mean = np.mean(X, axis=0)
    X -= X_mean
    m = X.shape[0]
    X = np.hstack((np.ones((m, 1)), X))  # add the one
    print(X.shape, Y.shape)

    thetas = one2all(X, Y, 10)
    print(thetas.shape)

    # 测试
    m, n = data.shape
    example_size = 10
    example_index = np.random.choice(m, example_size)
    for i, idx in enumerate(example_index):
        print(
            "%d th example is number %d,we predict it as %d" % (
            i, target[idx], predict(thetas, X[idx, :].reshape(1, -1))))
