# !/usr/bin/env python
# -*- coding: utf-8 -*-
"""
  @ Time     : 2018/12/14 10:34
  @ Author   : Vodka
  @ File     : logic_regression.py
  @ Software : PyCharm
"""
import numpy as np
import matplotlib.pyplot as plt
from scipy import optimize


def load_data(filename):
    """
    :param filename:
    :return:
    """
    data = []
    file = open(filename)
    for line in file.readlines():
        lineArr = line.strip().split(',')
        col_num = len(lineArr)
        temp = []
        for i in range(col_num):
            temp.append(float(lineArr[i]))
        data.append(temp)

    return np.array(data)


def draw_data(data):
    """
    可视化训练集
    :param data:
    :return:
    """
    X = data[:, :-1]
    Y = data[:, -1:]
    data_0 = np.where(Y.ravel() == 0)
    data_1 = np.where(Y.ravel() == 1)
    plt.scatter(X[data_0, 0], X[data_0, 1], marker='o', color='y', label='Not Admitted')
    plt.scatter(X[data_1, 0], X[data_1, 1], marker='+', color='black', label='Admitted')
    plt.xlabel('score1')
    plt.ylabel('score2')
    plt.legend()
    plt.show()


def draw_result(X, Y):
    """
    可视化一下线性的决策边界
    :param X:
    :param Y:
    :return:
    """
    label = np.array(Y)
    index_0 = np.where(label.ravel() == 0)
    plt.scatter(X[index_0, 1], X[index_0, 2], marker='x', color='b', label='Not admitted', s=15)
    index_1 = np.where(label.ravel() == 1)
    plt.scatter(X[index_1, 1], X[index_1, 2], marker='o', color='r', label='Admitted', s=15)
    # show the decision boundary
    x1 = np.arange(20, 100, 0.5)
    x2 = (- res[0] - res[1] * x1) / res[2]
    plt.plot(x1, x2, color='black')
    plt.xlabel('X1')
    plt.ylabel('X2')
    plt.legend(loc='upper left')
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
    :param theta:
    :param X:
    :param Y:
    :return:
    """
    m, n = X.shape
    theta = theta.reshape(-1, 1)  # 在使用了reshape（-1，1）之后，数据变成了一列
    H = h(theta, X)
    grad = np.zeros((X.shape[1], 1))
    grad = X.T.dot((H - Y)) / m
    g = grad.ravel()
    return g


def cost_function(theta, X, Y):
    """
    :param theta:
    :param X:
    :param Y:
    :return:
    """
    m = X.shape[0]
    theta = theta.reshape(-1, 1)  # 不加这一语句就不对
    # '*'代表点乘 .dot是矩阵相乘
    return (np.sum((-Y * (np.log(h(theta, X)))) - (1 - Y) * (np.log(1 - h(theta, X))))) / m

if __name__ == '__main__':
    data = load_data('ex2data1.txt')
    draw_data(data)

    X = data[:, :-1]
    X = np.hstack((np.ones((X.shape[0], 1)), X))  # 加一列
    m = X.shape[0]  # 行数
    Y = data[:, -1:]
    theta = np.zeros((X.shape[1], 1))

    # res = optimize.minimize(cost_function,x0=theta,args=(X,Y),method='BFGS',jac=gradient,options={'gtol': 1e-6, 'disp': True})
    res = optimize.fmin_cg(cost_function, x0=theta, fprime=gradient, args=(X, Y))  # cost个gradient的第一个参数必须是theta
    print(res)  # 第一个值返回的是最优的theta数组

    draw_result(X, Y)
