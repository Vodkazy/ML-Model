# !/usr/bin/env python
# -*- coding: utf-8 -*-
"""
  @ Time     : 2019/01/16 08:17
  @ Author   : Vodka
  @ File     : linear_multiple.py
  @ Software : PyCharm
"""
import numpy as np
import matplotlib.pyplot as plt


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


def draw_cost(J_history):
    """
    plot the j_cost,绘制训练的cost曲线,可以调节学习率
    :param J_history:
    :return:
    """
    plt.plot(J_history, color='g')
    plt.xlabel('iters')
    plt.ylabel('J_cost')
    plt.title('cost variety')
    plt.show()


def featureNormalize(X):
    """
    定义一下特征缩放函数，因为每个特征的取值范围不同，且差异很大
    :param X:
    :return:
    """
    avg = np.mean(X, axis=0)  # 均值
    sigma = np.std(X, axis=0)  # 标准差
    X_norm = (X - avg) / sigma
    return X_norm, avg, sigma


def h(theta, X):
    """
    预测函数
    :param theta:
    :param X:
    :return:
    """
    return X.dot(theta)


def cost_function(theta, X, Y):
    """
    代价函数
    :param theta:
    :param X:
    :param Y:
    :return:
    """
    m = X.shape[0]
    result = np.sum(np.square(h(theta, X) - Y)) / (2 * m)
    return result


def gradient_descent(theta, X, Y, alpha, iterate_times):
    """
    梯度下降
    :param theta:
    :param X:
    :param Y:
    :param alpha:
    :param iterate_times:
    :return:
    """
    J_history = []
    m = X.shape[0]
    for i in range(iterate_times):
        theta = theta - alpha * X.T.dot(h(theta, X) - Y) / m
        cost = cost_function(theta, X, Y)
        J_history.append(cost)
    return theta, J_history

if __name__ == '__main__':
    data = load_data('ex1data2.txt')
    X = data[:, :-1]
    X, avr, sigma = featureNormalize(X)
    Y = data[:, -1:]
    m = X.shape[0]
    X = np.hstack((np.ones((m, 1)), X))  # 加一列
    theta = np.zeros((X.shape[1], 1))
    alpha = 0.01
    iterate_times = 1000

    theta, J_history = gradient_descent(theta, X, Y, alpha, iterate_times)
    draw_cost(J_history)
