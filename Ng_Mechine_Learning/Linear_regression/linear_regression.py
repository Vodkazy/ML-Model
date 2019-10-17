# !/usr/bin/env python
# -*- coding: utf-8 -*-
"""
  @ Time     : 2019/01/16 08:17
  @ Author   : Vodka
  @ File     : linear_regression_scratch.py
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


def draw_data(X, Y):
    """
    :param X:
    :param Y:
    :return:
    """
    plt.scatter(X[:, -1:], Y, color='b', marker='o')
    plt.xlabel('x')
    plt.xlabel('y')
    plt.show()


def draw_result(X, Y):
    """
    :param X:
    :param Y:
    :return:
    """
    # 可视化边界线
    plt.subplot(211)
    plt.scatter(X[:, 1], Y, color='r', marker='x')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.plot(X[:, 1], X.dot(theta), '-', color='black')
    # 可视化一下cost变化曲线
    plt.subplot(212)
    plt.plot(J_history)
    plt.xlabel('iters')
    plt.ylabel('cost')
    plt.show()


def h(theta, X):
    """
    :param theta:
    :param X:
    :return:
    """
    return X.dot(theta)


def cost_function(theta, X, Y):
    """
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
    data = load_data('ex1data1.txt')
    X = data[:, :-1]  # 使用数字指定列的话会返回一个一维数组，但是如果用切片则会返回二维的
    X = np.hstack((np.ones((X.shape[0], 1)), X))  # 加一列
    m = X.shape[0]  # 行数
    # X.size = m*(n+1) theta.size = (n+1)*1
    Y = data[:, -1:]
    # 描述训练集
    draw_data(X, Y)

    theta = np.zeros((2, 1))
    alpha = 0.01
    iterate_times = 1000
    theta, J_history = gradient_descent(theta, X, Y, alpha, iterate_times)

    # 描述训练结果
    draw_result(X, Y)
