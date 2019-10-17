# !/usr/bin/env python
# -*- coding: utf-8 -*-
"""
  @ Time     : 2019/02/21 10:31
  @ Author   : Vodka
  @ File     : gaussian_kernels.py
  @ Software : PyCharm
"""
import matplotlib.pyplot as plt
from sklearn import svm
import numpy as np
import pandas as pd
import scipy.io as sio


def gaussian_kernel(x1, x2, sigma):
    """
    kernel function 高斯核函数
    :param x1:
    :param x2:
    :param sigma:
    :return:
    """
    return np.exp(- np.power(x1 - x2, 2).sum() / (2 * (sigma ** 2)))


if __name__ == '__main__':
    # 加载数据
    mat = sio.loadmat('ex6data2.mat')
    data = pd.DataFrame(mat.get('X'), columns=['X1', 'X2'])
    data['y'] = mat.get('y')
    print(data)

    # 可视化数据
    positive = data[data['y'].isin([1])]
    negative = data[data['y'].isin([0])]
    fig, ax = plt.subplots()
    ax.scatter(positive['X1'], positive['X2'], s=10, marker='x', label='Positive')
    ax.scatter(negative['X1'], negative['X2'], s=10, marker='o', label='Negative')
    ax.legend()
    plt.show()

    svc = svm.SVC(C=100, kernel='rbf', gamma=10, probability=True)
    svc.fit(data[['X1', 'X2']], data['y'])
    svc.score(data[['X1', 'X2']], data['y'])
    fig, ax = plt.subplots()
    data['SVM1 Confidence'] = svc.decision_function(data[['X1', 'X2']])
    ax.scatter(data['X1'], data['X2'], s=10, c=data['SVM1 Confidence'], cmap='RdPu')
    plt.show()
