# !/usr/bin/env python
# -*- coding: utf-8 -*-
"""
  @ Time     : 2019/02/21 10:32
  @ Author   : Vodka
  @ File     : gaussian_kernels_divide.py
  @ Software : PyCharm
"""
# svm分类器，使用的sklearn库函数
import scipy.io as sio
import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm


def plot_decision_boundary(pred_func, X, y, gap):
    """
    :param pred_func:
    :param X:
    :param y:
    :param gap:
    :return:
    """
    # 设定最大最小值，附加一点点gap,进行边缘填充
    x_min, x_max = X[:, 0].min() - gap, X[:, 0].max() + gap
    y_min, y_max = X[:, 1].min() - gap, X[:, 1].max() + gap
    h = 0.01

    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))

    # 用预测函数预测一下
    Z = pred_func(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    # plt.xlim(x_min,x_max)
    # plt.ylim(y_min,y_max)
    # 然后画出图
    plt.contour(xx, yy, Z, )
    label0 = np.where(y.ravel() == 0)
    label1 = np.where(y.ravel() == 1)
    plt.scatter(X[label0, 0], X[label0, 1], s=10, marker='x', label='Positive')
    plt.scatter(X[label1, 0], X[label1, 1], s=10, marker='o', label='Negative')
    plt.legend()
    plt.show()


if __name__ == '__main__':
    # 导入数据
    data = sio.loadmat('ex6data2.mat')
    X = data['X']
    y = data['y']
    print(X, y)

    # 画图
    label0 = np.where(y.ravel() == 0)
    label1 = np.where(y.ravel() == 1)
    plt.scatter(X[label0, 0], X[label0, 1], s=10, marker='x', label='Positive')
    plt.scatter(X[label1, 0], X[label1, 1], s=10, marker='o', label='Negative')
    plt.legend()
    plt.show()

    # 训练模型
    # Training SVM with RBF Kernel (Dataset 2)
    # 这里需要注意一下，在高斯核函数中，参数sigma与RBF核函数中的gamma的关系是：gamma= 1/2*(sigma**2)
    # 所以当sigma =0.1 ，gamma= 50
    svc = svm.SVC(kernel='rbf', gamma=50, C=1.0)
    svc.fit(X, y.ravel())
    # 看一下训练的准确率
    y_pred = svc.predict(X)
    acc_train = np.mean(y_pred == y.ravel()) * 100
    print("the accuracy of train data set : %.2f %%" % acc_train)

    # 可视化决策边界
    plot_decision_boundary(lambda x: svc.predict(x), X, y, 0.3)
