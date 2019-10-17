# !/usr/bin/env python
# -*- coding: utf-8 -*-
"""
  @ Time     : 2019/02/22 18:21
  @ Author   : Vodka
  @ File     : pca.py
  @ Software : PyCharm
"""
# PCA用来降维
import scipy.io as sio
import numpy as np
import matplotlib.pyplot as plt


def featureNormalize(X):
    """
    feature normalize 特征归一化
    :param X:
    :return:
    """
    X_norm = X
    mu = np.zeros((1, X.shape[1]))
    sigma = np.zeros((1, X.shape[1]))

    mu = np.mean(X, axis=0)  # mean value of every feature
    sigma = np.std(X, axis=0)  # std of every feature
    X_norm = (X - mu) / sigma

    return X_norm, mu, sigma


def pca(data_normal):
    """
    complete the pca
    :param data_normal:
    :return:
    """
    m, n = data_normal.shape
    sigma = data_normal.T.dot(data_normal) / m  # np.cov()
    U, S, V = np.linalg.svd(sigma)  # np.linalg.eig()
    return U, S, V


def projectData(x, u, k):
    """
    降维2->1
    :param x:
    :param u:
    :param k:
    :return:
    """
    z = np.zeros((x.shape[0], k))
    u_reduce = u[:, :k]  # get the first k line
    z = x.dot(u_reduce)  # [m,n]*[n,k] = [m,k]
    return z


def recoverData(z, u, k):
    """
    还原 1-->2
    :param z:
    :param u:
    :param k:
    :return:
    """
    x_rec = np.zeros((z.shape[0], u.shape[0]))
    u_reduce = u[:, :k]
    x_rec = z.dot(u_reduce.T)  # [m,k]*[k,n] = [m,n]
    return x_rec


if __name__ == '__main__':
    # 加载数据
    data = sio.loadmat('ex7data1.mat')
    X = data['X']
    print(X)
    plt.scatter(X[:, 0], X[:, 1], marker='x', color='r')
    plt.xlabel('X1')
    plt.ylabel('X2')
    plt.show()

    X_norm, mu, sigma = featureNormalize(X)
    U, S, V = pca(X_norm)
    print(U, S, V)

    Z = projectData(X_norm, U, k=1)  # n-->k
    print(Z.shape)
    X_rec = recoverData(Z, U, k=1)  # get back k--->n
    print(X_rec.shape)

    # 可视化一下
    plt.plot(X_norm[:, 0], X_norm[:, 1], 'bo')
    plt.plot(X_rec[:, 0], X_rec[:, 1], 'rx')
    plt.xlabel('x1')
    plt.ylabel('x2')
    plt.show()
