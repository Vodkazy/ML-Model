# !/usr/bin/env python
# -*- coding: utf-8 -*-
"""
  @ Time     : 2019/02/23 23:52
  @ Author   : Vodka
  @ File     : spam_filter.py
  @ Software : PyCharm
"""
# 在这一部分中，我们的目标是使用SVM来构建垃圾邮件过滤器。
# 在练习文本中，有一个任务涉及一些文本预处理，以获得适合SVM处理的格式的数据。
# 然而，这个任务很简单（将字词映射到为练习提供的字典中的ID），而其余的预处理步骤
# （如HTML删除，词干，标准化等）已经完成。 我将跳过这些预处理步骤，
# 其中包括从预处理过的训练集构建分类器，以及将垃圾邮件和非垃圾邮件转换为单词出现次数的向量的测试数据集。
from sklearn import svm
from sklearn import metrics
from sklearn.linear_model import LogisticRegression
import scipy.io as sio

# 加载训练数据
train_data = sio.loadmat('spamTrain.mat')
train_X, train_y = train_data.get('X'), train_data.get('y').ravel()
print(train_X, train_y)

# 加载测试数据
test_data = sio.loadmat('spamTest.mat')
test_X, test_y = test_data.get('Xtest'), test_data.get('ytest').ravel()
print(test_X, test_y)

# 训练SVM模型
svc = svm.SVC()
svc.fit(train_X, train_y)
pred = svc.predict(test_X)
print(metrics.classification_report(test_y, pred))

# 对比下线性回归
logit = LogisticRegression()
logit.fit(train_X, train_y)
pred = logit.predict(test_X)
print(metrics.classification_report(test_y, pred))
