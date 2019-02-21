import numpy as np
import pandas as pd
import sklearn.svm
import seaborn as sns
import scipy.io as sio
import matplotlib.pyplot as plt

# 加载数据
mat = sio.loadmat('ex6data1.mat')
print(mat)
data = pd.DataFrame(mat.get('X'), columns=['X1', 'X2'])
data['y'] = mat.get('y')
print(data)

# 画图
positive = data[data['y'].isin([1])]
negative = data[data['y'].isin([0])]
fig, ax = plt.subplots()
ax.scatter(positive['X1'], positive['X2'], s=50, marker='x', label='Positive')
ax.scatter(negative['X1'], negative['X2'], s=50, marker='o', label='Negative')
ax.legend()
plt.show()

# 定义SVM
# C = 1
svc1 = sklearn.svm.LinearSVC(C=1, loss='hinge')
svc1.fit(data[['X1','X2']],data['y'])
print(svc1.score(data[['X1','X2']],data['y']))
data['SVM1 Confidence'] = svc1.decision_function(data[['X1', 'X2']])
# C = 100
svc100 = sklearn.svm.LinearSVC(C=100, loss='hinge')
svc100.fit(data[['X1', 'X2']], data['y'])
print(svc100.score(data[['X1', 'X2']], data['y']))
data['SVM100 Confidence'] = svc100.decision_function(data[['X1', 'X2']])

# 渐变色画图看置信度
fig, ax = plt.subplots()
ax.scatter(data['X1'], data['X2'], s=50, c=data['SVM100 Confidence'], cmap='RdBu')
ax.set_title('SVM (C=100) Decision Confidence')
plt.show()

# 查看数据
print(data.head())