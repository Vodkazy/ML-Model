import scipy.io as sio
import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm

# 定义一个函数来画决策边界
def plot_decision_boundary(pred_func, X, y, gap):
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
    plt.scatter(X[label0, 0], X[label0, 1], s=30, marker='x', label='Positive')
    plt.scatter(X[label1, 0], X[label1, 1], s=30, marker='o', label='Negative')
    plt.title("LinearSVM")
    plt.legend()
    plt.show()


# 导入数据
data = sio.loadmat('ex6data1.mat')
X = data['X']
y = data['y']
print(X,y)

# 画图
label0 = np.where(y.ravel() == 0)
label1 = np.where(y.ravel() == 1)
plt.scatter(X[label0,0], X[label0,1], s=30, marker='x', label='Positive')
plt.scatter(X[label1,0], X[label1,1], s=30, marker='o', label='Negative')
plt.legend()
plt.show()

# 训练模型
svc = svm.LinearSVC(C=1)
svc.fit(X,y.ravel())
y1_pred = svc.predict(X)
acc_train = np.mean(y1_pred==y.ravel())*100
print("the accuracy of train data set : %.2f %%" %acc_train)

plot_decision_boundary(lambda x:svc.predict(x),X,y,0.1)