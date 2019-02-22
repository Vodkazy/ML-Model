# svm分类器，使用的sklearn库函数
import scipy.io as sio
import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm


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
    plt.scatter(X[label0, 0], X[label0, 1], s=10, marker='x', label='Positive')
    plt.scatter(X[label1, 0], X[label1, 1], s=10, marker='o', label='Negative')
    plt.title("RBF-SVM")
    plt.legend()
    plt.show()


# 导入数据
data = sio.loadmat('ex6data3.mat')
X = data['X']
y = data['y']
Xval = data['Xval']
yval = data['yval']
print(data)

# 画图
label0 = np.where(y.ravel() == 0)
label1 = np.where(y.ravel() == 1)
plt.scatter(X[label0, 0], X[label0, 1], s=10, marker='x', label='Positive')
plt.scatter(X[label1, 0], X[label1, 1], s=10, marker='o', label='Negative')
plt.legend()
plt.show()

# 训练模型
# Training SVM with RBF Kernel (Dataset 3)
# 这里需要注意一下，在高斯核函数中，参数sigma与RBF核函数中的gamma的关系是：gamma= 1/2*(sigma**2)
# 所以当sigma =0.1 ，gamma= 50
svc = svm.SVC(kernel='rbf', gamma=50, C=1.0)
svc.fit(X, y.ravel())

# 看一下训练集合以及验证集合的准确率
# 这里我就没有用交叉验证了，直接用的实验的结果c=1,sigma=0.1。
# 读者当然可以试试实现交叉验证，选择合适的超参数值
y_pred = svc.predict(X)
yval_pred = svc.predict(Xval)
acc_train = np.mean(y_pred == y.ravel())
acc_val = np.mean(yval_pred == yval.ravel())
print("the accuracy of train data set : ", acc_train)
print("the accuracy of validation data set : ", acc_val)

# 总结一下：
# C越大：高方差，低偏差
# sigma越小==gamma越大：高方差，低偏差

# 可视化一下决策边界
plot_decision_boundary(lambda x: svc.predict(x), X, y, 0.3)
