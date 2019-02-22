import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures


def load_data(filename):
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


# 可视化训练集
def draw_data(data):
    X = data[:, :-1]
    Y = data[:, -1:]
    data_0 = np.where(Y.ravel() == 0)
    data_1 = np.where(Y.ravel() == 1)
    plt.scatter(X[data_0, 0], X[data_0, 1], marker='o', color='yellow', label='y=0')
    plt.scatter(X[data_1, 0], X[data_1, 1], marker='+', color='black', label='y=1')
    plt.xlabel('score1')
    plt.ylabel('score2')
    plt.legend()
    plt.show()


# 添加多项式特征，例如x1*x2等
def mapFeature(X1, X2):
    degree = 6
    out = np.ones((X1.shape[0], 1))
    for i in np.arange(1, degree + 1, 1):
        for j in np.arange(0, i + 1, 1):
            temp = X1 ** (i - j) * X2 ** (j)
            out = np.hstack((out, temp))
    return out


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def h(theta, x):
    return sigmoid(x.dot(theta))


# 正则化用于逻辑回归
def cost_reg(theta, XX, YY, reg):
    m = XX.shape[0]
    J = 0
    grad = np.zeros((XX.shape[1], 1))
    theta_1 = theta[1:, :]
    J = -1 * np.sum(YY * np.log(h(theta, XX)) + (1 - YY) * np.log((1 - h(theta, XX)))) / m + 0.5 * reg * np.sum(
        theta_1 * theta_1) / m
    grad = XX.T.dot((h(theta, XX) - YY)) / m
    grad[1:, :] += reg * theta_1 / m  # theta0 without reg
    return J, grad


# 实现batch gradient decent批量梯度下降法
def bgd(X_train, y_train, theta, alpha=0.1, iters=5000, reg=1):
    J_history = []
    for i in range(iters):
        cost, grad = cost_reg(theta, X_train, y_train, reg)
        theta = theta - alpha * grad
        J_history.append(float(cost))
        if i % 200 == 0:
            print('iter=%d,cost=%f ' % (i, cost))
    return theta, J_history


data = load_data('ex2data2.txt')
draw_data(data)
X = data[:, :-1]
Y = data[:, -1:]
X1 = data[:, 0:1]
X2 = data[:, 1:2]
X_map = mapFeature(X1, X2)
theta = np.zeros((X_map.shape[1], 1))
reg = 1
cost, grad = cost_reg(theta, X_map, Y, reg)
print(cost)

W = 0.001 * np.random.randn(X_map.shape[1], 1).reshape((-1, 1))
theta, J_history = bgd(X_map, Y, W)

plt.plot(J_history)
plt.xlabel('iters')
plt.ylabel('j_cost')
plt.show()
# 可视化一下cost

# plot the scatter
label0 = np.where(Y.ravel() == 0)
plt.scatter(X[label0, 0], X[label0, 1], marker='x', color='r', label='0')
label1 = np.where(Y.ravel() == 1)
plt.scatter(X[label1, 0], X[label1, 1], marker='o', color='b', label='1')
plt.xlabel('Exam 1 score')
plt.ylabel('Exam 2 score')
plt.legend(loc='upper left')
# plot the boundary
poly = PolynomialFeatures(6)
x1Min = X[:, 0].min()
x1Max = X[:, 0].max()
x2Min = X[:, 1].min()
x2Max = X[:, 1].max()
xx1, xx2 = np.meshgrid(np.linspace(x1Min, x1Max), np.linspace(x2Min, x2Max))
h1 = poly.fit_transform(np.c_[xx1.ravel(), xx2.ravel()]).dot(theta)
h2 = sigmoid(poly.fit_transform(np.c_[xx1.ravel(), xx2.ravel()]).dot(theta))  # boundary
h1 = h1.reshape(xx1.shape)
h2 = h2.reshape(xx1.shape)
plt.contour(xx1, xx2, h1, [0.5], colors='b', linewidth=.5)
plt.contour(xx1, xx2, h2, [0.5], colors='black', linewidth=.5)
plt.show()
