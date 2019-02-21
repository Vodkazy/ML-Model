# kmeans算法的实现
import scipy.io as sio
import numpy as np
import matplotlib.pyplot as plt

# 加载数据
data = sio.loadmat('ex7data2.mat')
X = data['X']
print(X)
plt.scatter(X[:,0],X[:,1],marker='x',color = 'r')
plt.xlabel('X1')
plt.ylabel('X2')
plt.show()

# 计算距离
def computeDistance(A, B):
    return np.sqrt(np.sum(np.square(A - B)))

# 为数据集x中的每个点都找到离它最近的质心
def findClosestCentroids(x, centroids,K=3):
    k = centroids.shape[0]
    m = x.shape[0]
    idx = np.zeros((x.shape[0], 1))
    for i in range(m):
        minDist = np.inf
        minIndex = -1
        for j in range(K):
            distance = computeDistance(x[i, :], centroids[j, :])
            if distance < minDist:
                minDist = distance
                minIndex = j
        idx[i, :] = minIndex
    return idx

# compute the mean of the data ,it is centroids
# 计算data的均值，改变centroids
def change_centroids(x,idx,K=3):
    m,n = x.shape
    centroids = np.zeros((K,n))
    for i in range(K):
        index = np.where(idx.ravel() == i)
        centroids[i] = np.mean(x[index],axis=0)
    return centroids

# 一开始应该随机初始化centroids
def randCentroids(x,k=3):
    m,n = x.shape
    centroids = np.zeros((k,n))
    randIndex = np.random.choice(m,k)
    centroids = x[randIndex]
    return centroids

# 定义kmeans class
class Kmeans(object):
    def __init__(self):
        pass
    def runKmeans(self,data,init_centroids,iers=10,k=3):
        idx = None
        centroids = None
        #记录一下每个质心的变化
        cent0 = np.zeros((iers,initial_centroids.shape[1]))
        cent1 = np.zeros((iers,initial_centroids.shape[1]))
        cent2 = np.zeros((iers,initial_centroids.shape[1]))
        for i in range(iers):
            idx = findClosestCentroids(data,init_centroids)
            centroids = change_centroids(data,idx,k)
            init_centroids = centroids
            cent0[i,:] = centroids[0,:]
            cent1[i,:] = centroids[1,:]
            cent2[i,:] = centroids[2,:]
        return idx,centroids,cent0,cent1,cent2

# 全局变量
K = 3 #聚类数
initial_centroids = randCentroids(X) #聚类中心点

#用初始化的centroids测试一下聚类的过程和结果
kmeans = Kmeans()
idx,centroids,cent0,cent1,cent2 = kmeans.runKmeans(X,initial_centroids)
idx0 = np.where(idx.ravel()==0)
idx1 = np.where(idx.ravel()==1)
idx2 = np.where(idx.ravel()==2)
plt.scatter(X[idx0,0],X[idx0,1],marker='x',color = 'r')
plt.scatter(X[idx1,0],X[idx1,1],marker='*',color = 'g')
plt.scatter(X[idx2,0],X[idx2,1],marker='+',color = 'y')
#plt.scatter(centroids[:,0],centroids[:,1],marker='o',color = 'cyan')
#plt.scatter(initial_centroids[:,0],initial_centroids[:,1],\
#            marker='^',color = 'black',linewidths=10)
plt.plot(cent0[:,0],cent0[:,1],"b-o")
plt.plot(cent1[:,0],cent1[:,1],"r-o")
plt.plot(cent2[:,0],cent2[:,1],"g-o")
plt.xlabel('X1')
plt.ylabel('X2')
plt.show()