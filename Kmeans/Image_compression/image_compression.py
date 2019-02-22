import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from skimage import io

# 加载图片
pic = io.imread('bird_small.png') / 255.
io.imshow(pic)
print(pic.shape)
data = pic.reshape(128 * 128, 3)

# 训练模型
model = KMeans(n_clusters=16, n_init=100, n_jobs=-1)
model.fit(data)
centroids = model.cluster_centers_
print(centroids.shape)
C = model.predict(data)
print(C.shape)
compressed_pic = centroids[C].reshape((128, 128, 3))

# 展示对比图
fig, ax = plt.subplots(1, 2)
ax[0].imshow(pic)
ax[1].imshow(compressed_pic)
plt.show()
