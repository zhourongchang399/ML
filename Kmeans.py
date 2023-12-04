from cProfile import label
import pandas as pd
import numpy as np
from sklearn.datasets import make_blobs
from sklearn.model_selection import train_test_split
from math import sqrt
import matplotlib.pyplot as plt

dataset = make_blobs(n_samples=1000,
                     centers=5,
                     random_state=23,
                     n_features=2,
                     cluster_std=1.5)
X = dataset[0]
Y = dataset[1]

plt.scatter(X[:,0],X[:,1],label='sample')
plt.xlabel('x')
plt.ylabel('y')
plt.show()
x_train, x_test, y_train, y_test = train_test_split(X,
                                                    Y,
                                                    test_size=0.2,
                                                    random_state=23)

class Kmeans:

    def __init__(self, K, max_iters=100) -> None:
        self.K = K
        self.max_iters = max_iters
        self.centroids = []
        self.labels = []
        self.sse = 0

    def fit(self, X):
        # np.random.seed(1)
        # 随机选取K个簇中心点
        self.centroids = X[np.random.choice(len(X), size=self.K,
                                            replace=False)]
        for i in range(self.max_iters):
            # print(f"This is {i+1} iters of centroids:{self.centroids}")
            self.labels = []
            new_centroids = []
            self.sse = 0
            # 计算sample和各个中心点的距离，并归类
            for x in X:
                d = np.sum((self.centroids - x)**2, axis=1)
                self.sse += d[np.argmin(d)]
                self.labels.append(np.argmin(np.sqrt(d)))
            # 更新新的中心点
            for k in range(self.K):
                new_centroids.append(
                    np.mean(X[np.where(
                        np.array(self.labels) == k, True, False)],
                            axis=0))
            # 判断新旧中心点是否变化
            if np.allclose(new_centroids, self.centroids):
                break
            self.centroids = new_centroids

        return self.labels, self.centroids, self.sse


# 手肘法，选择合适的k值
sses = []
for i in range(2, 11):
    model = Kmeans(i)
    result = model.fit(X)
    sses.append(result[2])

x = np.arange(2, 11, 1)
plt.plot(x, sses)
plt.show()

model = Kmeans(5)
result = model.fit(X)
labels = np.array(result[0])
x0 = X[labels == 0]
x1 = X[labels == 1]
x2 = X[labels == 2]
x3 = X[labels == 3]
x4 = X[labels == 4]


plt.scatter(x0[:,0],x0[:,1],c='r',marker='o',label='label_01')
plt.scatter(x1[:,0],x1[:,1],c='b',marker='.',label='label_02')
plt.scatter(x2[:,0],x2[:,1],c='g',marker='+',label='label_03')
plt.scatter(x3[:,0],x3[:,1],c='#000000',marker='s',label='label_04')
plt.scatter(x4[:,0],x4[:,1],c='#641203',marker='*',label='label_05')

plt.show()