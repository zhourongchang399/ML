from sklearn.preprocessing import StandardScaler
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import numpy as np
from math import sqrt
from collections import Counter
from matplotlib import pyplot as plt


class PCA:

    def __init__(self, X, n_dismension=1) -> None:
        self.X = X
        self.n_dismension = n_dismension

    # 计算协方差矩阵
    def compute_covariance(self, X):
        feature_nums = X.shape[1]
        sample_nums = X.shape[0]
        S = np.dot(X.T, X) / (sample_nums - 1)
        return S

    def compute_value_vector(self, S):
        value, vector = np.linalg.eig(S)
        sorted_indices = np.argsort(value)[::-1]
        vector = vector[:, sorted_indices]
        W = vector[:, :self.n_dismension]
        return W

    def fit(self):
        S = self.compute_covariance(self.X)
        W = self.compute_value_vector(S)
        X_pca = np.dot(self.X, W)
        return X_pca


# 计算欧式距离
def distance(x1, x2):
    total = np.sum((x1 - x2)**2)
    return sqrt(total)


class KNN:

    def __init__(self, k=3) -> None:
        self.k = k

    def fit(self, X, Y):
        self.x_train = X
        self.y_train = Y

    def predict(self, X):
        predic_label = []
        for x_test in X:
            # 计算例子与每个训练样本之间的距离
            distances = [distance(x_train, x_test) for x_train in self.x_train]
            # 获取前k个index
            k_indices = np.argsort(distances)[:self.k]
            k_labels = [self.y_train[i] for i in k_indices]
            # 投票选出最高票数的label
            predic_label.append(Counter(k_labels).most_common(1)[0][0])
        return predic_label

    def accuracy(self, result, y):
        error_total = 0.0
        for i in range(y.shape[0]):
            if y[i] != result[i]:
                error_total += 1
        return 1 - (error_total / y.shape[0])


data = load_breast_cancer()
X, y = data["data"], data["target"]

# 创建 StandardScaler 对象
scaler = StandardScaler()

# 对数据进行标准化
X_scaled = scaler.fit_transform(X)

pca = PCA(X_scaled, 2)
X_pca = pca.fit()

# 绘制降维后的数据散点图
plt.scatter(X_pca[:, 0], X_pca[:, 1], c=y, cmap='viridis')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.show()

# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X_pca,
                                                    y,
                                                    test_size=0.2,
                                                    random_state=42)

for k in range(1, 8, 2):
    model = KNN(k)
    model.fit(X_train, y_train)
    result = model.predict(X_test)

    acc = model.accuracy(result, y_test)
    print(f"Accuracy when k is {k} : {acc}")