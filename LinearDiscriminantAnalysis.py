from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import numpy as np
from math import sqrt
from collections import Counter

iris = load_breast_cancer()
X = iris.data
y = iris.target

# 计算类内均值
def compute_class_means(X, y):
    class_means = {}
    unique_classes = np.unique(y)

    for cls in unique_classes:
        class_means[cls] = np.mean(X[y == cls], axis=0)

    return class_means


# 计算类内散度矩阵
def compute_within_class_scatter_matrix(X, y, class_means):
    num_features = X.shape[1]
    S_W = np.zeros((num_features, num_features))

    unique_classes = np.unique(y)

    for cls in unique_classes:
        class_data = X[y == cls]
        diff = class_data - class_means[cls]
        S_W += np.dot(diff.T, diff)

    return S_W


# 计算类间散度矩阵
def compute_between_class_scatter_matrix(X, y, class_means, overall_mean):
    num_features = X.shape[1]
    S_B = np.zeros((num_features, num_features))

    unique_classes = np.unique(y)

    for cls in unique_classes:
        n = np.sum(y == cls)
        mean_diff = class_means[cls] - overall_mean
        S_B += n * np.outer(mean_diff, mean_diff)

    return S_B


def lda(X, y, n_components=1):
    class_means = compute_class_means(X, y)
    overall_mean = np.mean(X, axis=0)

    S_W = compute_within_class_scatter_matrix(X, y, class_means)
    S_B = compute_between_class_scatter_matrix(X, y, class_means, overall_mean)

    eigenvalues, eigenvectors = np.linalg.eig(np.linalg.inv(S_W).dot(S_B))

    # Sort eigenvectors based on eigenvalues in descending order
    sorted_indices = np.argsort(eigenvalues)[::-1]
    eigenvectors = eigenvectors[:, sorted_indices]

    # Select the top n_components eigenvectors
    W = eigenvectors[:, :n_components]

    # Project the data onto the new subspace
    X_lda = np.dot(X, W)

    return X_lda


# Example usage:
# Assuming X is your data and y is the corresponding labels
X_lda = lda(X, y, n_components=3)

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

# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X_lda[:,0],
                                                    y,
                                                    test_size=0.2,
                                                    random_state=42)

for k in range(1,8,2):
    model = KNN(k)
    model.fit(X_train, y_train)
    result = model.predict(X_test)

    acc = model.accuracy(result, y_test)
    print(f"Accuracy when k is {k} : {acc}")