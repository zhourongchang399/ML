import numpy as np
from collections import Counter
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_iris
from math import sqrt

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


datasets = load_iris()
X = datasets['data']
Y = datasets['target']
x_train, x_test, y_train, y_test = train_test_split(X,
                                                    Y,
                                                    test_size=0.3)


for k in range(1,8,2):
    model = KNN(k)
    model.fit(x_train, y_train)
    result = model.predict(x_test)

    acc = model.accuracy(result, y_test)
    print(f"Accuracy when k is {k} : {acc}")
