import numpy as np
from sklearn.datasets import make_blobs
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from matplotlib import pyplot as plt
from sklearn.metrics import accuracy_score

# 创建二维数据集
dataset = make_blobs(n_samples=1000,
                     n_features=2,
                     centers=2,
                     random_state=23,
                     cluster_std=2.5)

X = dataset[0]
Y = dataset[1].reshape(-1, 1)
Y = np.where(Y == 0, -1, 1)

# 划分训练集和测试集
x_train, x_test, y_train, y_test = train_test_split(X,
                                                    Y,
                                                    test_size=0.3,
                                                    random_state=23)

# 特征缩放
ss_x = StandardScaler()
x_train = ss_x.fit_transform(x_train)
x_test = ss_x.transform(x_test)


class Support_vector_machine:

    def __init__(self,
                 learning_rate,
                 C,
                 max_iterations,
                 kernel="Liner",
                 gamma=None) -> None:
        self.learning_rate = learning_rate
        self.C = C
        self.max_iterations = max_iterations
        self.w = None
        self.b = None
        self.kernel = kernel
        self.gamma = gamma

    def fit(self, X, Y):
        num_sample, num_feature = X.shape
        self.w = np.zeros(num_feature)
        self.b = 0
        for epoch in range(self.max_iterations):
            for index, x in enumerate(X):
                # 正负决策超平面（约束条件）
                self._kernel(X[index], X)
                condition = Y[index] * (np.dot(x, self.w) + self.b) >= 1
                if condition:
                    self.w -= self.learning_rate * self.w
                else:
                    self.w -= self.learning_rate * (
                        self.w - self.C * np.dot(x.reshape(-1, 1), Y[index]))
                    self.b -= self.learning_rate * (-Y[index])

    def _kernel(self, x1, x2):
        if self.kernel == 'Linear':
            return np.dot(x1, x2.T)
        if self.kernel == 'RBF':
            if self.gamma == None:
                self.gamma = 1 / x2.shape[1]
            temp = np.exp(-self.gamma * np.sum((x1 - x2)**2, axis=1))
            return temp

    def predict(self, X):
        y_pre = np.sign(np.dot(X, self.w) + self.b)
        return np.where(y_pre == 0, 1, y_pre)


# 创建并训练SVM模型
model = Support_vector_machine(0.01, 1, 100, "RBF")
model.fit(x_train, y_train)
y_pre = model.predict(x_test)

# 计算准确性
accuracy = accuracy_score(y_test, y_pre)
print(f"准确性：{accuracy:.2f}")

# 生成决策边界的网格
x1_min, x1_max = np.min(X[:, 0]), np.max(X[:, 0])
x2_min, x2_max = np.min(X[:, 1]), np.max(X[:, 1])
x1, x2 = np.meshgrid(np.arange(x1_min, x1_max, 0.01),
                     np.arange(x2_min, x2_max, 0.01))

# 预测网格上的点
z = model.predict(np.c_[np.ravel(x1), np.ravel(x2)])
z = z.reshape(x1.shape)

# 绘制散点图和决策边界
plt.scatter(X[:, 0],
            X[:, 1],
            c=Y.flatten(),
            cmap=plt.cm.Paired,
            label='sample')
plt.contourf(x1, x2, z, cmap=plt.cm.Paired, alpha=0.3, levels=[-1, 0, 1])
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.legend()
plt.title('Support Vector Machine')
plt.show()
