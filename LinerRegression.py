import numpy as np
from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from matplotlib import pyplot as plt

dataset = load_boston()
X = dataset['data']
Y = dataset['target']

# 标准化
ss_x = StandardScaler()
ss_y = StandardScaler()

# X = np.array([[1],[2],[3],[4],[5],[6],[7],[8],[9],[10]])
# Y = np.array([[3],[4],[6],[5],[5],[7],[8],[9],[11],[10]])

X = ss_x.fit_transform(X)
Y = ss_y.fit_transform(Y.reshape(-1, 1))
x_train, x_test, y_train, y_test = train_test_split(X,
                                                    Y,
                                                    test_size=0.3,
                                                    random_state=23)

columns = dataset['feature_names']


class LinerRegression:

    def __init__(self, learning_rate, iters) -> None:
        self.learning_rate = learning_rate
        self.iters = iters

    def fit(self, X, Y):
        X = np.array(X)
        # 添加截距
        X = np.insert(X, 0, 1, axis=1)
        num, f_num = X.shape
        self.w = np.random.rand(f_num).reshape(-1, 1)
        min_loss = 1e-4
        self.b = 0
        losses = []
        temp = 0
        for i in range(self.iters):
            y = np.dot(X, self.w) + self.b
            diff = y - Y
            dw = np.dot(diff.T, X) / num
            db = np.mean(diff)
            self.w -= self.learning_rate * dw.T
            self.b -= self.learning_rate * db
            loss = np.mean((diff**2) / 2)
            losses.append(loss)
            if abs(temp - loss) <= min_loss:
                break
            temp = loss
        plt.plot(losses)
        plt.xlabel('iters')
        plt.ylabel('loose')
        plt.show()

    def evaluate(self, X, Y):
        pre_y = np.dot(X, self.w) + self.b
        return np.mean((pre_y - Y)**2)


model = LinerRegression(0.01, 10000)
model.fit(x_train, y_train)
x_test = np.insert(x_test, 0, 1, axis=1)
result = model.evaluate(x_test, y_test)
print(result)
