from cmath import exp, log
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_breast_cancer
from matplotlib import pyplot as plt
from sklearn.metrics import accuracy_score,confusion_matrix,f1_score,recall_score

# X = np.array([[1 2 3],[2 1 2],[3 3 2],[4 1 1],[5 3 2],[5 2 1],[3 1 2],[3 2 1],[4 2 1],[4 5 1]])
# Y = np.array([[0],[1],[1],[0],[1],[1],[0],[0],[1],[0]])

dataset = load_breast_cancer()
X = dataset['data']
Y = np.array(dataset['target']).reshape(-1, 1)

x_train, x_test, y_train, y_test = train_test_split(X,
                                                    Y,
                                                    test_size=0.3,
                                                    random_state=23)


class LogisticRegression:

    def __init__(self, alpha, max_iters) -> None:
        self.alpha = alpha
        self.max_iters = max_iters
        self.b = 0
        self.w = None
        self.loose = []

    def fit(self, X, Y):
        # 加入截距
        X = np.insert(X, 0, 1, axis=1)
        n, f_n = X.shape
        self.w = np.zeros((f_n, 1))
        for i in range(self.max_iters):
            y = np.dot(X, self.w) + self.b
            y = np.clip(y, -5000, 5000)
            # 正例的概率
            p = 1 / (1 + np.exp(-y))
            p = np.clip(p, 0.00001, 0.99999)  # 避免 log(0) 或 log(1) 的问题
            diff = p - Y
            loss = -np.mean(Y * np.log(p) + (1 - Y) * np.log(1 - p))
            self.loose.append(loss)
            dw = np.dot(X.T, diff) / n
            self.b -= self.alpha * np.mean(diff)
            self.w -= self.alpha * dw
        print(self.loose)
        plt.plot(self.loose)
        plt.show()

    def predict(self, X, Y, thresholds):
        X = np.insert(X, 0, 1, axis=1)
        z = np.dot(X, self.w) + self.b
        y = 1 / (1 + np.exp(-z))
        for threshold in thresholds:
            predictions = (y > threshold).astype(int)
            accuracy = accuracy_score(Y, predictions)
            print(f"When threshold is {threshold},Accuracy: {accuracy}")
            print(f"confusion_matrix:{confusion_matrix(Y,predictions)}")
            print(f"f1_score:{f1_score(Y,predictions)}")
            print(f"recall_score:{recall_score(Y,predictions)}")



model = LogisticRegression(0.00001, 1000)
model.fit(x_train, y_train)
thresholds = [0.3, 0.5, 0.7]
model.predict(x_test, y_test, thresholds)
