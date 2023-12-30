from cgi import test
from re import S
from tkinter import Y
from sklearn.datasets import load_breast_cancer,load_iris
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
import numpy as np


class AdaBoost:

    def __init__(self, X, y, n_estimator) -> None:
        self.X = X
        self.y = y
        self.n_estimator = n_estimator
        self.models = []
        self.alphas = []
        self.label = None

    def fit(self):
        n_sample, n_feature = self.X.shape
        self.label = np.unique(self.y)
        # 初始化权重
        D = np.ones(n_sample) / n_sample

        for _ in range(self.n_estimator):

            # 构建弱分类器
            model = DecisionTreeClassifier(max_depth=1)
            model.fit(self.X, self.y, sample_weight=D)
            predict = model.predict(self.X)

            self.models.append(model)

            # 计算加权错误率
            err = np.sum(D[predict != self.y]) / np.sum(D)

            # 计算弱分类器权重
            alpha = np.log((1 - err) / err) / 2

            self.alphas.append(alpha)

            # 调整权重
            D[predict != self.y] *= np.exp(alpha)
            D[predict == self.y] *= np.exp(-alpha)
            D /= np.sum(D)

    def predict(self, X):
        predictions = np.array([model.predict(X) for model in self.models])
        weighted_predictions = []
        for label in self.label:
            temp_predict = (predictions == label).astype(int)
            weighted_predictions.append(np.dot(self.alphas, temp_predict))
        result = np.argmax(weighted_predictions, axis=0)
        return result


dataset = load_iris()
X, y = dataset["data"], dataset["target"]
X_train, X_test, y_train, y_test = train_test_split(X,
                                                    y,
                                                    test_size=0.2,
                                                    random_state=23)

adaboostModel = AdaBoost(X_train, y_train, 10)
adaboostModel.fit()
prediction = adaboostModel.predict(X_test)
print(accuracy_score(prediction, y_test))
