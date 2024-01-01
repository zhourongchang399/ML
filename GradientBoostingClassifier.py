from sklearn.datasets import load_iris, load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score, confusion_matrix
import numpy as np


class GBClassifier:

    # 初始化
    def __init__(self, n_estimator, learn_rate, subsample, max_depth) -> None:

        # models:模型列表
        # n_estomator:迭代次数
        # learn_rate:学习率
        # subsample:子集
        # max_depth:最大深度
        # classes:多分类

        self.models = []
        self.n_estimator = n_estimator
        self.learn_rate = learn_rate
        self.subsample = subsample
        self.max_depth = max_depth
        self.classes = None

    def softmax(self, y):
        y_exp = np.exp(y)
        predict = y_exp / np.sum(y_exp, axis=1, keepdims=True)
        return predict

    def gradient(self, y, y_pre):
        return y - y_pre

    def fit(self, X, y):

        n_sample, _ = X.shape
        self.classes = np.unique(y)
        n_classes = len(self.classes)

        # 独热编码
        y = np.eye(n_classes)[y]

        t = np.ones(n_sample)
        t = np.eye(n_classes)[t.astype(int)]

        prediction = self.softmax(t)

        for _ in range(self.n_estimator):
            residual = self.gradient(y, prediction)

            # 获取子数据集
            index = np.random.choice(n_sample,
                                     size=int(n_sample * self.subsample),
                                     replace=False)
            X_sub, y_sub = X[index], residual[index]

            # 构建弱学习器
            model = DecisionTreeRegressor(max_depth=self.max_depth)
            model.fit(X_sub, y_sub)
            prediction += self.learn_rate * model.predict(X)
            self.models.append(model)

    def predict(self, X):
        predictions = np.array([model.predict(X) for model in self.models])
        prediction = np.sum(self.learn_rate * predictions, axis=0)

        return np.argmax(prediction, axis=1)


# 加载数据集
dataset = load_iris()
X, y = dataset.data, dataset.target
X_train, X_test, y_train, y_test = train_test_split(X,
                                                    y,
                                                    random_state=23,
                                                    test_size=0.3)

gbdt = GBClassifier(100, 0.1, 1, 1)
gbdt.fit(X_train, y_train)
predictiton = gbdt.predict(X_test)
print(
    f"f1_score:{f1_score(y_test,predictiton,average='micro')},accuracy_score:{accuracy_score(y_test,predictiton)},precision_score:{precision_score(y_test,predictiton,average='micro')},recall_score:{recall_score(y_test,predictiton,average='micro')}"
)

gbc = GradientBoostingClassifier()
gbc.fit(X_train, y_train)
predictiton = gbc.predict(X_test)
print(
    f"f1_score:{f1_score(y_test,predictiton,average='micro')},accuracy_score:{accuracy_score(y_test,predictiton)},precision_score:{precision_score(y_test,predictiton,average='micro')},recall_score:{recall_score(y_test,predictiton,average='micro')}"
)