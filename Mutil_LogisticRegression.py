import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from scipy.optimize import minimize
from matplotlib import pyplot as plt
from sklearn.metrics import accuracy_score, f1_score, recall_score, confusion_matrix

data = load_iris()
X = data["data"]
Y = data['target']
ss_x = StandardScaler()
X = ss_x.fit_transform(X)
x_train, x_test, y_train, y_test = train_test_split(X,
                                                    Y,
                                                    test_size=0.3,
                                                    random_state=23)


class Mutil_LogisticRegression:

    def __init__(self) -> None:
        pass

    def fit(self, X, Y, alpha, max_iterations):
        # 添加偏置项
        bias = np.ones((np.array(X).shape[0], 1))
        X = np.concatenate((X, bias), axis=1)
        X = np.insert(Mutil_LogisticRegression.preparation(X), 0, 1, axis=1)
        Y = Mutil_LogisticRegression.preparation(Y).reshape(-1, 1)
        self.unique_lable = np.unique(Y)
        self.num_label = len(self.unique_lable)
        self.num_example, self.num_feature = X.shape
        self.cost_history = []
        self.alpha = alpha
        #初始化权重
        self.theat = np.zeros((self.num_label, self.num_feature))
        # result = minimize(
        #     lambda current_theat: self.cost_function(X, Y, current_theat),
        #     # 初始权值
        #     theat,
        #     # 优化方法
        #     method='CG',
        #     # 更新梯度
        #     jac=lambda current_theat: self.gradient(
        #         X, Y, current_theat),
        #     # 迭代次数
        #     options={'maxiter': max_iterations},
        #     callback=lambda current_theat: self.cost_history.append(
        #         self.cost_function(X, Y, current_theat)))

        # # 使用优化结果更新模型权重
        # self.optimal_theat = result.x.reshape(self.num_label, -1)
        # print(result)
        # print(f"cost:{self.cost_history}")
        for i in range(max_iterations):
            cost = self.cost_function(X, Y, self.theat)
            self.cost_history.append(cost)
            dw = self.gradient(X, Y, self.theat)
            self.theat -= dw
        plt.plot(self.cost_history)
        plt.show()

    def cost_function(self, X, Y, theat):
        temp_theat = theat.reshape(self.num_label, self.num_feature)
        y = np.dot(X, temp_theat.T)
        y_exp = np.exp(y)
        predictions = np.zeros((self.num_example, self.num_label))
        loss = 0
        for i in range(self.num_label):
            predictions[:, i] = (y_exp[:, i] / np.sum(y_exp, axis=1))
        for index, label in enumerate(self.unique_lable):
            loss += np.sum(
                np.log(predictions[:, index][(Y == label).flatten()]))
        return (-1 / self.num_example) * loss

    def gradient(self, X, Y, theat):
        temp_theat = theat.reshape(self.num_label, self.num_feature)
        y = np.dot(X, temp_theat.T)
        y_exp = np.exp(y)
        predictions = np.zeros((self.num_example, self.num_label))
        for i in range(self.num_label):
            predictions[:, i] = (y_exp[:, i] / np.sum(y_exp, axis=1))
        encoder = OneHotEncoder(sparse=False)
        Y = encoder.fit_transform(Y)
        diff = predictions - Y
        dw = self.alpha * (1 / self.num_example) * np.dot(X.T, diff)
        return dw.T

    @staticmethod
    def preparation(data):
        return np.array(data)

    def predict(self, X):
        bias = np.ones((X.shape[0], 1))
        X = np.concatenate((np.insert(X, 0, 1, axis=1), bias), axis=1)
        y = np.dot(X, self.theat.T)
        y_exp = np.exp(y)
        prediction = np.zeros((X.shape[0], self.num_label))
        for index, label in enumerate(self.unique_lable):
            prediction[:, index] = y_exp[:, index] / np.sum(y_exp, axis=1)
        y_pred = np.argmax(prediction, axis=1)
        return y_pred


model = Mutil_LogisticRegression()
model.fit(x_train, y_train, 0.001, 10000)
y_pred = model.predict(x_test)
oneHot_encode = OneHotEncoder(sparse=False)
y_pred_oneHot = oneHot_encode.fit_transform(y_pred.reshape(-1, 1))
y_test_oneHot = oneHot_encode.fit_transform(y_test.reshape(-1, 1))
print(f"accuracy:{accuracy_score(y_test,y_pred)};")
for i in range(y_pred_oneHot.shape[1]):
    print(f"the {i} class of confusion matrix:\n{confusion_matrix(y_test_oneHot[:,i],y_pred_oneHot[:,i])}")
    print(
        f"the {i} class of F1:{f1_score(y_test_oneHot[:,i],y_pred_oneHot[:,i])};"
    )
    print(
        f"the {i} class of recall:{recall_score(y_test_oneHot[:,i],y_pred_oneHot[:,i])}"
    )
