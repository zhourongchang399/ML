from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.datasets import load_boston
from sklearn.metrics import r2_score, mean_squared_error
import numpy as np

dataset = load_boston()
X, y = dataset["data"], dataset["target"]
X_train, X_test, y_train, y_test = train_test_split(X,
                                                    y,
                                                    random_state=23,
                                                    train_size=0.3)


class GBDTRegression:

    def __init__(self, n_estimators, learning_rate, Subsampling,
                 max_depth) -> None:
        self.models = []
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.Subsampling = Subsampling
        self.max_depth = max_depth

    def fit(self, X, y):
        n_sample, _ = X.shape
        diff_y = y.copy()

        for _ in range(self.n_estimators):
            # 选择子集训练
            index = np.random.choice(n_sample,
                                     size=int(self.Subsampling * n_sample),
                                     replace=False)
            x_sub, y_sub = X[index], diff_y[index]

            # 构建决策树
            tree = DecisionTreeRegressor(max_depth=self.max_depth)
            tree.fit(x_sub, y_sub)
            predict = tree.predict(X)
            diff_y -= self.learning_rate * predict
            self.models.append(tree)

    def predict(self, X):
        predictions = np.array(
            [self.learning_rate * model.predict(X) for model in self.models])
        predict = np.sum(predictions, axis=0)
        return predict


gbdt = GBDTRegression(100, 0.1, 1, 3)
gbdt.fit(X_train, y_train)
predict = gbdt.predict(X_test)
print(f"R2:{r2_score(y_test, predict)}")
print(f"mean_squared_error:{mean_squared_error(y_test, predict)}")

gbr = GradientBoostingRegressor()
gbr.fit(X_train, y_train)
predict = gbr.predict(X_test)
print(f"R2:{r2_score(y_test, predict)}")
print(f"mean_squared_error:{mean_squared_error(y_test, predict)}")