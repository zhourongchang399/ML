from operator import index
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_boston
from sklearn.metrics import r2_score, mean_squared_error
import xgboost as xgb
import numpy as np


class Node:

    def __init__(self, data, depth) -> None:
        self.depth = depth
        self.left_node = None
        self.right_node = None
        self.isLeaf = False
        self.value = None
        self.n_sample = None
        self.data = data
        self.threshold = None
        self.feature_index = None


class XGBoostTree:

    def __init__(self, max_depth=3, reg_lambda=1, reg_alpha=1) -> None:
        self.max_depth = max_depth
        self.reg_lambda = reg_lambda
        self.reg_alpha = reg_alpha
        self.root = None

    def gradient(self, y, y_pred):
        return y - y_pred

    def hessian(self, y, y_pred):
        return np.ones(len(y))

    #计算目标函数
    def obj(self, y, y_pred, depth):
        T = 2**depth
        G = np.sum(self.gradient(y, y_pred)**2)
        H = np.sum(self.hessian(y, y_pred) + self.reg_lambda)
        o = self.reg_alpha * T - 0.5 * (G / H)
        return o

    def information_gain(self, X, y, feature_index, threshold, depth):
        #父节点的目标函数值
        parent_obj = self.obj(y, np.mean(y), depth)

        #左右节点
        left_mask = X[:, feature_index] <= threshold
        right_mask = ~left_mask

        if np.sum(left_mask) == 0 or np.sum(right_mask) == 0:
            return 0

        left_child_obj = self.obj(y[left_mask], np.mean(y[left_mask]), depth)
        right_child_obj = self.obj(y[right_mask], np.mean(y[right_mask]),
                                   depth)

        left_weight = np.sum(left_mask) / len(y)
        right_weight = np.sum(right_mask) / len(y)

        child_entropy = left_weight * left_child_obj + right_weight * right_child_obj

        #增益
        return parent_obj - child_entropy

    # 查找目标函数增益最大的划分节点
    def find_best_split(self, X, y, num_features, depth):
        best_feature = None
        best_threshold = None
        max_obj_gain = -float('inf')

        index = np.random.choice(num_features,
                                 int(num_features * 0.8),
                                 replace=False)

        for feature_index in index:
            thresholds = np.unique(X[:, feature_index])
            for threshold in thresholds:
                obj_gain = self.information_gain(X, y, feature_index,
                                                 threshold, depth)
                if obj_gain > max_obj_gain:
                    max_obj_gain = obj_gain
                    best_feature = feature_index
                    best_threshold = threshold

        return best_feature, best_threshold

    def build_tree(self, X, y, depth):
        node = Node(data=(X, y), depth=depth)

        # 判断是否终止
        if depth == self.max_depth or len(np.unique(y)) == 1 or X.shape[0] < 2:
            node.isLeaf = True
            node.value = np.mean(y)
            node.n_sample = X.shape[0]
        else:
            num_features = X.shape[1]
            best_feature, best_threshold = self.find_best_split(
                X, y, num_features, depth)
            left_mask = X[:, best_feature] <= best_threshold
            right_mask = ~left_mask
            node.feature_index = best_feature
            node.threshold = best_threshold
            node.left_node = self.build_tree(X[left_mask], y[left_mask],
                                             depth + 1)
            node.right_node = self.build_tree(X[right_mask], y[right_mask],
                                              depth + 1)

        return node

    def fit(self, X, y):
        self.root = self.build_tree(X, y, 0)

    def predict_tree(self, node, X):
        # 递归预测
        if node.isLeaf:
            return node.value
        elif X[node.feature_index] <= node.threshold:
            return self.predict_tree(node.left_node, X)
        else:
            return self.predict_tree(node.right_node, X)

    def predict(self, X):
        predictions = np.array([self.predict_tree(self.root, x) for x in X])
        return predictions


class XGboost:

    def __init__(self,
                 n_trees=100,
                 max_depth=3,
                 learning_rate=0.1,
                 reg_lambda=1,
                 reg_alpha=1,
                 subsample=1) -> None:
        self.n_trees = n_trees
        self.max_depth = max_depth
        self.learning_rate = learning_rate
        self.reg_lambda = reg_lambda
        self.reg_alpha = reg_alpha
        self.subsample = subsample
        self.trees = []

    def gradient(self, y, y_pred):
        return y - y_pred

    def fit(self, X, y):
        n_sample, n_feature = X.shape
        prediction = np.zeros(n_sample, dtype=float)

        for _ in range(self.n_trees):
            residuals = self.gradient(y, prediction)

            index = np.random.choice(n_sample,
                                     int(self.subsample * n_sample),
                                     replace=False)

            X_sub, residuals_sub = X[index], residuals[index]

            tree = XGBoostTree(max_depth=self.max_depth,
                               reg_alpha=self.reg_alpha,
                               reg_lambda=self.reg_lambda)
            tree.fit(X_sub, residuals_sub)

            prediction += self.learning_rate * tree.predict(X)

            self.trees.append(tree)

    def predict(self, X):
        prediction = np.array(
            [self.learning_rate * tree.predict(X) for tree in self.trees])
        return np.sum(prediction, axis=0)


dataset = load_boston()
X, y = dataset.data, dataset.target
X_train, X_test, y_train, y_test = train_test_split(X,
                                                    y,
                                                    test_size=0.3,
                                                    random_state=23)

model = XGboost(n_trees=200,
                max_depth=3,
                learning_rate=0.1,
                reg_lambda=1,
                reg_alpha=1,
                subsample=0.8)
model.fit(X_train, y_train)
predict = model.predict(X_test)
mse = mean_squared_error(y_test, predict)
print(f"R2:{r2_score(y_test, predict)}")
print(f"Mean Squared Error: {mse}")

model = xgb.XGBRegressor()
model.fit(X_train, y_train)
predict = model.predict(X_test)
# Calculate Mean Squared Error (MSE)
mse = mean_squared_error(y_test, predict)
print(f"R2:{r2_score(y_test, predict)}")
print(f"Mean Squared Error: {mse}")