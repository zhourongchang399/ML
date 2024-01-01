from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_breast_cancer
from sklearn.metrics import accuracy_score
import xgboost as xgb
import numpy as np

class XGBoostNode:
    def __init__(self, depth=0, max_depth=None):
        self.depth = depth
        self.max_depth = max_depth
        self.feature_index = None
        self.threshold = None
        self.value = None
        self.left = None
        self.right = None

class XGBoostTree:
    def __init__(self, max_depth=3, reg_lambda=1, reg_alpha=1):
        self.max_depth = max_depth
        self.root = None
        self.reg_lambda = reg_lambda
        self.reg_alpha = reg_alpha

    def sigmoid(self, X):
        return 1 / (1 + np.exp(-X))

    def compute_gradient(self, y, y_pred):
        return -(y / y_pred) + (1 - y) / (1 - y_pred)

    def compute_hessian(self, y, y_pred):
        return y / (y_pred**2) + (1 - y) / (1 - y_pred)**2

    def compute_loss(self, y, y_pred):
        return -np.mean(y * np.log(self.sigmoid(y_pred)) + (1 - y) * np.log(1 - self.sigmoid(y_pred)))

    def find_best_split(self, X, y):
        m, n = X.shape
        if m <= 1:
            return None, None  # Not enough samples to split

        num_features = n
        max_gain = float('-inf')  # Use negative infinity to ensure gain is maximized
        best_feature = None
        best_threshold = None

        for feature in range(num_features):
            thresholds = np.unique(X[:, feature])

            for threshold in thresholds:
                left_indices = X[:, feature] <= threshold
                right_indices = ~left_indices

                if np.sum(left_indices) == 0 or np.sum(right_indices) == 0:
                    continue  # Skip if all samples are in one side

                gain = self.compute_gain(y, y[left_indices], y[right_indices])

                if gain > max_gain:
                    max_gain = gain
                    best_feature = feature
                    best_threshold = threshold

        return best_feature, best_threshold

    def compute_objective(self, y):
        # 计算损失函数
        loss = self.compute_loss(y, np.log(self.sigmoid(y)))

        # 计算正则化项
        regularization = self.regularization_term(len(y),np.array(np.mean(y)))

        # 目标函数是损失函数和正则化项的总和
        objective = loss + regularization

        return objective


    def compute_gain(self, y, left_y, right_y):
        current_objective = self.compute_objective(y)
        left_objective = self.compute_objective(left_y)
        right_objective = self.compute_objective(right_y)

        gain = (current_objective - left_objective - right_objective)

        return gain

    def regularization_term(self, num,node):
        # L2 regularization term
        l2_term = 0.5 * self.reg_lambda * np.sum(node**2)

        # L1 regularization term
        l1_term = self.reg_alpha * np.sum(num)

        return l2_term + l1_term

    def fit_tree(self, X, y, node):
        if node.depth == self.max_depth:
            node.value = np.mean(y)
            return

        best_feature, best_threshold = self.find_best_split(X, y)

        if best_feature is None:
            node.value = np.mean(y)
            return

        node.feature_index = best_feature
        node.threshold = best_threshold

        left_indices = X[:, best_feature] <= best_threshold
        right_indices = ~left_indices

        node.left = XGBoostNode(depth=node.depth + 1, max_depth=self.max_depth)
        self.fit_tree(X[left_indices, :], y[left_indices], node.left)

        node.right = XGBoostNode(depth=node.depth + 1, max_depth=self.max_depth)
        self.fit_tree(X[right_indices, :], y[right_indices], node.right)

    def fit(self, X, y):
        self.root = XGBoostNode(max_depth=self.max_depth)
        self.fit_tree(X, y, self.root)

    def predict_tree(self, x, node):
        if node.value is not None:
            return node.value

        if x[node.feature_index] <= node.threshold:
            return self.predict_tree(x, node.left)
        else:
            return self.predict_tree(x, node.right)

    def predict(self, X):
        return np.array([self.predict_tree(x, self.root) for x in X])

class XGboost:

    def __init__(self,
                 n_trees=10,
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

    def sigmoid(self, X):
        return 1 / (1 + np.exp(-X))

    #计算损失
    def compute_loss(self, y, y_pred):
        # 交叉熵损失函数
        return -np.mean(y * np.log(self.sigmoid(y_pred)) +
                        (1 - y) * np.log(1 - self.sigmoid(y_pred)))

    def fit(self, X, y):
        n_sample, n_feature = X.shape
        prediction = np.zeros(n_sample, dtype=float)

        for _ in range(self.n_trees):
            residuals = y - prediction

            index = np.random.choice(n_sample,
                                     int(self.subsample * n_sample),
                                     replace=False)
            X_sub, residuals_sub = X[index], residuals[index]

            tree = XGBoostTree(max_depth=3)
            tree.fit(X_sub, residuals_sub)

            prediction += self.learning_rate * tree.predict(X)

            self.trees.append(tree)

    def predict(self, X, thresehold = 0.5):

        prediction =  np.sum(self.learning_rate * tree.predict(X)
                      for tree in self.trees)
        return (prediction > thresehold).astype(int)


dataset = load_breast_cancer()
X, y = dataset.data, dataset.target
X_train, X_test, y_train, y_test = train_test_split(X,
                                                    y,
                                                    test_size=0.3,
                                                    random_state=23)

model = XGboost()
model.fit(X_train, y_train)
predict = model.predict(X_test)
print(f"accuracy_score:{accuracy_score(predict,y_test)}")

model = xgb.XGBClassifier()
model.fit(X_train, y_train)
predict = model.predict(X_test)
print(f"accuracy_score:{accuracy_score(predict,y_test)}")