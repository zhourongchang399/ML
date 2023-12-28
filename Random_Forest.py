import numpy as np
from sklearn.datasets import load_breast_cancer, load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix, precision_score, recall_score

data = load_breast_cancer()
x_train, x_test, y_train, y_test = train_test_split(data['data'],
                                                    data['target'],
                                                    test_size=0.3,random_state=23)


class Node:

    def __init__(self, data, depth) -> None:
        self.left = None
        self.right = None
        self.label = None
        self.depth = depth
        self.is_leaf = None
        self.data = data
        self.best_threshold = None
        self.best_feature = None


class Tree:

    def __init__(self, Node, select_feature) -> None:
        self.Node = Node
        self.select_feature = select_feature


class Random_forest:

    def __init__(self, tree_nums, layer_nums, X, y) -> None:
        self.tree_nums = tree_nums
        self.layers = layer_nums
        self.X = X
        self.y = y
        self.result = []
        self.trees = []

    def gini_coefficient(self, y):
        labels = np.unique(y)
        gini = 1
        for label in labels:
            gini -= (np.sum(y == label) / len(y))**2
        return gini

    def gini_gain(self, X, y, threshold, feature_index):
        parent_gini = self.gini_coefficient(y)

        left_mask = X[:, feature_index] <= threshold
        right_mask = ~left_mask

        left_gini = (len(y[left_mask]) / len(y)) * self.gini_coefficient(
            y[left_mask])
        right_gini = (len(y[right_mask]) / len(y)) * self.gini_coefficient(
            y[right_mask])

        gini_gain = parent_gini - left_gini - right_gini
        return gini_gain

    def find_best_split(self, X, y, feature_nums):
        best_feature = None
        best_threshold = None
        max_gini_gain = -float('inf')
        for feature_index in range(feature_nums):
            thresholds = np.unique(X[:, feature_index])
            for threshold in thresholds:
                gini_gain = self.gini_gain(X, y, threshold, feature_index)
                if (gini_gain > max_gini_gain):
                    max_gini_gain = gini_gain
                    best_feature = feature_index
                    best_threshold = threshold
        return best_feature, best_threshold

    def build_dicision_tree(self, X, y, depth, max_depth):
        node = Node(data=(X, y), depth=depth)
        if (depth == max_depth or len(np.unique(y)) == 1):
            node.label = np.argmax(np.bincount(np.ravel(y)))
            node.is_leaf = True
        else:
            best_feature, best_threshold = self.find_best_split(
                X, y, X.shape[1])
            left_mask = X[:, best_feature] <= best_threshold
            right_mask = ~left_mask
            node.left = self.build_dicision_tree(X[left_mask, :],
                                                 y[left_mask],
                                                 depth=depth + 1,
                                                 max_depth=max_depth)
            node.right = self.build_dicision_tree(X[right_mask, :],
                                                  y[right_mask],
                                                  depth=depth + 1,
                                                  max_depth=max_depth)
            node.best_threshold = best_threshold
            node.best_feature = best_feature

        return node

    def fit(self):
        sample_nums, feature_nums = self.X.shape
        for num in range(self.tree_nums):
            select_sample = np.random.choice(sample_nums, size=int(sample_nums*0.8,), replace=False)
            select_feature = np.random.choice(feature_nums, size=int(feature_nums*0.8), replace=False)
            # select_sample = np.random.randint(0, 2,
            #                                   size=sample_nums).astype(bool)
            # select_feature = np.random.randint(0, 2,
            #                                    size=feature_nums).astype(bool)
            x = self.X[select_sample][:,select_feature]
            y = self.y[select_sample]

            tree = Tree(self.build_dicision_tree(x, y, 0, self.layers),
                        select_feature)
            self.trees.insert(len(self.trees), tree)

    def predict(self, X):
        for tree in self.trees:
            predictions = []
            x = X[:, tree.select_feature]
            node = tree.Node
            for i in x:
                predictions.append(self.predict_tree(node, i))
            self.result.append(predictions)
        self.result = np.array(self.result)
        predictions = np.apply_along_axis(lambda x: np.argmax(np.bincount(x)),axis=0,arr=self.result)
        return predictions

    def predict_tree(self, node, X):
        # 递归预测
        if node.is_leaf:
            return node.label
        elif X[node.best_feature] <= node.best_threshold:
            return self.predict_tree(node.left, X)
        else:
            return self.predict_tree(node.right, X)


rf = Random_forest(2, 5, x_train, y_train)
rf.fit()
predictions = rf.predict(x_test)
# 计算混淆矩阵
conf_matrix = confusion_matrix(y_test, predictions)
print("混淆矩阵:")
print(conf_matrix)

# 计算精确度
precision = precision_score(y_test, predictions)
print(f"精确度: {precision:.2f}")

# 计算召回率
recall = recall_score(y_test, predictions)
print(f"召回率: {recall:.2f}")

accuracy = accuracy_score(y_test, predictions)
print(f"准确性：{accuracy:.2f}")