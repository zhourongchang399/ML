import numpy as np
from sklearn.datasets import load_iris,load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import networkx as nx


def plot_tree(node,
              parent_name='root',
              graph=None,
              pos=None,
              level=1,
              feature_names=None):
    if graph is None:
        graph = nx.DiGraph()
        plt.title("Decision Tree")
        pos = dict()

    pos[parent_name] = (level, -node.depth)

    if node.is_leaf:
        graph.add_node(parent_name, pos=pos[parent_name], label=node.label)
    else:
        feature_name = feature_names[node.feature_index] + f" {node.threshold}"
        graph.add_node(parent_name, pos=pos[parent_name], label=feature_name)
        if node.left is not None:
            child_name = f"{parent_name+feature_name}_L"
            graph.add_edge(parent_name, child_name)
            plot_tree(node.left,
                      child_name,
                      graph=graph,
                      pos=pos,
                      level=level + 1,
                      feature_names=feature_names)

        if node.right is not None:
            child_name = f"{parent_name+feature_name}_R"
            graph.add_edge(parent_name, child_name)
            plot_tree(node.right,
                      child_name,
                      graph=graph,
                      pos=pos,
                      level=level + 1,
                      feature_names=feature_names)

    return graph


# 可视化决策树
def visualize_tree(tree_root, feature_names):
    graph = plot_tree(tree_root, feature_names=feature_names)
    pos = nx.spring_layout(graph, seed=42)  # 使用 spring layout 布局
    nx.draw(graph,
            pos,
            with_labels=True,
            font_weight='bold',
            node_size=3000,
            node_color='skyblue',
            font_size=8)
    plt.show()

#节点
class Node:

    def __init__(self, data, depth) -> None:
        self.data = data
        self.depth = depth
        self.left = None
        self.right = None
        self.feature_index = None
        self.threshold = None
        self.is_leaf = False
        self.label = None

#计算熵
def entropy(y):
    classes, counts = np.unique(y, return_counts=True)
    probablities = counts / len(y)
    return -np.sum(probablities * np.log(probablities))


def information_gain(X, y, feature_index, threshold):
    #父节点的熵
    parent_entropy = entropy(y)

    #左右节点
    left_mask = X[:, feature_index] <= threshold
    right_mask = ~left_mask

    if np.sum(left_mask) == 0 or np.sum(right_mask) == 0:
        return 0

    left_child_entropy = entropy(y[left_mask])
    right_child_entropy = entropy(y[right_mask])

    left_weight = np.sum(left_mask) / len(y)
    right_weight = np.sum(right_mask) / len(y)

    child_entropy = left_weight * left_child_entropy + right_weight * right_child_entropy

    #信息增益
    return parent_entropy - child_entropy

# 查找信息增益最大的划分节点
def find_best_split(X, y, num_features):
    best_feature = None
    best_threshold = None
    max_info_gain = -float('inf')

    for feature_index in range(num_features):
        thresholds = np.unique(X[:, feature_index])
        for threshold in thresholds:
            info_gain = information_gain(X, y, feature_index, threshold)
            if info_gain > max_info_gain:
                max_info_gain = info_gain
                best_feature = feature_index
                best_threshold = threshold

    return best_feature, best_threshold


# 递归构建决策树
def build_tree(X, y, depth, max_depth):
    #創建節點
    node = Node(data=(X, y), depth=depth)

    # 判断是否终止
    if depth == max_depth or len(np.unique(y)) == 1:
        node.is_leaf = True
        node.label = np.argmax(np.bincount(y))
    else:
        num_features = X.shape[1]
        best_feature, best_threshold = find_best_split(X, y, num_features)
        left_mask = X[:, best_feature] <= best_threshold
        right_mask = ~left_mask
        node.feature_index = best_feature
        node.threshold = best_threshold
        node.left = build_tree(X[left_mask], y[left_mask], depth + 1,
                               max_depth)
        node.right = build_tree(X[right_mask], y[right_mask], depth + 1,
                                max_depth)

    return node


def predict(node, X):
    predictions = []
    for x in X:
        predictions.append(predict_tree(node, x))
    return predictions


def predict_tree(node, X):
    # 递归预测
    if node.is_leaf:
        return node.label
    elif X[node.feature_index] <= node.threshold:
        return predict_tree(node.left, X)
    else:
        return predict_tree(node.right, X)


data = load_breast_cancer()
x_train, x_test, y_train, y_test = train_test_split(data['data'],
                                                    data['target'],
                                                    test_size=0.3)

# 构建决策树
tree_root = build_tree(x_train, y_train, depth=0, max_depth=3)

# 预测
predictions = predict(tree_root, x_test)
accuracy = accuracy_score(y_test, predictions)
print(f"准确性：{accuracy:.2f}")

# 可视化决策树
feature_names = [f"{i}" for i in data.feature_names]
visualize_tree(tree_root, feature_names)