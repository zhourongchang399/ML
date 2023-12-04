from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
from collections import Counter

# 读取数据集
dataset = pd.read_csv("dataset/drug200.csv")
dataset['Age'] = pd.to_numeric(dataset['Age'])

# 对数据集的数据作处理
dataset.insert(value=dataset['Sex'].replace(['F', 'M'], [1, 0]),
               loc=0,
               column='F')
dataset.insert(value=dataset['Sex'].replace(['F', 'M'], [0, 1]),
               loc=0,
               column='M')
dataset.insert(value=dataset['BP'].replace(['HIGH', 'LOW', 'NORMAL'],
                                           [1, 0, 0]),
               loc=0,
               column='BP_HIGH')
dataset.insert(value=dataset['BP'].replace(['HIGH', 'LOW', 'NORMAL'],
                                           [0, 1, 0]),
               loc=0,
               column='BP_LOW')
dataset.insert(value=dataset['BP'].replace(['HIGH', 'LOW', 'NORMAL'],
                                           [0, 0, 1]),
               loc=0,
               column='BP_NORMAL')
dataset.insert(value=dataset['Cholesterol'].replace(['HIGH', 'NORMAL'],
                                                    [1, 0]),
               loc=0,
               column='Cholesterol_HIGH')
dataset.insert(value=dataset['Cholesterol'].replace(['HIGH', 'NORMAL'],
                                                    [0, 1]),
               loc=0,
               column='Cholesterol_NORMAL')
dataset = dataset.replace(['drugA', 'drugB', 'drugC', 'drugX', 'DrugY'],
                          [0, 1, 2, 3, 4])
dataset.loc[dataset['Age'] < 18, 'Age'] = 0
dataset.loc[(dataset['Age'] >= 18) & (dataset['Age'] < 65), 'Age'] = 1
dataset.loc[dataset['Age'] >= 65, 'Age'] = 2
dataset.insert(value=dataset['Age'].replace([0, 1, 2], [1, 0, 0]),
               loc=0,
               column='Age_juvenile')
dataset.insert(value=dataset['Age'].replace([0, 1, 2], [0, 1, 0]),
               loc=0,
               column='Age_youth')
dataset.insert(value=dataset['Age'].replace([0, 1, 2], [0, 0, 1]),
               loc=0,
               column='Age_old')
dataset.loc[dataset['Na_to_K'] < 15, 'Na_to_K'] = 0
dataset.loc[(dataset['Na_to_K'] >= 15) & (dataset['Na_to_K'] < 30),
            'Na_to_K'] = 1
dataset.loc[dataset['Na_to_K'] >= 30, 'Na_to_K'] = 2
dataset.insert(value=dataset['Na_to_K'].replace([0, 1, 2], [1, 0, 0]),
               loc=0,
               column='Na_to_K_low')
dataset.insert(value=dataset['Na_to_K'].replace([0, 1, 2], [0, 1, 0]),
               loc=0,
               column='Na_to_K_mid')
dataset.insert(value=dataset['Na_to_K'].replace([0, 1, 2], [0, 0, 1]),
               loc=0,
               column='Na_to_K_hight')
dataset = dataset.drop('Sex', axis=1)
dataset = dataset.drop('BP', axis=1)
dataset = dataset.drop('Cholesterol', axis=1)
dataset = dataset.drop('Age', axis=1)
dataset = dataset.drop('Na_to_K', axis=1)
# print(dataset)
X = dataset.iloc[:, :13]
Y = dataset.iloc[:, -1]
x_train, x_test, y_train, y_test = train_test_split(X,
                                                    Y,
                                                    test_size=0.2,
                                                    random_state=21)

# print(x_train, y_train)


class Navia_Bayes:

    def __init__(self) -> None:
        self.label_num = 0
        self.probability = []

    def fit(self, X, y):
        # 分类类别
        self.label_num = len(np.unique(y))
        # 特征数量
        self.feature_num = X.shape[1]
        # 每个label的数量
        each_label = Counter(y)
        #每个类别出现的概率
        self.each_probability = np.ones((self.label_num, self.feature_num))
        y = np.array(y)
        data = np.hstack((X, y.reshape(X.shape[0], 1)))
        # 计算每个类别各个特征出现的概率
        for i in range(self.label_num):
            self.probability.append(each_label[i] / len(y))
            x = data[data[:, -1] == i, 0:self.feature_num]
            molec = np.sum(x, axis=0)
            for j in range(self.feature_num):
                self.each_probability[i][j] = molec[j] / each_label[i]

    def predict(self, X):
        X = np.array(X)
        self.sample_num = X.shape[0]
        self.result = []
        # 分别计算该例子的每个类别可能的概率
        for x in X:
            probability = []
            for i in range(self.label_num):
                temp = self.probability[i]
                for j in range(self.feature_num):
                    if x[j] == 1:
                        temp *= self.each_probability[i][j]
                probability.append(temp)
            self.result.append(np.argsort(probability)[-1])
        return self.result

    def accuracy(self, y):
        y = np.array(y)
        error = 0
        for i in range(self.sample_num):
            if result[i] != y[i]:
                error += 1
        acc = 1 - (error / self.sample_num)
        return acc


# x_train = np.array([[1, 0, 1], [0, 1, 0], [1, 1, 1], [0, 0, 0], [1, 1, 0],
#                     [0, 0, 0], [1, 1, 1]])
# y_train = np.array([1, 0, 1, 0, 1, 1, 0])
# x_test = np.array([[1,0,0,1,0,0,0,1,1,0]])
# y_test = np.array([4])

model = Navia_Bayes()
model.fit(x_train, y_train)
result = model.predict(x_test)
acc = model.accuracy(y_test)
print(f"Accuracy : {acc}")