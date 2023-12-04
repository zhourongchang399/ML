import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

pd.set_option("display.max_columns", None)  # 设置显示完整的列
pd.set_option("display.max_rows", None)  # 设置显示完整的行
pd.set_option("display.expand_frame_repr", False)  # 设置不折叠数据
pd.set_option("display.max_colwidth", 100)  # 设置列的最大宽度

dataset = pd.read_csv("dataset/house-votes-84.data", header=None)
columns = [
    '党派', '残疾人婴幼儿法案', '水项目费用分摊', '预算决议案', '医生费用冻结议案', '萨尔瓦多援助', '校园宗教团体决议',
    '反卫星禁试决议', '援助尼加拉瓜反政府', 'MX导弹议案', '移民决议案', '合成燃料公司削减决议', '教育支出决议',
    '超级基金起诉权', '犯罪决议案', '免税出口决议案', '南非出口管理决议案'
]
dataset.columns = columns

for col in columns:
    dataset[col] = dataset[col].apply(lambda x: col + ":" + x)


class Apriori:

    def __init__(self, data, min_support) -> None:
        dataset = np.array(data)
        self.min_support = min_support
        self.data_num = dataset.shape[0]
        D = list(map(set, dataset))
        C, C_support, L1 = self.get_C_Support(dataset)
        L = [L1]
        k = 2
        while len(L[k - 2]) != 0:
            L_new = []
            Lk = L[k - 2]
            for i in range(len(Lk)):
                for j in range(i + 1, len(Lk)):
                    L1 = list(Lk[i])[:k - 2]
                    L2 = list(Lk[j])[:k - 2]
                    L1.sort()
                    L2.sort()
                    if L1 == L2:
                        L_new.append(Lk[i] | Lk[j])
            support = {}
            l = []
            for i in L_new:
                for j in D:
                    if i.issubset(j):
                        if i not in support:
                            support[i] = 1
                        else:
                            support[i] += 1
            for key, value in support.items():
                support[key] = value / self.data_num
                if support[key] >= min_support:
                    l.insert(0, key)
            L.append(l)
            C_support.update(support)
            k += 1
        self.L = L
        self.C_support = C_support

    def get_C_Support(self, dataset):
        C = []
        C_support = {}
        L = []
        for d in dataset:
            for i in d:
                if [i] not in C:
                    C.append([i])
                temp = frozenset([i])
                if temp not in C_support:
                    C_support[temp] = 1
                else:
                    C_support[temp] += 1
        for key, value in C_support.items():
            C_support[key] = value / self.data_num
            if C_support[key] >= self.min_support:
                L.insert(0, key)
        return C,C_support, L

    def association_rules(self, min_confident):
        support_data = self.C_support
        L = self.L
        result = []
        k = 2
        j = 0
        while not len(L[k - 1]) == 0:
            for l in L[k - 1]:
                temp = list(l)
                for i in temp:
                    antecedents = frozenset({i})
                    consequent = l - antecedents
                    j += 1
                    confident = support_data[l] / support_data[antecedents]
                    if confident >= min_confident:
                        result.append([
                            antecedents, consequent, support_data[antecedents],
                            support_data[consequent], support_data[l],
                            confident
                        ])
                        # print(f"{antecedents}->{consequent}:antecedents_support:{support_data[antecedents]},consequent_support:{support_data[consequent]}")
            k += 1
        return result


apriori = Apriori(dataset, 0.5)
result = apriori.association_rules(0.5)

columns = ['antecedents','consequent','antecedents_support','consequent_support','support','confident']
result = pd.DataFrame(result,columns=columns)
print(result)
result.to_csv("dataset/apriori_result.csv")