import numpy as np
import pandas as pd

dataset = pd.read_csv("dataset/house-votes-84.data", header=None)
columns = [
    '党派', '残疾人婴幼儿法案', '水项目费用分摊', '预算决议案', '医生费用冻结议案', '萨尔瓦多援助', '校园宗教团体决议',
    '反卫星禁试决议', '援助尼加拉瓜反政府', 'MX导弹议案', '移民决议案', '合成燃料公司削减决议', '教育支出决议',
    '超级基金起诉权', '犯罪决议案', '免税出口决议案', '南非出口管理决议案'
]
dataset.columns = columns
dataset.to_csv("data.csv")

for col in columns:
    dataset[col] = dataset[col].apply(lambda x: col + ":" + x)

def create_C1(dataset):
    C1 = []
    for transaction in dataset:
        for item in transaction:
            if not [item] in C1:
                C1.append([item])
    C1.sort()
    return list(map(frozenset, C1))

def scan_D(D, Ck, min_support):
    ss_cnt = {}
    for tid in D:
        for can in Ck:
            if can.issubset(tid):
                if not can in ss_cnt:
                    ss_cnt[can] = 1
                else:
                    ss_cnt[can] += 1
    # print(ss_cnt)
    num_items = float(len(D))
    ret_list = []
    support_data = {}
    for key in ss_cnt:
        support = ss_cnt[key] / num_items
        if support >= min_support:
            ret_list.insert(0, key)
        support_data[key] = support
    # print(ret_list, support_data)
    return ret_list, support_data


def apriori_gen(Lk, k):
    ret_list = []
    len_Lk = len(Lk)
    for i in range(len_Lk):
        for j in range(i + 1, len_Lk):
            L1 = list(Lk[i])[:k - 2]
            L2 = list(Lk[j])[:k - 2]
            print(L1,L2)
            L1.sort()
            L2.sort()
            if L1 == L2:
                ret_list.append(Lk[i] | Lk[j])
    return ret_list


def apriori(dataset, min_support=0.5):
    C1 = create_C1(dataset)
    D = list(map(set, dataset))
    # print(D)
    L1, support_data = scan_D(D, C1, min_support)
    L = [L1]
    k = 2
    while not len(L[k - 2]) == 0:
        Ck = apriori_gen(L[k - 2], k)
        Lk, supK = scan_D(D, Ck, min_support)
        support_data.update(supK)
        L.append(Lk)
        k += 1
    return L, support_data

def association_rules(L,support_data,min_confident):
    result = []
    k = 2
    j = 0
    while not len(L[k-1]) == 0:
        for l in L[k-1]:
            temp = list(l)
            for i in temp:
                antecedents = frozenset({i})
                consequent = l - antecedents
                j+=1
                confident = support_data[l]/support_data[antecedents]
                if confident >= min_confident:
                    result.append([antecedents,consequent,support_data[antecedents],support_data[consequent],support_data[l],confident])
                    # print(f"{antecedents}->{consequent}:antecedents_support:{support_data[antecedents]},consequent_support:{support_data[consequent]}")
        k+=1
    return result

dataset = np.array(dataset)
L,support_data = apriori(dataset)
print(L)

result = association_rules(L,support_data,0.9)
columns = ['antecedents','consequent','antecedents_support','consequent_support','support','confident']
result = pd.DataFrame(result,columns=columns)
print(result)
result.to_csv("dataset/apriori_result.csv")