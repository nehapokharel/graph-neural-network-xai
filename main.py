from dgl.data import BAShapeDataset
dataset = BAShapeDataset()


g = dataset[0]
label = g.ndata['label']

edges_from = g.edges()[0].tolist()
edges_to = g.edges()[1].tolist()

print(len(label), len(edges_to))

edge_list = [[] for i in label]

for index, item in enumerate(edges_to):
    edge_list[item].append(edges_from[index])
    edge_list[edges_from[index]].append(item)


def in_C(start, index, C_len, visited):
    global edge_list

    if (index in visited) and (C_len > 0):
        return 0

    if (C_len == 0) and (index != start):
        return 0

    if (C_len == 0) and (index == start):
        return 1

    for i in edge_list[index]:
        val = in_C(start, i, C_len-1, visited + [index])
        if val == 1:
            return 1
    return 0


def neighbour_C(index, C_len):
    global edge_list, in_C3, in_C4
    if C_len == 3:
        for i in edge_list[index]:
            if in_C3[i] == 1:
                return 1
        return 0
    else:
        for i in edge_list[index]:
            if in_C4[i] == 1:
                return 1
        return 0


neighbours = [len(i) for i in edge_list]
in_C3 = [in_C(i, i, 3, []) for i in range(len(label))]
in_C4 = [in_C(i, i, 4, []) for i in range(len(label))]

neighbour_C3 = [neighbour_C(i, 3) for i in range(len(label))]
neighbour_C4 = [neighbour_C(i, 4) for i in range(len(label))]

import pandas as pd

data = {'neighbours':  neighbours,
        'in_C3': in_C3,
        'in_C4': in_C4,
        'neighbour_C3': neighbour_C3,
        'neighbour_C4': neighbour_C4,
        'label': label,
        }

df = pd.DataFrame(data)



#print(g.edges()[0].tolist())

#print(edge_list)
#print(edge_list[89], edge_list[90])

#print(sum(in_C3), sum(in_C4))
#print(neighbour_C4)

feature_cols = ['neighbours', 'in_C3', 'in_C4', 'neighbour_C3','neighbour_C4']
X = df[feature_cols] # Features
y = df.label # Target variable

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

from sklearn.tree import DecisionTreeClassifier
clf = DecisionTreeClassifier(criterion="entropy", max_depth=5)
clf = clf.fit(X_train, y_train)

y_pred = clf.predict(X_test)

tmp = 0
y_pred = y_pred.tolist()
y_test = y_test.tolist()
for index, item in enumerate(y_pred):
    tmp += (item - y_test[index])**2

print(tmp)

#print(y_test)
#print(y_pred)
