import dgl
import torch
import pandas as pd
from dgl.nn import GraphConv
import torch.nn.functional as F
from dgl.data import BAShapeDataset
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn import tree
from matplotlib import pyplot as plt
from sklearn.metrics import accuracy_score

dataset = BAShapeDataset()


# Define the Graph Convolutional Network (GCN) model
class GCN(torch.nn.Module):
    def __init__(self, in_feats, hidden_size, num_classes):
        super(GCN, self).__init__()
        self.conv1 = GraphConv(in_feats, hidden_size)
        self.conv2 = GraphConv(hidden_size, num_classes)

    def forward(self, g, features):
        x = self.conv1(g, features)
        x = F.relu(x)
        x = self.conv2(g, x)
        return x


def in_C(start, index, C_len, visited, edge_list):
    """
    Recursive function to check if a node is part of a cycle.

    Args:
        start (int): The starting node index.
        index (int): The current node index.
        C_len (int): The length of the cycle to be found.
        visited (list): List of visited nodes.
        edge_list (list): List of edges.

    Returns:
        int: 1 if the node is part of a cycle, 0 otherwise.
    """
    if (index in visited) and (C_len > 0):
        return 0
    if (C_len == 0) and (index != start):
        return 0
    if (C_len == 0) and (index == start):
        return 1
    for i in edge_list[index]:
        val = in_C(start, i, C_len-1, visited[:] + [index], edge_list[:])
        if val == 1:
            return 1
    return 0


def neighbour_C(index, C_len, edge_list, in_C3, in_C4):
    """
    Function to check if a node has neighbors in a specific cycle length.

    Args:
        index (int): The node index.
        C_len (int): The length of the cycle.
        edge_list (list): List of edges.
        in_C3 (list): List indicating whether
        each node is part of a length 3 cycle.
        in_C4 (list): List indicating whether
        each node is part of a length 4 cycle.

    Returns:
        int: 1 if the node has neighbors in
        the specified cycle length, 0 otherwise.
    """
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


def main():
    # Get the BAShapeDataset
    g = dataset[0]
    # Retrieve the 'label' attribute for each node in the graph
    label = g.ndata['label']

    # Retrieve the source nodes of all edges in the graph
    edges_from = g.edges()[0].tolist()
    # Retrieve the destination nodes of all edges in the graph
    edges_to = g.edges()[1].tolist()
    # Initialize an empty list for each node in the graph
    edge_list = [[] for i in label]

    # Build an adjacency list representation of the graph
    for index, item in enumerate(edges_to):
        # Add the source node to the destination node's adjacency list
        edge_list[item].append(edges_from[index])
        # Add the destination node to the source node's adjacency list
        edge_list[edges_from[index]].append(item)

    # Compute the number of neighbors for each node
    neighbours = [len(i) for i in edge_list]

    # Compute whether each node is in a C3 subgraph
    in_C3 = [in_C(i, i, 3, [], edge_list) for i in range(len(label))]
    # Compute whether each node is in a C4 subgraph
    in_C4 = [in_C(i, i, 4, [], edge_list) for i in range(len(label))]

    # Compute the number of neighbors within a C3 subgraph for each node
    neighbour_C3 = \
        [neighbour_C(i, 3, edge_list, in_C3, in_C4) for i in range(len(label))]

    # Compute the number of neighbors within a C4 subgraph for each node
    neighbour_C4 = \
        [neighbour_C(i, 4, edge_list, in_C3, in_C4) for i in range(len(label))]

    data = {
        'neighbours': neighbours,
        'in_C3': in_C3,
        'in_C4': in_C4,
        'neighbour_C3': neighbour_C3,
        'neighbour_C4': neighbour_C4,
        'label': label,
    }
    df = pd.DataFrame(data)

    # Create the DGL graph
    G = dgl.add_self_loop(g)  # Add self-loops

    # Node features
    x = torch.tensor(g.ndata['label'].tolist(), dtype=torch.float).view(-1, 1)

    # Node labels
    y = torch.tensor(df.label.tolist(), dtype=torch.long)

    # Number of classes
    num_classes = torch.unique(y).size(0)

    # Get your indices from train_test_split
    train_indices, test_indices = \
        train_test_split(list(range(G.number_of_nodes())),
                         test_size=0.25, random_state=42)

    # Initialize the GCN model, optimizer, and loss function
    model = GCN(x.size(1), 16, num_classes)
    optimizer = torch.optim.Adam(model.parameters(),
                                 lr=0.01, weight_decay=5e-4)
    criterion = torch.nn.CrossEntropyLoss()

    def explainableModel(data_indices, plot=False):
        # get predictions of GNN on training data
        logits = model(G, x)
        _, indices = torch.max(logits[train_indices], dim=1)
        indices = indices.tolist()

        # decision tree on predictions of GNN (explaiable model)
        cn = ["BA graph", "middle", "bottom", "top"]
        feature_cols = ['neighbours', 'in_C3', 'in_C4',
                        'neighbour_C3', 'neighbour_C4']
        clf = DecisionTreeClassifier(criterion="entropy", max_depth=10)
        clf = clf.fit(df[feature_cols].iloc[train_indices], indices)

        if plot:
            plt.figure(dpi=200)
            tree.plot_tree(clf,
                           feature_names=feature_cols,
                           class_names=cn,
                           filled=True)
            plt.show()

        # data to predict
        _, indices = torch.max(logits[data_indices], dim=1)
        indices = indices.tolist()

        # predict forecast of GNN
        y_pred = clf.predict(df[feature_cols].iloc[data_indices])
        y_pred = y_pred.tolist()
        return accuracy_score(y_pred, indices)

    def train():
        model.train()
        logits = model(G, x)
        # compute loss
        loss = criterion(logits[train_indices], y[train_indices])
        # compute gradients
        optimizer.zero_grad()
        loss.backward()
        # update parameters
        optimizer.step()
        return loss.item()

    def test():
        model.eval()
        with torch.no_grad():
            logits = model(G, x)
            _, indices = torch.max(logits[test_indices], dim=1)
            correct = torch.sum(indices == y[test_indices])
            return correct.item() * 1.0 / len(test_indices)

    # Training loop
    for epoch in range(201):
        loss = train()
        if epoch % 10 == 0:
            acc = test()
            if epoch in [10, 100, 200]:
                acc_train_exp = explainableModel(train_indices, plot=True)
            else:
                acc_train_exp = explainableModel(train_indices, plot=False)
            acc_test_exp = explainableModel(test_indices)
            print(f'GNN         Epoch: {epoch}, '
                  f'Loss: {loss:.4f}, Accuracy: {acc:.4f}')
            print(f'Exp Model   Accuracy training: {acc_train_exp:.4f},'
                  f' Accuracy test: {acc_test_exp:.4f}')


if __name__ == "__main__":
    main()
