#!/usr/bin/env python3
"""
Deep Learning With TensorFlow and Keras Third Edition (Packt Publishing)
Chapter 17: Graph Neural Networks
"""
import numpy as np
import matplotlib.pyplot as plt
import dgl
import networkx
import torch
from networkx.exception import NetworkXError

#%% Single Graph
class KarateClubDataset(dgl.data.DGLDataset):
    def __init__(self):
        super().__init__(name = 'karate_club')

    def __getitem__(self, i):
        return self.graph

    def __len__(self):
        return 1

    def process(self):
        Graph = networkx.karate_club_graph()
        nodes = [node for node in Graph.nodes]
        edges = [edge for edge in Graph.edges]
        node_features = torch.rand((len(nodes), 10), dtype = torch.float32)
        label2int = {'Mr. Hi': 0, 'Officer': 1}
        node_labels = torch.as_tensor([label2int[Graph.nodes[node]['club']] for node in nodes],
                                      dtype = torch.int32)
        edge_features = torch.randint(low = 3, high = 10, size = (len(edges), 1), dtype = torch.int32)
        edges_src = [u for u, v in edges]
        edges_dest = [v for u, v in edges]

        self.graph = dgl.graph((edges_src, edges_dest), num_nodes = len(nodes))
        self.graph.ndata['feat'] = node_features
        self.graph.ndata['label'] = node_labels
        self.graph.edata['weight'] = edge_features

        # Assign masks to indicate the split (training, validation, test)
        n_nodes = len(nodes)
        n_train = int(n_nodes * 0.6)
        n_validation = int(n_nodes * 0.2)
        training_mask = torch.as_tensor(np.hstack([np.ones(n_train), np.zeros(n_nodes - n_train)]),
                                        dtype = torch.bool)
        validation_mask = torch.as_tensor(np.hstack([np.zeros(n_train), np.ones(n_validation),
                                                     np.zeros(n_nodes - n_train - n_validation)]),
                                          dtype = torch.bool)
        test_mask = torch.as_tensor(np.hstack([np.zeros(n_train + n_validation),
                                               np.ones(n_nodes - n_train - n_validation)]),
                                    dtype = torch.bool)

        self.graph.ndata['train_mask'] = training_mask
        self.graph.ndata['val_mask'] = validation_mask
        self.graph.ndata['test_mask'] = test_mask

karate_club_dataset = KarateClubDataset()
karate_club_graph = karate_club_dataset[0]
print('Karate Club Dataset (Single Graph):', karate_club_graph, sep = '\n')

#%% A Set Of Multiple Graphs
class SyntheticDataset(dgl.data.DGLDataset):
    def __init__(self):
        super().__init__(name = 'synthetic')

    def __getitem__(self, i):
        return self.graphs[i], self.labels[i]

    def __len__(self):
        return len(self.graphs)

    def process(self):
        self.graphs = []
        self.labels = []
        num_graphs = 0
        while True:
            d = np.random.randint(3, 10)
            n = np.random.randint(5, 10)
            if ((n * d) % 2) != 0:
                continue
            if n < d:
                continue
            try:
                g = networkx.random_regular_graph(d, n)
            except NetworkXError:
                continue

            graph_edges = [edge for edge in g.edges]
            graph_src = [u for u, v in graph_edges]
            graph_dest = [v for u, v in graph_edges]
            graph_num_nodes = len(g.nodes)
            label = np.random.randint(0, 2)
            # Create a graph and add it to the list of graphs and labels
            dgl_graph = dgl.graph((graph_src, graph_dest), num_nodes = graph_num_nodes)
            dgl_graph.ndata['feats'] = torch.rand((graph_num_nodes, 10), dtype = torch.float32)

            self.graphs.append(dgl_graph)
            self.labels.append(label)
            num_graphs += 1
            if num_graphs > 100:
                break

        self.labels = torch.as_tensor(self.labels, dtype = torch.int64)

molecules_dataset = SyntheticDataset()
molecules_graphs, molecules_label = molecules_dataset[0]
print('\nMolecules Dataset (A Set of Multiple Graphs):', molecules_graphs, sep = '\n')
print('Label:', molecules_label)