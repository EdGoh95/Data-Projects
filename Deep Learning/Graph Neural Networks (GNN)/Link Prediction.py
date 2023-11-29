#!/usr/bin/env python3
"""
Deep Learning With TensorFlow and Keras Third Edition (Packt Publishing)
Chapter 17: Graph Neural Networks
"""
import itertools
import numpy as np
import scipy.sparse as sp
import dgl
import torch
from sklearn.metrics import roc_auc_score

CORA_dataset = dgl.data.CoraGraphDataset()
graph = CORA_dataset[0].to('cpu')

# Positive Edges
eids = np.random.permutation(np.arange(graph.number_of_edges()))
test_size = int(len(eids) * 0.2)
validation_size = int((len(eids) - test_size) * 0.1)
training_size = graph.number_of_edges() - (test_size + validation_size)

u, v = graph.edges()
u = u.numpy()
v = v.numpy()

positive_test_u = u[eids[:test_size]]
positive_test_v = v[eids[:test_size]]
positive_validation_u = u[eids[test_size:(test_size + validation_size)]]
positive_validation_v = v[eids[test_size:(test_size + validation_size)]]
positive_training_u = u[eids[(test_size + validation_size):]]
positive_training_v = v[eids[(test_size + validation_size):]]

# Negative Edges
adjacency_matrix = sp.coo_matrix((np.ones(len(u)), (u, v)))
negative_adjacency_matrix = 1 - adjacency_matrix.todense() - np.eye(graph.number_of_nodes())
negative_u, negative_v = np.where(negative_adjacency_matrix != 0)

negative_eids = np.random.choice(len(negative_u), graph.number_of_edges())
negative_test_u = negative_u[negative_eids[:test_size]]
negative_test_v = negative_v[negative_eids[:test_size]]
negative_validation_u = negative_u[negative_eids[test_size:(test_size + validation_size)]]
negative_validation_v = negative_v[negative_eids[test_size:(test_size + validation_size)]]
negative_training_u = negative_u[negative_eids[(test_size + validation_size):]]
negative_training_v = negative_v[negative_eids[(test_size + validation_size):]]

# Remove test and validation edges from the training graph
test_edges = eids[:test_size]
validation_edges = eids[test_size:(test_size + validation_size)]
training_edges = eids[(test_size + validation_size):]
training_graph = dgl.remove_edges(graph, np.concatenate([test_edges, validation_edges]))

class LinkPredictor(torch.nn.Module):
    def __init__(self, input_features, hidden_features):
        super(LinkPredictor, self).__init__()
        self.conv1 = dgl.nn.SAGEConv(input_features, hidden_features, 'mean')
        self.relu1 = torch.relu
        self.conv2 = dgl.nn.SAGEConv(hidden_features, hidden_features, 'mean')

    def forward(self, g, input_features):
        h = self.conv1(g, input_features)
        h = self.relu1(h)
        h = self.conv2(g, h)
        return h

positive_training_graph = dgl.graph((positive_training_u, positive_training_v),
                                    num_nodes = graph.number_of_nodes())
negative_training_graph = dgl.graph((negative_training_u, negative_training_v),
                                    num_nodes = graph.number_of_nodes())

positive_validation_graph = dgl.graph((positive_validation_u, positive_validation_v),
                                      num_nodes = graph.number_of_nodes())
negative_validation_graph = dgl.graph((negative_validation_u, negative_validation_v),
                                      num_nodes = graph.number_of_nodes())

positive_test_graph = dgl.graph((positive_test_u, positive_test_v), num_nodes = graph.number_of_nodes())
negative_test_graph = dgl.graph((negative_test_u, negative_test_v), num_nodes = graph.number_of_nodes())

class DotProductPredictor(torch.nn.Module):
    def forward(self, g, h):
        with g.local_scope():
            g.ndata['h'] = h
            """
            Compute a new edge feature named 'scope' using a dot product between the source node
            feature 'h' and the destination node feature 'h'
            u_dot_v: Returns a 1-element vector for each edge hence the vector needs to be squeezed
            """
            g.apply_edges(dgl.function.u_dot_v('h', 'h', 'score'))
            return g.edata['score'][:, 0]

class MLPPredictor(torch.nn.Module):
    def __init__(self, hidden_features):
        super().__init__()
        self.W1 = torch.nn.Linear(hidden_features * 2, hidden_features, activation = torch.relu)
        self.W2 = torch.nn.Linear(hidden_features, 1)

    def apply_edges(self, edges):
        h = torch.cat([edges.src['h'], edges.dst['h']], axis = 1)
        return {'score': self.W2(self.W1(h)).squeeze(1)}

    def forward(self, g, h):
        with g.local_scope():
            g.ndata['h'] = h
            g.apply_edges(self.apply_edges)
            return g.edata['score']

link_model = LinkPredictor(training_graph.ndata['feat'].shape[1], 16)
prediction = DotProductPredictor()
optimizer = torch.optim.Adam(itertools.chain(link_model.parameters(), prediction.parameters()), lr = 1e-2)

def compute_loss(positive_score, negative_score):
    scores = torch.cat([positive_score, negative_score])
    labels = torch.cat([torch.ones(positive_score.shape[0]), torch.zeros(negative_score.shape[0])])
    return torch.nn.functional.binary_cross_entropy_with_logits(scores, labels)

def compute_auc(positive_score, negative_score):
    scores = torch.cat([positive_score, negative_score]).numpy()
    labels = torch.cat([torch.ones(positive_score.shape[0]), torch.zeros(negative_score.shape[0])]).numpy()
    return roc_auc_score(labels, scores)

for epoch in range(100):
    input_features = training_graph.ndata['feat']
    h = link_model(training_graph, input_features)
    positive_score = prediction(positive_training_graph, h)
    negative_score = prediction(negative_training_graph, h)
    loss = compute_loss(positive_score, negative_score)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    with torch.no_grad():
        positive_validation_score = prediction(positive_validation_graph, h)
        negative_validation_score = prediction(negative_validation_graph, h)
        validation_auc = compute_auc(positive_validation_score, negative_validation_score)

        if (epoch + 1) % 5 == 0:
            print('Epoch {} - Training Loss: {:.5f}, Validation AUC Score: {:.5f}'.format(
                epoch + 1, loss.detach().numpy(), validation_auc))

with torch.no_grad():
    positive_test_score = prediction(positive_test_graph, h)
    negative_test_score = prediction(negative_test_graph, h)
    print('Test AUC Score: {:.5f}'.format(compute_auc(positive_test_score, negative_test_score)))