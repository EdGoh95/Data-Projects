#!/usr/bin/env python3
"""
Deep Learning With TensorFlow and Keras Third Edition (Packt Publishing)
Chapter 17: Graph Neural Networks
"""
import dgl
import torch

CORA_dataset = dgl.data.CoraGraphDataset()
graph = CORA_dataset[0].to('cpu')

class CustomGraphSAGE(torch.nn.Module):
    def __init__(self, input_features, output_features):
        super(CustomGraphSAGE, self).__init__()
        # A linear sub-module tfor projecting the input and its neighbour feature onto the output
        self.linear = torch.nn.Linear(input_features * 2, output_features)

    def forward(self, g, h):
        with g.local_scope():
            g.ndata['h'] = h
            g.update_all(message_func = dgl.function.copy_u('h', 'm'),
                         reduce_func = dgl.function.mean('m', 'h_N'))
            h_N = g.ndata['h_N']
            h_total = torch.cat([h, h_N], dim = 1)
            return self.linear(h_total)

class CustomGNN(torch.nn.Module):
    def __init__(self, input_features, hidden_features, num_classes):
        super(CustomGNN, self).__init__()
        self.conv1 = CustomGraphSAGE(input_features, hidden_features)
        self.relu1 = torch.relu
        self.conv2 = CustomGraphSAGE(hidden_features, num_classes)

    def forward(self, g, input_features):
        h = self.conv1(g, input_features)
        h = self.relu1(h)
        h = self.conv2(g, h)
        return h

class CustomWeightedGraphSAGE(torch.nn.Module):
    def __init__(self, input_features, output_features):
        super(CustomWeightedGraphSAGE, self).__init__()
        # A linear sub-module tfor projecting the input and its neighbour feature onto the output
        self.linear = torch.nn.Linear(input_features * 2, output_features)

    def forward(self, g, h, w):
        with g.local_scope():
            g.ndata['h'] = h
            g.edata['w'] = w
            g.update_all(message_func = dgl.function.u_mul_e('h', 'w', 'm'),
                         reduce_func = dgl.function.mean('m', 'h_N'))
            h_N = g.ndata['h_N']
            h_total = torch.cat([h, h_N], dim = 1)
            return self.linear(h_total)

class CustomWeightedGNN(torch.nn.Module):
    def __init__(self, input_features, hidden_features, num_classes):
        super(CustomWeightedGNN, self).__init__()
        self.conv1 = CustomWeightedGraphSAGE(input_features, hidden_features)
        self.relu1 = torch.relu
        self.conv2 = CustomWeightedGraphSAGE(hidden_features, num_classes)

    def forward(self, g, input_features, edge_weights):
        h = self.conv1(g, input_features, edge_weights)
        h = self.relu1(h)
        h = self.conv2(g, h, edge_weights)
        return h

graph.edata['w'] = torch.randint(low = 3, high = 10, size = (graph.num_edges(), 1)).float()

def evaluate(model, features, labels, mask, edge_weights = None):
    if edge_weights is None:
        logits = model(graph, features)[mask]
    else:
        logits = model(graph, features, edge_weights)[mask]
    labels = labels[mask]
    predictions = logits.argmax(1)
    accuracy = torch.mean((predictions == labels).float())
    return accuracy

def train(g, model, optimizer, num_epochs, use_edge_weights = False):
    features = g.ndata['feat']
    labels = g.ndata['label']
    if use_edge_weights:
        edge_weights = g.edata['w']

    training_mask = g.ndata['train_mask']
    validation_mask = g.ndata['val_mask']
    test_mask = g.ndata['test_mask']

    for epoch in range(num_epochs):
        if not use_edge_weights:
            logits = model(g, features)
        else:
            logits = model(g, features, edge_weights)
        loss = torch.nn.functional.nll_loss(torch.nn.functional.log_softmax(logits[training_mask], 1),
                                            labels[training_mask])
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if not use_edge_weights:
            validation_accuracy = evaluate(model, features, labels, validation_mask)
        else:
            validation_accuracy = evaluate(model, features, labels, validation_mask,
                                           edge_weights = edge_weights)

        if (epoch + 1) % 10 == 0:
            print('Epoch {} - Training Loss: {:.5f}, Validation Accuracy: {:.2f}%'.format(
                epoch + 1, loss.detach().numpy(), validation_accuracy*100))

    if not use_edge_weights:
        test_accuracy = evaluate(model, features, labels, test_mask)
    else:
        test_accuracy = evaluate(model, features, labels, test_mask, edge_weights = edge_weights)
    print('Test Accuracy: {:.2f}%'.format(test_accuracy*100))

print('\nWithout Edge Weights:')
custom_model = CustomGNN(graph.ndata['feat'].shape[1], 16, CORA_dataset.num_classes)
custom_optimizer = torch.optim.Adam(custom_model.parameters(), lr = 1e-2)
train(graph, custom_model, custom_optimizer, 200, use_edge_weights = False)

print('\nWith Edge Weights:')
custom_weighted_model = CustomWeightedGNN(graph.ndata['feat'].shape[1], 16, CORA_dataset.num_classes)
custom_weighted_optimizer = torch.optim.Adam(custom_weighted_model.parameters(), lr = 1e-2)
train(graph, custom_weighted_model, custom_weighted_optimizer, 200, use_edge_weights = True)