#!/usr/bin/env python3
"""
Deep Learning With TensorFlow and Keras Third Edition (Packt Publishing)
Chapter 17: Graph Neural Networks
"""
import numpy as np
import matplotlib.pyplot as plt
import dgl
import tensorflow as tf
import torch

CORA_dataset = dgl.data.CoraGraphDataset()
graph = CORA_dataset[0].to('cpu')

def set_gpu_if_available():
    device = '/cpu:0'
    gpus = tf.config.list_physical_devices('GPU')
    if len(gpus) > 0:
        device = gpus[0]
    return device

physical_device = set_gpu_if_available()
print(physical_device)

class NodeClassifier(torch.nn.Module):
    def __init__(self, input_features, hidden_features, num_classes):
        super(NodeClassifier, self).__init__()
        self.conv1 = dgl.nn.GraphConv(input_features, hidden_features, activation = torch.relu)
        self.conv2 = dgl.nn.GraphConv(hidden_features, num_classes)

    def forward(self, g, input_features):
        h = self.conv1(g, input_features)
        h = self.conv2(g, h)
        return h

def evaluate(model, features, labels, mask):
    logits = model(graph, features)[mask]
    labels = labels[mask]
    predictions = logits.argmax(1)
    accuracy = torch.mean((predictions == labels).float())
    return accuracy

with tf.device(physical_device.name.replace('/physical_device:', '')):
    features = graph.ndata['feat']
    labels = graph.ndata['label']
    training_mask = graph.ndata['train_mask']
    validation_mask = graph.ndata['val_mask']
    test_mask = graph.ndata['test_mask']

    input_features = features.shape[1]
    n_classes = CORA_dataset.num_classes
    n_edges = graph.number_of_edges()

    graph_model = NodeClassifier(input_features, 16, n_classes)
    optimizer = torch.optim.AdamW(graph_model.parameters(), lr = 1e-2, weight_decay = 5e-4)

    history = []
    for epoch in range(200):
        logits = graph_model(graph, features)
        loss = torch.nn.functional.nll_loss(torch.nn.functional.log_softmax(logits[training_mask], 1),
                                            labels[training_mask])
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        validation_accuracy = evaluate(graph_model, features, labels, validation_mask)
        history.append((epoch + 1, loss.detach().numpy(), validation_accuracy))

        if (epoch + 1) % 10 == 0:
            print('Epoch {} - Training Loss: {:.5f}, Validation Accuracy: {:.2f}%'.format(
                epoch + 1, loss.detach().numpy(), validation_accuracy*100))

(epochs, losses, val_acc) = [np.array(h) for h in zip(*history)]
plt.subplot(1, 2, 1)
plt.plot(epochs, losses)
plt.xlabel('Epoch')
plt.ylabel('Training Loss')

plt.subplot(1, 2, 2)
plt.plot(epochs, val_acc*100)
plt.xlabel('Epoch')
plt.ylabel('Validation Accuracy/%')

print('Test Accuracy: {:.2f}%'.format(evaluate(graph_model, features, labels, test_mask)*100))