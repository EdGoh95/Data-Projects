#!/usr/bin/env python3
"""
Deep Learning With TensorFlow and Keras Third Edition (Packt Publishing)
Chaper 17: Graph Neural Networks
"""
import os
os.environ["DGLBACKEND"] = 'tensorflow'
import dgl
import tensorflow as tf
from dgl import data, nn
from sklearn.model_selection import train_test_split

def set_gpu_if_available():
    device = '/cpu:0'
    gpus = tf.config.list_physical_devices('GPU')
    if len(gpus) > 0:
        device = gpus[0]
    return device

physical_device = '/cpu:0'

proteins_dataset = data.GINDataset('PROTEINS', self_loop = True)
print('Number of graphs in the dataset:', len(proteins_dataset))
print('Dimensionality of node features:', proteins_dataset.dim_nfeats)
print('Number of graph categories:', proteins_dataset.gclasses)

train_dataset, test_dataset = train_test_split(proteins_dataset, shuffle = True, test_size = 0.2)
training_dataset, validation_dataset = train_test_split(train_dataset, test_size = 0.1)
print('Dataset sizes: {} (Training), {} (Validation), {} (Test)'.format(
    len(training_dataset), len(validation_dataset), len(test_dataset)))

class GraphClassifier(tf.keras.Model):
    def __init__(self, input_features, hidden_features, num_classes):
        super(GraphClassifier, self).__init__()
        self.conv1 = nn.GraphConv(input_features, hidden_features, activation = tf.nn.relu)
        self.conv2 = nn.GraphConv(hidden_features, num_classes)

    def call(self, g, input_features):
        h = self.conv1(g, input_features)
        h = self.conv2(g, h)
        g.ndata['h'] = h
        return dgl.mean_nodes(g, 'h')

def evaluate(model, dataset):
    total_accuracy = 0
    total_records = 0
    indices = tf.data.Dataset.from_tensor_slices(range(len(dataset))).batch(batch_size = 16)
    for batch_indices in indices:
        graphs, labels = zip(*[dataset[i] for i in batch_indices])
        batched_graphs = dgl.batch(graphs).to(physical_device)
        batched_labels = tf.convert_to_tensor(labels, dtype = tf.int64)
        logits = model(batched_graphs, batched_graphs.ndata['attr'])
        batched_predictions = tf.math.argmax(logits, axis = 1)
        accuracy = tf.reduce_sum(tf.cast(batched_predictions == batched_labels, dtype = tf.float32))
        total_accuracy += accuracy.numpy()
        total_records += len(batched_labels)
    return total_accuracy/total_records

with tf.device(physical_device):
    proteins_model = GraphClassifier(proteins_dataset.dim_nfeats, 16, proteins_dataset.gclasses)
    optimizer = tf.keras.optimizers.Adam(learning_rate = 1e-2)
    loss_function = tf.keras.losses.SparseCategoricalCrossentropy(from_logits = True)

    training_indices = tf.data.Dataset.from_tensor_slices(range(len(training_dataset))).batch(batch_size = 16)

    for epoch in range(20):
        total_loss = 0
        for batched_indices in training_indices:
            with tf.GradientTape() as tape:
                graphs, labels = zip(*[training_dataset[j] for j in batched_indices])
                batched_graphs = dgl.batch(graphs).to(physical_device)
                batched_labels = tf.convert_to_tensor(labels, dtype = tf.int32)
                logits = proteins_model(batched_graphs, batched_graphs.ndata['attr'])
                loss = loss_function(batched_labels, logits)
                gradients = tape.gradient(loss, proteins_model.trainable_weights)
                optimizer.apply_gradients(zip(gradients, proteins_model.trainable_weights))
                total_loss += loss.numpy()

        validation_accuracy = evaluate(proteins_model, validation_dataset)
        print('Epoch {} - Training Loss: {:.5f}, Validation Accuracy: {:.2f}%'.format(
            epoch + 1, total_loss, validation_accuracy*100))

    print('Test Accuracy: {:.2f}%'.format(evaluate(proteins_model, test_dataset)*100))