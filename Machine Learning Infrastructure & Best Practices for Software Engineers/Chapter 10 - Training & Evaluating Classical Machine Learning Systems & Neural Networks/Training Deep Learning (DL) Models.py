#!/usr/bin/env python3
"""
Machine Learning Infrastructure And Best Practices For Software Engineers (Packt Publishing)
Chapter 10: Training and Evaluating Classical Machine Learning Systems and Neural Networks
"""
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score
from termcolor import colored

# Defining The Neural Network Architecture
class NeuralNetwork(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(NeuralNetwork, self).__init__()
        self.FC1 = nn.Linear(input_size, hidden_size)
        self.ReLU = nn.ReLU()
        self.FC2 = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        output = self.FC1(x)
        output = self.ReLU(output)
        output = self.FC2(output)
        return output

camel_df = pd.read_excel(
    "../Chapter 6 - Processing Data In Machine Learning Systems/Regenerated PROMISE and BPD Datasets.xlsx",
    sheet_name = 'camel_1_2', index_col = 0)
camel_features = camel_df.drop(['Defect'], axis = 1)
camel_target = camel_df['Defect']
camel_features_train, camel_features_test, camel_target_train, camel_target_test = train_test_split(
    camel_features, camel_target, train_size = 0.9, random_state = 42)

neural_network_model = NeuralNetwork(camel_features_train.shape[1], 96, 2)
# Converting The Training & Test Datasets To PyTorch Tensors
camel_features_train_tensor = torch.Tensor(camel_features_train.values)
camel_target_train_tensor = torch.LongTensor(camel_target_train.values)

batch_size = 32
loss_function = nn.CrossEntropyLoss()
optimizer = optim.Adam(neural_network_model.parameters(), lr = 1e-4)
for epoch in range(10000):
    for i in range(0, len(camel_features_train_tensor), batch_size):
        features_batch = camel_features_train_tensor[i:i+batch_size]
        target_batch = camel_target_train_tensor[i:i+batch_size]

        # Forward Pass
        outputs = neural_network_model(features_batch)
        loss = loss_function(outputs, target_batch)

        # Backward Pass + Optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    if (epoch % 100 == 0):
        print("Epoch {:d}/10000 - Loss: {:.3f}".format(epoch, loss.item()))

with torch.no_grad():
    neural_network_model.eval() # Set the neural network model to evaluation mode
    camel_features_test_tensor = torch.Tensor(camel_features_test.values)
    test_outputs = neural_network_model(camel_features_test_tensor)
    _, camel_predictions = torch.max(test_outputs.data, 1)
    camel_nn_predictions = camel_predictions.numpy()

print(colored('\ncamel_1_3 (Neural Network)', 'cyan', attrs = ['bold']))
print('Test Accuracy: {:.3f}'.format(accuracy_score(camel_target_test, camel_nn_predictions)),
      sep = '\n')
print('Test Precision: {:.3f}'.format(
    precision_score(camel_target_test, camel_nn_predictions, average = 'weighted')), sep = '\n')
print('Test Recall: {:.3f}'.format(
    recall_score(camel_target_test, camel_nn_predictions, average = 'weighted')), sep = '\n')