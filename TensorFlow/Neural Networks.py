#!/usr/bin/env python3
"""
TensorFlow Machine Learning Cookbook Second Edition (Packt Publishing) Chapter 6:
Neural Networks
"""
import time
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
from termcolor import colored
from tensorflow.python.framework import ops
from sklearn.datasets import load_iris

start = time.perf_counter()
#%% One-Layer Neural Network
ops.reset_default_graph()
iris = load_iris()
iris_features = np.array([x[0:3] for x in iris.data])
petal_width = np.array([x[3] for x in iris.data])
one_layer_session = tf.Session()

tf.set_random_seed(2)
np.random.seed(2)

# Split the dataset into training (80%) and testing (20%) sets
train_indices = np.random.choice(len(iris_features), round(len(iris_features) * 0.8),
                                 replace = False)
test_indices = np.array(list(set(range(len(iris_features))) - set(train_indices)))
train_features = iris_features[train_indices]
test_features = iris_features[test_indices]
train_petal_width = petal_width[train_indices]
test_petal_width = petal_width[test_indices]

def normalize_columns(m):
    column_min = m.min(axis = 0)
    column_max = m.max(axis = 0)
    return (m - column_min)/(column_max - column_min)

train_features = np.nan_to_num(normalize_columns(train_features))
test_features = np.nan_to_num(normalize_columns(test_features))

features = tf.placeholder(shape = [None, 3], dtype = tf.float32)
target = tf.placeholder(shape = [None, 1], dtype = tf.float32)

A1 = tf.Variable(tf.random_normal(shape = [3, 10]))
b1 = tf.Variable(tf.random_normal(shape = [10]))
A2 = tf.Variable(tf.random_normal(shape = [10, 1]))
b2 = tf.Variable(tf.random_normal(shape = [1]))

hidden_layer_output = tf.nn.relu(tf.add(tf.matmul(features, A1), b1))
final_output = tf.nn.relu(tf.add(tf.matmul(hidden_layer_output, A2), b2))
# Using MSE as the loss function
one_layer_loss = tf.reduce_mean(tf.square(target - final_output))

one_layer_optimizer = tf.train.GradientDescentOptimizer(0.005)
one_layer_train_step = one_layer_optimizer.minimize(one_layer_loss)
init = tf.global_variables_initializer()
one_layer_session.run(init)

loss = []
test_loss = []
print(colored('\u2500'*32 + ' Single Hidden Layer Neural Network ' + '\u2500'*32, 'green',
              attrs = ['bold']))
for i in range(500):
    rand_index = np.random.choice(len(train_features), size = 50)
    rand_x = train_features[rand_index]
    rand_y = np.transpose([train_petal_width[rand_index]])
    one_layer_session.run(one_layer_train_step, feed_dict = {features: rand_x,
                                                             target: rand_y})
    temp_loss = one_layer_session.run(one_layer_loss, feed_dict = {features: rand_x,
                                                                   target: rand_y})
    loss.append(np.sqrt(temp_loss))
    test_temp_loss = one_layer_session.run(
        one_layer_loss, feed_dict = {features: test_features,
                                     target: np.transpose([test_petal_width])})
    test_loss.append(np.sqrt(test_temp_loss))
    if (i + 1) % 50 == 0:
        print('Iteration {}: Loss = {:.5f}'.format(i+1, temp_loss))

plt.figure()
plt.plot(loss, 'k-', label = 'Training Loss')
plt.plot(test_loss, 'r--', label = 'Testing Loss')
plt.title('L2 Loss Variations Across Iterations (Single Hidden Layer Neural Network)')
plt.xlabel('Iteration')
plt.ylabel('Loss')
plt.legend(loc = 'upper right')

#%% Different Layers Within A Neural Network
#### Implementation on 1-Dimensional Data
ops.reset_default_graph()
session_1D = tf.Session()
tf.set_random_seed(13)
np.random.seed(13)
data = np.random.normal(size = 25)
x = tf.placeholder(dtype = tf.float32, shape = [25])

def conv_layer_1D(input_1D, random_filter):
    '''
    Creates a convolutional layer by extending the input dimensionality from 1 dimension
    to 4 dimensions and using a random filter
    '''
    input_2D = tf.expand_dims(input_1D, 0)
    input_3D = tf.expand_dims(input_2D, 0)
    input_4D = tf.expand_dims(input_3D, 3)
    convolution_output = tf.nn.conv2d(input_4D, filter = random_filter,
                                      strides = [1, 1, 1, 1], padding = 'VALID')
    # Drop the extra dimensions
    convolution_output_1D = tf.squeeze(convolution_output)
    return convolution_output_1D

random_filter_1D = tf.Variable(tf.random_normal(shape = [1, 5, 1, 1]))
convolution_output_1D = conv_layer_1D(x, random_filter_1D)

# Create the activation layer
activation_output_1D = tf.nn.relu(convolution_output_1D)

def maxpool_layer(input_1D, width):
    '''
    Create a maxpool on a moving window with a given width across an 1D input vector
    '''
    # Extend the input dimensionality from 1D to 4D
    input_2D = tf.expand_dims(input_1D, 0)
    input_3D = tf.expand_dims(input_2D, 0)
    input_4D = tf.expand_dims(input_3D, 3)
    pool_output = tf.nn.max_pool(input_4D, ksize = [1, 1, width, 1],
                                 strides = [1, 1, 1, 1], padding = 'VALID')
    pool_output_1D = tf.squeeze(pool_output)
    return pool_output_1D

maxpool_output_1D = maxpool_layer(activation_output_1D, width = 5)

def fully_connected_layer(input_layer, num_outputs):
    # Create weights
    weight_shape = tf.squeeze(tf.stack([tf.shape(input_layer), [num_outputs]]))
    weight = tf.random_normal(weight_shape, stddev = 0.1)
    bias = tf.random_normal(shape = [num_outputs])
    # Extend the input dimensionality from 1D to 2D
    input_layer_2D = tf.expand_dims(input_layer, 0)
    fully_connected_output = tf.add(tf.matmul(input_layer_2D, weight), bias)
    # Drop the extra dimensions
    fully_connected_output_1D = tf.squeeze(fully_connected_output)
    return fully_connected_output_1D

fully_connected_output_1D = fully_connected_layer(maxpool_output_1D, num_outputs = 5)

init = tf.global_variables_initializer()
session_1D.run(init)

print(colored('\u2500'*29 + ' Different Layers Within A Neural Network ' + '\u2500'*29,
              'blue', attrs = ['bold']))
print(colored('Implementation on 1D Data', 'magenta'))
print('Input: Array of Length = 25')
print('Convolutional Layer With Filter, Length = 5, Stride Size = 1:')
print(session_1D.run(convolution_output_1D, feed_dict = {x: data}))

print('\nInput: Above Array of Length = 21')
print('Element-Wise ReLU Activation Layer:')
print(session_1D.run(activation_output_1D, feed_dict = {x: data}))

print('\nInput: Above Array of Length = 21')
print('MaxPool Layer With Moving Window of Width = 5, Stride Size = 1:')
print(session_1D.run(maxpool_output_1D, feed_dict = {x: data}))

print('\nInput: Above Array of Length = 17')
print('Final Layer Fully Connecting All 4 Rows With 5 Outputs:')
print(session_1D.run(fully_connected_output_1D, feed_dict = {x: data}))

#### Implementation On 2-Dimensional Data
ops.reset_default_graph()
session_2D = tf.Session()
tf.set_random_seed(13)
np.random.seed(13)

data = np.random.normal(size = [10, 10])
x = tf.placeholder(dtype = tf.float32, shape = [10, 10])

def conv_layer_2D(input_2D, random_filter):
    '''
    Creates a convolutional layer just as in the 1D case.
    Since the input contains dimensions of both height and width, it just needs to be
    expanded in 2 dimensions, namely a batch size of 1 and a channel size = 1
    '''
    input_3D = tf.expand_dims(input_2D, 0)
    input_4D = tf.expand_dims(input_3D, 3)
    convolution_output = tf.nn.conv2d(input_4D, filter = random_filter,
                                      strides = [1, 2, 2, 1], padding = 'VALID')
    # Drop the extra dimensions
    convolution_output_2D = tf.squeeze(convolution_output)
    return convolution_output_2D

random_filter_2D = tf.Variable(tf.random_normal(shape = [2, 2, 1, 1]))
convolution_output_2D = conv_layer_2D(x, random_filter_2D)

# Create the activation layer
activation_output_2D = tf.nn.relu(convolution_output_2D)

def maxpool_layer(input_2D, width, height):
    '''
    Create a 2D maxpool layer just as in the 1D case,
    except both the width and height of the moving window needs to be provided
    '''
    # Extend the input dimensionality from 1D to 4D
    input_3D = tf.expand_dims(input_2D, 0)
    input_4D = tf.expand_dims(input_3D, 3)
    pool_output = tf.nn.max_pool(input_4D, ksize = [1, height, width, 1],
                                 strides = [1, 1, 1, 1], padding = 'VALID')
    pool_output_2D = tf.squeeze(pool_output)
    return pool_output_2D

maxpool_output_2D = maxpool_layer(activation_output_2D, width = 2, height = 2)

def fully_connected_layer(input_layer, num_outputs):
    # Flatten the input from 2D to 1D since the 2D input is considered as 1 object
    flattened_input = tf.reshape(input_layer, [-1])
    # Create weights
    weight_shape = tf.squeeze(tf.stack([tf.shape(flattened_input), [num_outputs]]))
    weight = tf.random_normal(weight_shape, stddev = 0.1)
    bias = tf.random_normal(shape = [num_outputs])
    # Change the dimensionality of the flattened input from 1D back to 2D
    input_2D = tf.expand_dims(flattened_input, 0)
    fully_connected_output = tf.add(tf.matmul(input_2D, weight), bias)
    # Drop the extra dimensions
    fully_connected_output_2D = tf.squeeze(fully_connected_output)
    return fully_connected_output_2D

fully_connected_output_2D = fully_connected_layer(maxpool_output_2D, num_outputs = 5)

init = tf.global_variables_initializer()
session_2D.run(init)

print(colored('\nImplementation on 2D Data', 'cyan'))
print('Input: Array of Size = [10, 10]')
print('Convolutional Layer of Size = [2, 2], Stride Size = [2, 2]:')
print(session_2D.run(convolution_output_2D, feed_dict = {x: data}))

print('\nInput: Above Array of Size = [5, 5]')
print('Element-Wise ReLU Activation Layer:')
print(session_2D.run(activation_output_2D, feed_dict = {x: data}))

print('\nInput: Above Array of Size = [5, 5]')
print('MaxPool Layer With Moving Window of Size = [2, 2], Stride Size = [1, 1]:')
print(session_2D.run(maxpool_output_2D, feed_dict = {x: data}))

print('\nInput: Above Array of Size = [4, 4]')
print('Final Layer Fully Connecting All 4 Rows With 5 Outputs:')
print(session_2D.run(fully_connected_output_2D, feed_dict = {x: data}))

#%% Multi-Layer Neural Network
ops.reset_default_graph()
multilayer_session = tf.Session()
tf.set_random_seed(3)
np.random.seed(3)

# Load the low birth weight dataset
birthweight_df = pd.read_csv('Data/Birthweight/birthweight.dat', sep = '\t')
# Extract the features and target from the dataset
birthweight_features = np.array(birthweight_df[birthweight_df.columns[1:8]])
birthweight = np.array(birthweight_df[birthweight_df.columns[8]])

# Split the dataset into training (80%) and testing (20%) sets
train_indices = np.random.choice(len(birthweight_features),
                                 round(len(birthweight_features) * 0.8), replace = False)
test_indices = np.array(
    list(set(range(len(birthweight_features))) - set(train_indices)))
birthweight_train_features = birthweight_features[train_indices]
birthweight_test_features = birthweight_features[test_indices]
birthweight_train = birthweight[train_indices]
birthweight_test = birthweight[test_indices]

def multilayer_normalize_columns(m, column_min = np.array([None]),
                                 column_max = np.array([None])):
    '''
    Scaling the features between 0 and 1 (Min-Max Scaling)
    '''
    if not column_min[0]:
        column_min = m.min(axis = 0)
    if not column_max[0]:
        column_max = m.max(axis = 0)
    return (m - column_min)/(column_max - column_min), column_min, column_max

birthweight_train_features, train_min, train_max = np.nan_to_num(
    multilayer_normalize_columns(birthweight_train_features))
birthweight_test_features, _, _ = np.nan_to_num(
    multilayer_normalize_columns(birthweight_test_features, train_min, train_max))

features = tf.placeholder(shape = [None, 7], dtype = tf.float32)
target = tf.placeholder(shape = [None, 1], dtype = tf.float32)

# Initialize the hidden and output layers
# First hidden layer with 50 nodes
weight1 = tf.Variable(tf.random_normal(shape = [7, 50], stddev = 10.0))
bias1 = tf.Variable(tf.random_normal(shape = [50], stddev = 10.0))
hidden_layer1 = tf.nn.relu(tf.add(tf.matmul(features, weight1), bias1))

# Second hidden layer with 25 nodes
weight2 = tf.Variable(tf.random_normal(shape = [50, 25], stddev = 10.0))
bias2 = tf.Variable(tf.random_normal(shape = [25], stddev = 10.0))
hidden_layer2 = tf.nn.relu(tf.add(tf.matmul(hidden_layer1, weight2), bias2))

# Third hidden layer with 5 nodes
weight3 = tf.Variable(tf.random_normal(shape = [25, 5], stddev = 10.0))
bias3 = tf.Variable(tf.random_normal(shape = [5], stddev = 10.0))
hidden_layer3 = tf.nn.relu(tf.add(tf.matmul(hidden_layer2, weight3), bias3))

# Output layer
weight4 = tf.Variable(tf.random_normal(shape = [5, 1], stddev = 10.0))
bias4 = tf.Variable(tf.random_normal(shape = [1], stddev = 10.0))
final_output = tf.nn.relu(tf.add(tf.matmul(hidden_layer3, weight4), bias4))

# Using the L1 (absolute value of the difference) loss function
multilayer_loss = tf.reduce_mean(tf.abs(target - final_output))

multilayer_optimizer = tf.train.AdamOptimizer(0.025)
multilayer_train_step = multilayer_optimizer.minimize(multilayer_loss)
init = tf.global_variables_initializer()
multilayer_session.run(init)

multilayer_train_loss = []
multilayer_test_loss = []
print(colored('\u2500'*36 + ' Multi-Layer Neural Network ' + '\u2500'*36, 'yellow',
              attrs = ['bold']))
for j in range(1600):
    rand_index = np.random.choice(len(birthweight_train_features), size = 150)
    rand_x = birthweight_train_features[rand_index]
    rand_y = np.transpose([birthweight_train[rand_index]])
    multilayer_session.run(multilayer_train_step, feed_dict = {features: rand_x,
                                                               target: rand_y})
    train_temp_loss = multilayer_session.run(
        multilayer_loss, feed_dict = {features: rand_x, target: rand_y})
    multilayer_train_loss.append(train_temp_loss)
    test_temp_loss = multilayer_session.run(
        multilayer_loss, feed_dict = {features: birthweight_test_features,
                                      target: np.transpose([birthweight_test])})
    multilayer_test_loss.append(test_temp_loss)
    if (j + 1) % 100 == 0:
        print('Iteration {}: Loss = {:.5f}'.format(j+1, train_temp_loss))

plt.figure()
plt.plot(multilayer_train_loss, 'k-', label = 'Training Loss')
plt.plot(multilayer_test_loss, 'r--', label = 'Testing Loss')
plt.title('L1 Loss Variations Across Iterations (Multi-Layer Neural Network)')
plt.xlabel('Iteration')
plt.xlim([0, 1600])
plt.ylabel('Loss')
plt.legend(loc = 'upper right')

actual = np.array(birthweight_df['LOW'])
train_actual = actual[train_indices]
test_actual = actual[test_indices]

train_predictions = [p[0] for p in multilayer_session.run(
    final_output, feed_dict = {features: birthweight_train_features})]
train_predictions = np.array([1.0 if p < 2500.0 else 0.0 for p in train_predictions])
test_predictions = [p[0] for p in multilayer_session.run(
    final_output, feed_dict = {features: birthweight_test_features})]
test_predictions = np.array([1.0 if p < 2500.0 else 0.0 for p in test_predictions])

# Calculate the accuracies for both the training and testing sets
train_accuracy = np.mean([a == b for a, b in zip(train_predictions, train_actual)])
test_accuracy = np.mean([a == b for a, b in zip(test_predictions, test_actual)])
print('\nTraining Accuracy = {:.5f}'.format(train_accuracy))
print('Testing Accuracy = {:.5f}'.format(test_accuracy))

#%% Improving The Logistic Regression Model
ops.reset_default_graph()
logistic_session = tf.Session()
np.random.seed(3)
tf.set_random_seed(3)

birthweight_target = np.array(birthweight_df['LOW'])

# Split the dataset into training (80%) and testing (20%) sets
training_indices = np.random.choice(
    len(birthweight_features), round(len(birthweight_features) * 0.8), replace = False)
testing_indices = np.array(
    list(set(range(len(birthweight_features))) - set(training_indices)))
birthweight_train_features = birthweight_features[training_indices]
birthweight_test_features = birthweight_features[testing_indices]
birthweight_train_target = birthweight_target[training_indices]
birthweight_test_target = birthweight_target[testing_indices]

birthweight_train_features, train_min, train_max = np.nan_to_num(
    multilayer_normalize_columns(birthweight_train_features))
birthweight_test_features, _, _ = np.nan_to_num(
    multilayer_normalize_columns(birthweight_test_features, train_min, train_max))

features = tf.placeholder(shape = [None, 7], dtype = tf.float32)
target = tf.placeholder(shape = [None, 1], dtype = tf.float32)

# Instantiate the 2 hidden layers and the output layer
# First hidden layer with 50 nodes
A1 = tf.Variable(tf.random_normal(shape = [7, 50]))
b1 = tf.Variable(tf.random_normal(shape = [50]))
logistic_layer1 = tf.nn.sigmoid(tf.add(tf.matmul(features, A1), b1))

# Second hidden layer with 25 nodes
A2 = tf.Variable(tf.random_normal(shape = [50, 25]))
b2 = tf.Variable(tf.random_normal(shape = [25]))
logistic_layer2 = tf.nn.sigmoid(tf.add(tf.matmul(logistic_layer1, A2), b2))

# Final output (Note the loss function applied to the final output is a sigmoid function)
A3 = tf.Variable(tf.random_normal(shape = [25, 1]))
b3 = tf.Variable(tf.random_normal(shape = [1]))
logistic_final_output = tf.add(tf.matmul(logistic_layer2, A3), b3)

# Using the cross entropy loss function for logistic regression
logistic_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
    logits = logistic_final_output, labels = target))

logistic_optimizer = tf.train.AdamOptimizer(0.002)
logistic_train_step = logistic_optimizer.minimize(logistic_loss)
init = tf.global_variables_initializer()
logistic_session.run(init)

logistic_prediction = tf.cast(tf.equal(tf.round(tf.nn.sigmoid(logistic_final_output)),
                                       target), tf.float32)
logistic_accuracy = tf.reduce_mean(logistic_prediction)

loss = []
logistic_train_accuracy = []
logistic_test_accuracy = []
print(colored('\u2500'*30 + ' Improving The Logistic Regression Model ' + '\u2500'*30,
              'cyan', attrs = ['bold']))
for k in range(1500):
    rand_index = np.random.choice(len(birthweight_train_features), size = 90)
    rand_x = birthweight_train_features[rand_index]
    rand_y = np.transpose([birthweight_train_target[rand_index]])
    logistic_session.run(logistic_train_step, feed_dict = {features: rand_x,
                                                           target: rand_y})
    temp_loss = logistic_session.run(logistic_loss, feed_dict = {features: rand_x,
                                                                 target: rand_y})
    loss.append(temp_loss)
    temp_train_accuracy = logistic_session.run(
        logistic_accuracy, feed_dict = {features: birthweight_train_features,
                                        target: np.transpose([birthweight_train_target])})
    logistic_train_accuracy.append(temp_train_accuracy)
    temp_test_accuracy = logistic_session.run(
        logistic_accuracy, feed_dict = {features: birthweight_test_features,
                                        target: np.transpose([birthweight_test_target])})
    logistic_test_accuracy.append(temp_test_accuracy)
    if (k+1) % 100 == 0:
        print('Iteration {}: Loss = {:.5f}'.format(k+1, temp_loss))

plt.figure()
plt.plot(loss, 'k-')
plt.title('Cross Entropy Loss Variations Across Iterations\n' +
          '(Multi-Layered Logistic Neural Network)')
plt.xlabel('Iteration')
plt.xlim([0, 1600])
plt.ylabel('Loss')

plt.figure()
plt.plot(logistic_train_accuracy, 'k-', label = 'Training Accuracy')
plt.plot(logistic_test_accuracy, 'r--', label = 'Testing Accuracy')
plt.title('Accuracy of Multi-Layered Logistic Neural Network')
plt.xlabel('Iteration')
plt.xlim([0, 1600])
plt.ylabel('Accuracy')
plt.legend(loc = 'lower right')

#%% Learning To Play Tic Tac Toe
ops.reset_default_graph()

def print_board(board):
    symbols = ['O', ' ', 'X']
    board_plus_one = [int(x) + 1 for x in board]
    board_line1 = ' {} | {} | {}'.format(symbols[board_plus_one[0]],
                                         symbols[board_plus_one[1]],
                                         symbols[board_plus_one[2]])
    board_line2 = ' {} | {} | {}'.format(symbols[board_plus_one[3]],
                                         symbols[board_plus_one[4]],
                                         symbols[board_plus_one[5]])
    board_line3 = ' {} | {} | {}'.format(symbols[board_plus_one[6]],
                                         symbols[board_plus_one[7]],
                                         symbols[board_plus_one[8]])
    print(board_line1)
    print('___________')
    print(board_line2)
    print('___________')
    print(board_line3)

def symmetry(board, response, transformation):
    '''
    Possible board configurations: List of 9 integers (opposing mark = -1,
                                                       friendly mark = 1, empty space = 0)
    Possible transformations: rotate90, rotate180, rotate270,
                              flip_h (horizontal reflection), flip_v (vertical reflection)
    Returns a tuple (new_board, new_response)
    '''
    if transformation == 'rotate180':
        new_response = 8 - response
        return board[::-1], new_response
    elif transformation == 'rotate90':
        new_response = [6, 3, 0, 7, 4, 1, 8, 5, 2].index(response)
        tuple_board = list(zip(*[board[6:9], board[3:6], board[0:3]]))
        return [value for item in tuple_board for value in item],  new_response
    elif transformation == 'rotate270':
        new_response = [2, 5, 8, 1, 4, 7, 0, 3, 6].index(response)
        tuple_board = list(zip(*[board[0:3], board[3:6], board[6:9]]))[::-1]
        return [value for item in tuple_board for value in item], new_response
    elif transformation == 'flip_v':
        new_response = [6, 7, 8, 3, 4, 5, 0, 1, 2].index(response)
        return board[6:9] + board[3:6] + board[0:3], new_response
    elif transformation == 'flip_h':
        new_response = [2, 1, 0, 5, 4, 3, 8, 7, 6].index(response)
        new_board = board[::-1]
        return new_board[6:9] + new_board[3:6] + new_board[0:3], new_response
    else:
        raise ValueError('Method Has Not Been Implemented')

def get_random_move(moves, num_random_transformations = 2):
    '''
    Performs random transformations to the board configuration
    '''
    (board, response) = random.choice(moves)
    possible_transformations = ['rotate90', 'rotate180', 'rotate270', 'flip_h', 'flip_v']
    for l in range(num_random_transformations):
        random_transformations = random.choice(possible_transformations)
        (board, response) = symmetry(board, response, random_transformations)
        return board, response

tic_tac_toe_session = tf.Session()
optimal_moves_df = pd.read_csv('Data/base_tic_tac_toe_moves.csv', header = None)
optimal_moves = []
for m in range(len(optimal_moves_df)):
    optimal_moves.append((
        list(np.array(optimal_moves_df[optimal_moves_df.columns[0:9]])[m]),
         optimal_moves_df[9][m]))

train_set = []
for t in range(500):
    train_set.append(get_random_move(optimal_moves))

# Remove all instances of the board and an optimal response from the training set
# This ensures that the model will be able to generalize in making the most optimal move
test_board = [-1, 0, 0, 1, -1, -1, 0, 0, 1]
train_set = [c for c in train_set if c[0] != test_board]

X = tf.placeholder(dtype = tf.float32, shape = [None, 9])
Y = tf.placeholder(dtype = tf.int32, shape = [None])

# Instantiate the hidden layer and the output layer
# Hidden layer (sigmoid activation function used) with 81 nodes
A1 = tf.Variable(tf.random_normal(shape = [9, 81]))
b1 = tf.Variable(tf.random_normal(shape = [81]))
hidden_layer = tf.nn.sigmoid(tf.add(tf.matmul(X, A1), b1))

# Output layer
A2 = tf.Variable(tf.random_normal(shape = [81, 9]))
b2 = tf.Variable(tf.random_normal(shape = [9]))
model_output = tf.add(tf.matmul(hidden_layer, A2), b2)

tic_tac_toe_loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(
    logits = model_output, labels = Y))

tic_tac_toe_optimizer = tf.train.GradientDescentOptimizer(0.025)
tic_tac_toe_train_step = tic_tac_toe_optimizer.minimize(tic_tac_toe_loss)
init = tf.global_variables_initializer()
tic_tac_toe_session.run(init)

tic_tac_toe_prediction = tf.argmax(model_output, 1)

loss = []
print(colored('\u2500'*35 + ' Learning To Play Tic-Tac-Toe ' + '\u2500'*35, 'magenta',
              attrs = ['bold']))
for n in range(10000):
    rand_indices = np.random.choice(range(len(train_set)), size = 50, replace = False)
    batch = [train_set[index] for index in rand_indices]
    board_configuration = np.array([c[0] for c in batch])
    target = np.array([c[1] for c in batch])
    tic_tac_toe_session.run(tic_tac_toe_train_step,
                            feed_dict = {X: board_configuration, Y: target})
    temp_loss = tic_tac_toe_session.run(
        tic_tac_toe_loss, feed_dict = {X: board_configuration, Y: target})
    loss.append(temp_loss)
    if (n+1) % 500 == 0:
        print('Iteration {}: Loss = {:.5f}'.format(n+1, temp_loss))

plt.figure()
plt.plot(loss, 'k-')
plt.title('Loss (Soft-Max) Variations Across Iterations (Tic-Tac-Toe)')
plt.xlabel('Iteration')
plt.xlim([0, 10000])
plt.ylabel('Loss')
plt.ylim(bottom = 0)

predictions = tic_tac_toe_session.run(tic_tac_toe_prediction,
                                      feed_dict = {X: [test_board]})
print('\nPredicted optimal index to move:', predictions)

stop = time.perf_counter()
print('\u2550'*100)
duration = stop - start
minutes = divmod(duration, 60)[0]
seconds = divmod(duration, 60)[1]
print(colored('Execution Duration: {:.2f}s ({:.1f}mins, {:.2f}s)'.format(
    duration, minutes, seconds), 'red'))

def check(board):
    wins = [[0, 1, 2], [3, 4, 5], [6, 7, 8], # Horizontal possibilities
            [0, 3, 6], [1, 4, 7], [2, 5, 8], # Vertical possibilities
            [0, 4, 8], [2, 4, 6]] # Diagonal possibilities
    for q in range(len(wins)):
        if board[wins[q][0]] == board[wins[q][1]] == board[wins[q][2]] == 1.0:
            return 1
        elif board[wins[q][0]] == board[wins[q][1]] == board[wins[q][2]] == -1.0:
            return 1
    return 0

game_tracker = [0.0]*9
win_logical = False
num_moves = 0
print(colored('\nSample Game:', 'grey', attrs = ['bold']))
print(' 0 | 1 | 2 ')
print('___________')
print(' 3 | 4 | 5 ')
print('___________')
print(' 6 | 7 | 8 ', end = '')
while not win_logical:
    player_input = input('Please input your move based on the grid above: ')
    if game_tracker[int(player_input)] == 1.0:
        print('Incorrect Move! Please input an appropriate move!')
        player_input = input('Please input your move based on the grid above: ')
    if game_tracker[int(player_input)] == -1.0:
        print('Incorrect Move! Please input an appropriate move!')
        player_input = input('Please input your move based on the grid above: ')
    num_moves += 1
    game_tracker[int(player_input)] = 1.0
   # Obtain all logits for each index first to produce all possible moves
    [potential_moves] = tic_tac_toe_session.run(model_output,
                                                feed_dict = {X: [game_tracker]})
    # Find the allowed moves where the element in the game tracker list = 0.0
    allowed_moves = [index for index, e in enumerate(game_tracker) if e == 0.0]
    # Taking argmax of logits to find the best move if it is in the list of allowed moves
    best_move = np.argmax([move if index in allowed_moves else -999.0
                           for index, move in enumerate(potential_moves)])
    game_tracker[int(best_move)] = -1.0
    print('Model has made a move')
    print_board(game_tracker)
    if check(game_tracker) == 1 or num_moves >= 5:
        # Maximum number of moves = 4
        print('Game Over!')
        win_logical = True