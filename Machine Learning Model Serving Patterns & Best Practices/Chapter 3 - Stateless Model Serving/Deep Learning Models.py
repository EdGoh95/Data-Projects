#!/usr/bin/env python3
"""
Machine Learning Model Serving Patterns and Best Practices (Packt Publishing)
Chapter 3: Stateless Model Serving
"""
import tensorflow as tf
import tensorflow_datasets
from keras.models import Sequential
from keras.layers import Flatten, Dense, Dropout, Embedding, LSTM
from keras.optimizers import Adam
from keras.losses import SparseCategoricalCrossentropy
from keras.metrics import SparseCategoricalAccuracy

#%% Using Model Weights As Model States
#### Densely Connected Neural Network
(MNIST_train, MNIST_test), MNIST_info = tensorflow_datasets.load(
    'mnist', split = ['train', 'test'], shuffle_files = True, as_supervised = True, with_info = True)

def normalize_image(image, label):
    return tf.cast(image, tf.float32)/255.0, label

MNIST_train = MNIST_train.map(normalize_image, num_parallel_calls = tf.data.AUTOTUNE)
MNIST_train = MNIST_train.cache()
MNIST_train = MNIST_train.shuffle(MNIST_info.splits['train'].num_examples).batch(128)
MNIST_train = MNIST_train.prefetch(tf.data.AUTOTUNE)

MNIST_test = MNIST_test.map(normalize_image, num_parallel_calls = tf.data.AUTOTUNE)
MNIST_test = MNIST_test.batch(128).cache().prefetch(tf.data.AUTOTUNE)

neural_network = Sequential([Flatten(input_shape = (28, 28)), Dense(8, activation = 'relu'),
                            Dropout(0.5), Dense(10, activation = 'softmax')])
neural_network.compile(optimizer = Adam(1e-3), loss = SparseCategoricalCrossentropy(from_logits = True),
                       metrics = [SparseCategoricalAccuracy()])
neural_network.fit(MNIST_train, epochs = 3, validation_data = MNIST_test)
print(neural_network.summary())

# neural_network.save('Neural Network Model (MNIST)')
neural_network.save_weights('MNIST Weights')

#### Recurrent Neural Network (RNN)
RNN_model = Sequential()
RNN_model.add(Embedding(input_dim = 1000, output_dim = 64))
RNN_model.add(LSTM(128))
RNN_model.add(Dense(10))
print(RNN_model.summary())