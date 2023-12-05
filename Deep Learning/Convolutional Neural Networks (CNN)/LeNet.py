#!/usr/bin/env python3
"""
Deep Learning With TensorFlow and Keras Third Edition (Packt Publishing)
Chapter 3: Convolutional Neural Networks
"""
import tensorflow as tf
from tensorflow.keras import datasets, layers, utils, models, optimizers, callbacks

#%% Example of A Deep Convolutional Neural Network (DCNN): LeNet
def build_model(input_shape, classes):
    model = models.Sequential()
    # First Convolution Layer
    model.add(layers.Convolution2D(30, (5, 5), activation = 'relu', input_shape = input_shape))
    model.add(layers.MaxPooling2D(pool_size = (2, 2), strides = (2, 2)))
    # Second Convolution Layer
    model.add(layers.Convolution2D(100, (5, 5), activation = 'relu'))
    model.add(layers.MaxPooling2D(pool_size = (2, 2), strides = (2, 2)))
    # Flattening Layer
    model.add(layers.Flatten())
    model.add(layers.Dense(512, activation = 'elu'))
    # Classifier
    model.add(layers.Dense(classes, activation = 'softmax'))
    return model

(MNIST_images_train, MNIST_labels_train), (MNIST_images_test, MNIST_labels_test) = datasets.mnist.load_data()

# Reshape, normalize and cast the images into float32
MNIST_images_train = MNIST_images_train.reshape((60000, 28, 28, 1))/255.0
MNIST_images_train = MNIST_images_train.astype('float32')
MNIST_images_test = MNIST_images_test.reshape((10000, 28, 28, 1))/255.0
MNIST_images_test = MNIST_images_test.astype('float32')

# Convert the label vectors to binary class matrices
MNIST_labels_train = utils.to_categorical(MNIST_labels_train, 10)
MNIST_labels_test = utils.to_categorical(MNIST_labels_test, 10)

# Initialize the model and optimizer
MNIST_LeNet = build_model(input_shape = (28, 28, 1), classes = 10)
MNIST_LeNet.compile(loss = 'categorical_crossentropy', optimizer = optimizers.Adam(),
                    metrics = ['accuracy'])
MNIST_LeNet.summary()

callbacks = [callbacks.TensorBoard(log_dir = 'logs')]

MNIST_LeNet_fit = MNIST_LeNet.fit(MNIST_images_train, MNIST_labels_train, batch_size = 128, epochs = 20,
                                  verbose = 1, validation_split = 0.2, callbacks = callbacks)

MNIST_LeNet_score = MNIST_LeNet.evaluate(MNIST_images_test, MNIST_labels_test, verbose = 1)
print('\nTest score: {:.4f}'.format(MNIST_LeNet_score[0]))
print('Test Accuracy: {:.4f}'.format(MNIST_LeNet_score[1]))
