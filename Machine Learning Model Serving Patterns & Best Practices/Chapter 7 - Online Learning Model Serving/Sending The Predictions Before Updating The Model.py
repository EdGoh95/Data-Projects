#!/usr/bin/env python3
"""
Machine Learning Model Serving Patterns and Best Practices (Packt Publishing)
Chapter 7: Online Learning Model Serving
"""
import numpy as np
import keras
from keras import datasets, utils, Sequential, Input, layers

(images_train, labels_train), (images_test, labels_test) = datasets.mnist.load_data()
images_train = images_train.astype('float32')/255.0
images_test = images_test.astype('float32')/255.0
# Ensure that the images have the following shape (28, 28, 1)
images_train = np.expand_dims(images_train[0:1000], -1)
images_test = np.expand_dims(images_test[0:100], -1)

labels_train = utils.to_categorical(labels_train[0:1000], 10)
labels_test = utils.to_categorical(labels_test[0:100], 10)

CNN_model = Sequential([
    Input(shape = (28, 28, 1)), layers.Conv2D(32, kernel_size = (3, 3), activation = 'relu'),
    layers.MaxPooling2D(pool_size = (2, 2)), layers.Flatten(), layers.Dense(10, activation = 'softmax')])

CNN_model.compile(loss = 'categorical_crossentropy', optimizer = 'adam', metrics = ['accuracy'])
CNN_model.fit(images_train, labels_train, batch_size = 128, epochs = 15, validation_split = 0.1)
CNN_model.save('MNIST CNN Model (Updated)', save_format = 'tf')