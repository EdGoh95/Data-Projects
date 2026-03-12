#!/usr/bin/env python3
"""
Machine Learning Model Serving Patterns and Best Practices (Packt Publishing)
Chapter 8: Two-Phase Model Serving
"""
import numpy as np
import pandas as pd
import tensorflow as tf

def phase1_model_prediction(x):
    print('The value of x is:', x)
    if x < 0.5:
        return True
    else:
        return False

def phase2_model_prediction():
    print('Phase 2 model is being called...')
    prediction = np.random.choice(['Class A', 'Class B', 'Class C'])
    return prediction

phase1_prediction = phase1_model_prediction(np.random.uniform(0, 1))
if phase1_prediction == True:
    response = phase2_model_prediction()
    print(response)
else:
    print('Phase 2 model was not called')

(MNIST_images_train, MNIST_labels_train), (MNIST_images_test, MNIST_labels_test) = tf.keras.datasets.mnist.load_data()
MNIST_images_train = MNIST_images_train.astype(np.float32)/255.0
MNIST_images_train = MNIST_images_train.reshape(len(MNIST_images_train), 28, 28, 1)
MNIST_images_test = MNIST_images_test.astype(np.float32)/255.0
MNIST_images_test = MNIST_images_test.reshape(len(MNIST_images_test), 28, 28, 1)
MNIST_labels_train = tf.keras.utils.to_categorical(MNIST_labels_train, 10)
MNIST_labels_test = tf.keras.utils.to_categorical(MNIST_labels_test, 10)

vgg_model = tf.keras.Sequential([
    tf.keras.layers.Conv2D(64, 3, activation = 'relu', padding = 'same', input_shape = (28, 28, 1)),
    tf.keras.layers.Conv2D(64, 3, activation = 'relu', padding = 'same'),
    tf.keras.layers.MaxPooling2D(2, 2), tf.keras.layers.BatchNormalization(),

    tf.keras.layers.Conv2D(128, 3, activation = 'relu', padding = 'same'),
    tf.keras.layers.Conv2D(128, 3, activation = 'relu', padding = 'same'),
    tf.keras.layers.MaxPooling2D(2, 2), tf.keras.layers.BatchNormalization(),

    tf.keras.layers.Conv2D(256, 3, activation = 'relu', padding = 'same'),
    tf.keras.layers.Conv2D(256, 3, activation = 'relu', padding = 'same'),
    tf.keras.layers.Conv2D(256, 3, activation = 'relu', padding = 'same'),
    tf.keras.layers.MaxPooling2D(2, 2), tf.keras.layers.BatchNormalization(),

    tf.keras.layers.Conv2D(512, 3, activation = 'relu', padding = 'same'),
    tf.keras.layers.Conv2D(512, 3, activation = 'relu', padding = 'same'),
    tf.keras.layers.Conv2D(512, 3, activation = 'relu', padding = 'same'),
    tf.keras.layers.MaxPooling2D(2, 2), tf.keras.layers.BatchNormalization(),

    tf.keras.layers.Flatten(), tf.keras.layers.Dense(4096, activation = 'relu'),
    tf.keras.layers.Dropout(0.5), tf.keras.layers.Dense(4096, activation = 'relu'),
    tf.keras.layers.Dropout(0.5), tf.keras.layers.Dense(10, activation = 'softmax')])

vgg_model.compile(loss = tf.keras.losses.CategoricalCrossentropy(from_logits = False),
                  optimizer = 'adam', metrics = ['accuracy'])
vgg_model.fit(MNIST_images_train, MNIST_labels_train, epochs = 5, batch_size = 256, verbose = 1,
              validation_data = (MNIST_images_test, MNIST_labels_test))
vgg_model.save('MNIST VGG-16 Model')