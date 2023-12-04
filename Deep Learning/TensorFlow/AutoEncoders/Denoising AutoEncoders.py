#!/usr/bin/env python3
"""
Deep Learning With TensorFlow and Keras Third Edition (Packt Publishing) Chapter 8: AutoEncoders
"""
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

np.random.seed(11)
tf.random.set_seed(11)

class Encoder(tf.keras.layers.Layer):
    def __init__(self, hidden_dimension):
        super(Encoder, self).__init__()
        self.hidden_layer = tf.keras.layers.Dense(units = hidden_dimension, activation = tf.nn.relu)

    def call(self, input_features):
        return self.hidden_layer(input_features)

class Decoder(tf.keras.layers.Layer):
    def __init__(self, hidden_dimension, original_dimension):
        super(Decoder, self).__init__()
        self.output_layer = tf.keras.layers.Dense(units = original_dimension, activation = tf.nn.relu)

    def call(self, encoded):
        return self.output_layer(encoded)

class AutoEncoder(tf.keras.Model):
    def __init__(self, hidden_dimension, original_dimension):
        super(AutoEncoder, self).__init__()
        self.loss = []
        self.encoder = Encoder(hidden_dimension = hidden_dimension)
        self.decoder = Decoder(hidden_dimension = hidden_dimension, original_dimension = original_dimension)

    def call(self, input_features):
        encoded = self.encoder(input_features)
        return self.decoder(encoded)

#%% Load The MNIST Dataset, Normalize it And Introduce Noise To It
(MNIST_images_train, _), (MNIST_images_test, _) = tf.keras.datasets.mnist.load_data()
MNIST_images_train = np.reshape(MNIST_images_train.astype(np.float32)/255.0,
                                (MNIST_images_train.shape[0], 784))
MNIST_images_test = np.reshape(MNIST_images_test.astype(np.float32)/255.0,
                               (MNIST_images_test.shape[0], 784))
# Generate a corrupted set of MNIST images by adding noise obtained from a normal distribution
MNIST_images_train_corrupted = MNIST_images_train + np.random.normal(loc = 0.5, scale = 0.5,
                                                                     size = MNIST_images_train.shape)
MNIST_images_test_corrupted = MNIST_images_test + np.random.normal(loc = 0.5, scale = 0.5,
                                                                   size = MNIST_images_test.shape)

denoising_autoencoder = AutoEncoder(hidden_dimension = 50, original_dimension = 784)
denoising_autoencoder.compile(loss = 'mse', optimizer = 'adam')
denoising_fit = denoising_autoencoder.fit(MNIST_images_train, MNIST_images_train_corrupted,
                                          validation_data = (MNIST_images_test, MNIST_images_test_corrupted),
                                          epochs = 50, batch_size = 256)

plt.figure()
plt.plot(range(1, 51), denoising_fit.history['loss'])
plt.xlabel('Epoch')
plt.ylabel('Loss')

plt.figure(figsize = (20, 4))
for index in range(10):
    # Display the original images
    ax_original = plt.subplot(2, 10, index+1)
    plt.imshow(MNIST_images_test_corrupted[index].reshape(28, 28), cmap = 'gray')
    ax_original.get_xaxis().set_visible(False)
    ax_original.get_yaxis().set_visible(False)
    # Display the reconstructed images
    ax_reconstructed = plt.subplot(2, 10, (index+1)+10)
    plt.imshow(denoising_autoencoder(MNIST_images_test_corrupted)[index].numpy().reshape(28, 28),
               cmap = 'gray')
    ax_reconstructed.get_xaxis().set_visible(False)
    ax_reconstructed.get_yaxis().set_visible(False)