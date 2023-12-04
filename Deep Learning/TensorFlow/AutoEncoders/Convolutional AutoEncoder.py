#!/usr/bin/env python3
"""
Deep Learning With TensorFlow and Keras Third Edition (Packt Publishing) Chapter 8: AutoEncoders
"""
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.layers import Layer, Conv2D, MaxPooling2D, UpSampling2D

np.random.seed(11)
tf.random.set_seed(11)

#%% Convolutional AutoEncoder
class ConvolutionalEncoder(Layer):
    def __init__(self, filters):
        super(ConvolutionalEncoder, self).__init__()
        self.conv1 = Conv2D(filters = filters[0], kernel_size = 3, strides = 1, activation = 'relu',
                            padding = 'same')
        self.conv2 = Conv2D(filters = filters[1], kernel_size = 3, strides = 1, activation = 'relu',
                            padding = 'same')
        self.conv3 = Conv2D(filters = filters[2], kernel_size = 3, strides = 1, activation = 'relu',
                            padding = 'same')
        self.pool = MaxPooling2D((2, 2), padding = 'same')

    def call(self, input_features):
        x = self.conv1(input_features)
        x = self.pool(x)
        x = self.conv2(x)
        x = self.pool(x)
        x = self.conv3(x)
        x = self.pool(x)
        return x

class ConvolutionalDecoder(Layer):
    def __init__(self, filters):
        super(ConvolutionalDecoder, self).__init__()
        self.conv1 = Conv2D(filters = filters[2], kernel_size = 3, strides = 1, activation = 'relu',
                            padding = 'same')
        self.conv2 = Conv2D(filters = filters[1], kernel_size = 3, strides = 1, activation = 'relu',
                            padding = 'same')
        self.conv3 = Conv2D(filters = filters[0], kernel_size = 3, strides = 1, activation = 'relu',
                            padding = 'valid')
        self.conv4 = Conv2D(1, 3, 1, activation = 'sigmoid', padding = 'same')
        self.upsample = UpSampling2D((2, 2))

    def call(self, encoded):
        x = self.conv1(encoded)
        x = self.upsample(x)
        x = self.conv2(x)
        x = self.upsample(x)
        x = self.conv3(x)
        x = self.upsample(x)
        x = self.conv4(x)
        return x

class ConvolutionalAutoEncoder(tf.keras.Model):
    def __init__(self, filters):
        super(ConvolutionalAutoEncoder, self).__init__()
        self.loss = []
        self.encoder = ConvolutionalEncoder(filters)
        self.decoder = ConvolutionalDecoder(filters)

    def call(self, input_features):
        encoded = self.encoder(input_features)
        return self.decoder(encoded)

#%% Load The MNIST Dataset, Normalize it And Introduce Noise To It
(MNIST_images_train, _), (MNIST_images_test, _) = tf.keras.datasets.mnist.load_data()
MNIST_images_train = np.reshape(MNIST_images_train.astype(np.float32)/255.0,
                                (len(MNIST_images_train), 28, 28, 1))
MNIST_images_test = np.reshape(MNIST_images_test.astype(np.float32)/255.0,
                               (len(MNIST_images_test), 28, 28, 1))
# Generate a corrupted set of MNIST images by adding noise obtained from a normal distribution
MNIST_images_train_corrupted = MNIST_images_train + (0.5 * np.random.normal(loc = 0.0, scale = 1.0,
                                                                     size = MNIST_images_train.shape))
MNIST_images_test_corrupted = MNIST_images_test + (0.5 * np.random.normal(loc = 0.0, scale = 1.0,
                                                                          size = MNIST_images_test.shape))
MNIST_images_train_corrupted = np.clip(MNIST_images_train_corrupted, 0.0, 1.0)
MNIST_images_test_corrupted = np.clip(MNIST_images_test_corrupted, 0.0, 1.0)

convolutional_denoising_autoencoder = ConvolutionalAutoEncoder([32, 32, 16])
convolutional_denoising_autoencoder.compile(loss = 'binary_crossentropy', optimizer = 'adam')
convolutional_denoising_fit = convolutional_denoising_autoencoder.fit(
    MNIST_images_train_corrupted, MNIST_images_train,
    validation_data = (MNIST_images_test_corrupted, MNIST_images_test), epochs = 50, batch_size = 128)

plt.figure()
plt.plot(range(1, 51), convolutional_denoising_fit.history['loss'])
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
    plt.imshow(tf.reshape(convolutional_denoising_autoencoder(MNIST_images_test_corrupted)[index],
                          (28, 28)), cmap = 'gray')
    ax_reconstructed.get_xaxis().set_visible(False)
    ax_reconstructed.get_yaxis().set_visible(False)