#!/usr/bin/env python3
"""
Deep Learning With TensorFlow and Keras Third Edition (Packt Publishing) Chapter 8: AutoEncoders
"""
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

np.random.seed(11)
tf.random.set_seed(11)

#%% Vanilla AutoEncoder
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

#%% Sparse AutoEncoder
class SparseEncoder(tf.keras.layers.Layer):
    def __init__(self, hidden_dimension):
        super(SparseEncoder, self).__init__()
        self.hidden_layer = tf.keras.layers.Dense(units = hidden_dimension, activation = tf.nn.relu,
                                                  activity_regularizer = tf.keras.regularizers.l1(1e-4))

    def call(self, input_features):
        return self.hidden_layer(input_features)

class SparseDecoder(tf.keras.layers.Layer):
    def __init__(self, hidden_dimension, original_dimension):
        super(SparseDecoder, self).__init__()
        self.output_layer = tf.keras.layers.Dense(units = original_dimension, activation = tf.nn.relu)

    def call(self, encoded):
        return self.output_layer(encoded)

class SparseAutoEncoder(tf.keras.Model):
    def __init__(self, hidden_dimension, original_dimension):
        super(SparseAutoEncoder, self).__init__()
        self.loss = []
        self.encoder = SparseEncoder(hidden_dimension = hidden_dimension)
        self.decoder = SparseDecoder(hidden_dimension = hidden_dimension, original_dimension = original_dimension)

    def call(self, input_features):
        encoded = self.encoder(input_features)
        return self.decoder(encoded)

def loss(predictions, actual):
    return tf.reduce_mean(tf.square(tf.subtract(predictions, actual)))

def train(loss, model, optimizer, original):
    with tf.GradientTape() as tape:
        predictions = model(original)
        reconstruction_error = loss(predictions, original)
        gradients = tape.gradient(reconstruction_error, model.trainable_variables)
        gradient_variables = zip(gradients, model.trainable_variables)
    optimizer.apply_gradients(gradient_variables)
    return reconstruction_error

def train_loop(model, optimizer, loss, dataset, epochs = 20):
    for epoch in range(epochs):
        epoch_loss = 0
        for step, batch_features in enumerate(dataset):
            epoch_loss += train(loss, model, optimizer, batch_features)
        model.loss.append(epoch_loss)
        print('Epoch {}/{}: Loss = {:.5f}'.format(epoch+1, epochs, epoch_loss.numpy()))

(MNIST_images_train, _), (MNIST_images_test, _) = tf.keras.datasets.mnist.load_data()
MNIST_images_train = np.reshape(MNIST_images_train.astype(np.float32)/255.0,
                                (MNIST_images_train.shape[0], 784))
MNIST_images_test = np.reshape(MNIST_images_test.astype(np.float32)/255.0,
                               (MNIST_images_test.shape[0], 784))
MNIST_training_dataset = tf.data.Dataset.from_tensor_slices(MNIST_images_train).batch(256)

autoencoder = SparseAutoEncoder(hidden_dimension = 128, original_dimension = 784)
optimizer = tf.keras.optimizers.Adam(learning_rate = 1e-2)
train_loop(autoencoder, optimizer, loss, MNIST_training_dataset, epochs = 50)

plt.figure()
plt.plot(range(1, 51), autoencoder.loss)
plt.xlabel('Epoch')
plt.ylabel('Loss')

plt.figure(figsize = (20, 4))
for index in range(10):
    # Display the original image
    ax_original = plt.subplot(2, 10, index+1)
    plt.imshow(MNIST_images_test[index].reshape(28, 28), cmap = 'gray')
    ax_original.get_xaxis().set_visible(False)
    ax_original.get_yaxis().set_visible(False)
    # Display the reconstructed image
    ax_reconstructed = plt.subplot(2, 10, (index+1)+10)
    plt.imshow(autoencoder(MNIST_images_test)[index].numpy().reshape(28, 28), cmap = 'gray')
    ax_reconstructed.get_xaxis().set_visible(False)
    ax_reconstructed.get_yaxis().set_visible(False)