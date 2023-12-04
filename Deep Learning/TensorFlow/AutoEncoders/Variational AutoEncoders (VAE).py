#!/usr/bin/env python3
"""
Deep Learning With TensorFlow and Keras Third Edition (Packt Publishing) Chapter 8: AutoEncoders
"""
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

np.random.seed(333)
tf.random.set_seed(333)

class VAE(tf.keras.Model):
    def __init__(self, dimensions, **kwargs):
        h_dimension = dimensions[0]
        z_dimension = dimensions[1]
        super(VAE, self).__init__(**kwargs)
        self.FC1 = tf.keras.layers.Dense(h_dimension)
        self.FC2 = tf.keras.layers.Dense(z_dimension)
        self.FC3 = tf.keras.layers.Dense(z_dimension)
        self.FC4 = tf.keras.layers.Dense(h_dimension)
        self.FC5 = tf.keras.layers.Dense(image_size)

    def encode(self, x):
        h = tf.nn.relu(self.FC1(x))
        return self.FC2(h), self.FC3(h)

    def reparametrize(self, mu, log_sigma):
        std_dev = tf.exp(log_sigma * 0.5)
        epsilon = tf.random.normal(std_dev.shape)
        return mu + (epsilon * std_dev)

    def decode_logits(self, z):
        h = tf.nn.relu(self.FC4(z))
        return self.FC5(h)

    def decode(self, z):
        return tf.nn.sigmoid(self.decode_logits(z))

    def call(self, inputs, training = None, mask = None):
        mu, log_sigma = self.encode(inputs)
        z = self.reparametrize(mu, log_sigma)
        reconstructed_logits = self.decode_logits(z)
        return reconstructed_logits, mu, log_sigma

(fashion_images_train, fashion_labels_train), (fashion_images_test, fashion_labels_test) = \
    tf.keras.datasets.fashion_mnist.load_data()
fashion_images_train = fashion_images_train.astype(np.float32)/255.0
fashion_images_test = fashion_images_test.astype(np.float32)/255.0
print('Shape of training images: {}, Shape of training labels: {}'.format(fashion_images_train.shape,
                                                                          fashion_labels_train.shape))
print('Shape of test images: {}, Shape of test labels: {}'.format(fashion_images_test.shape,
                                                                  fashion_labels_test.shape))
image_size = fashion_images_train.shape[1] * fashion_images_train.shape[2]

fashion_dataset = tf.data.Dataset.from_tensor_slices(fashion_images_train).shuffle(100*5).batch(100)

variational_autoencoder = VAE([512, 10])
variational_autoencoder.build(input_shape = (4, image_size))
variational_autoencoder.summary()
optimizer = tf.keras.optimizers.Adam(1e-3)

for epoch in range(80):
    for step, image in enumerate(fashion_dataset):
        image = tf.reshape(image, [-1, image_size])
        with tf.GradientTape() as tape:
            # Forward Pass
            reconstructed_logits, mu, log_sigma = variational_autoencoder(image)
            '''
            Computing the reconstruction and the KL divergence losses, scaled by image_size for
            each individual pixel
            '''
            reconstruction_loss = tf.nn.sigmoid_cross_entropy_with_logits(
                labels = image, logits = reconstructed_logits)
            reconstruction_loss = tf.reduce_sum(reconstruction_loss)/100
            KL_divergence_loss = tf.reduce_mean(
                -0.5 * tf.reduce_sum(1.0 + log_sigma - tf.square(mu) - tf.exp(log_sigma), axis = -1))
            # Backpropagate and optimize
            loss = tf.reduce_mean(reconstruction_loss) + KL_divergence_loss
        gradients = tape.gradient(loss, variational_autoencoder.trainable_variables)
        for gradient in gradients:
            tf.clip_by_norm(gradient, 15)
        optimizer.apply_gradients(zip(gradients, variational_autoencoder.trainable_variables))
        if (step + 1) % 50 == 0:
            print('Epoch {}/{}, Step {}/{}: Reconstruction Loss = {:.5f}, KL Divergence Loss = {:.5f}'.format(
                epoch+1, 80, step+1, fashion_images_train.shape[0]//100, float(reconstruction_loss),
                float(KL_divergence_loss)))

z = tf.random.normal((100, 10))
output_image = variational_autoencoder.decode(z)
output_image = (tf.reshape(output_image, [-1, 28, 28]).numpy() * 255.0).astype(np.uint8)

plt.figure(figsize = (20, 4))
for index in range(10):
    # Display the original images
    ax_original = plt.subplot(2, 10, index+1)
    plt.imshow(fashion_images_train[index], cmap = 'gray')
    ax_original.get_xaxis().set_visible(False)
    ax_original.get_yaxis().set_visible(False)
    # Display the reconstructed images
    ax_reconstructed = plt.subplot(2, 10, index+1)
    plt.imshow(output_image[index], cmap = 'gray')
    ax_reconstructed.get_xaxis().set_visible(False)
    ax_reconstructed.get_yaxis().set_visible(False)