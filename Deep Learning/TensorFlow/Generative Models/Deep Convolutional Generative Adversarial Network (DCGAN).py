#!/usr/bin/env python3
"""
Deep Learning With TensorFlow and Keras Third Edition (Packt Publishing) Chapter 9: Generative Models
"""
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras import datasets, layers, models, optimizers

#%% Load And Normalize The MNIST Dataset
(MNIST_images_train, _), (_, _)= datasets.mnist.load_data()
# Normalize the pixel values to within a range [-1, 1]
MNIST_images_train = np.expand_dims(((MNIST_images_train - 127.5)/127.5), axis = 3)

#%% Defining The Deep Convolutional Generative Adversarial Network (DCGAN)
class  DCGAN():
    def __init__(self, rows, cols, channels, z = 100):
        # Input shape
        self.image_rows = rows
        self.image_cols = cols
        self.channels = channels
        self.image_shape = (self.image_rows, self.image_cols, self.channels)
        self.latent_dimension = z

        optimizer = optimizers.legacy.Adam(1e-4, 0.5)

        # Build and compile the discriminator
        self.discriminator = self.build_discriminator()
        self.discriminator.compile(loss = 'binary_crossentropy', optimizer = optimizer, metrics = ['accuracy'])

        # Build the generator
        self.generator = self.build_generator()
        # The generator then takes in noise as input and generates images based on the noisy input
        z = layers.Input(shape = (self.latent_dimension,))
        image = self.generator(z)

        self.discriminator.trainable = False
        validity = self.discriminator(image)
        self.combined_model = models.Model(z, validity)
        self.combined_model.compile(loss = 'binary_crossentropy', optimizer = optimizer)

    def build_generator(self):
        model = models.Sequential()
        model.add(layers.Dense(128*7*7, activation = 'relu', input_dim = self.latent_dimension))
        model.add(layers.Reshape((7, 7, 128)))
        model.add(layers.UpSampling2D())
        model.add(layers.Convolution2D(128, kernel_size = 3, padding = 'same'))
        model.add(layers.BatchNormalization(momentum = 0.8))
        model.add(layers.Activation('relu'))
        model.add(layers.UpSampling2D())
        model.add(layers.Convolution2D(64, kernel_size = 3, padding = 'same'))
        model.add(layers.BatchNormalization(momentum = 0.8))
        model.add(layers.Activation('relu'))
        model.add(layers.Convolution2D(self.channels, kernel_size = 3, padding = 'same'))
        model.add(layers.Activation('tanh'))
        print(model.summary())
        noise = layers.Input(shape = (self.latent_dimension,))
        image = model(noise)
        return models.Model(noise, image)

    def build_discriminator(self):
        model = models.Sequential()
        model.add(layers.Convolution2D(32, kernel_size = 3, strides = 2, input_shape = self.image_shape,
                                      padding = 'same'))
        model.add(layers.LeakyReLU(alpha = 0.2))
        model.add(layers.Dropout(0.3))
        model.add(layers.Convolution2D(64, kernel_size = 3, strides = 2, padding = 'same'))
        model.add(layers.ZeroPadding2D(padding = ((0, 1), (0, 1))))
        model.add(layers.BatchNormalization(momentum = 0.8))
        model.add(layers.LeakyReLU(alpha = 0.2))
        model.add(layers.Dropout(0.3))
        model.add(layers.Convolution2D(128, kernel_size = 3, strides = 2, padding = 'same'))
        model.add(layers.BatchNormalization(momentum = 0.8))
        model.add(layers.LeakyReLU(alpha = 0.2))
        model.add(layers.Dropout(0.3))
        model.add(layers.Convolution2D(256, kernel_size = 3, strides = 1, padding = 'same'))
        model.add(layers.BatchNormalization(momentum = 0.8))
        model.add(layers.LeakyReLU(alpha = 0.2))
        model.add(layers.Dropout(0.3))
        model.add(layers.Flatten())
        model.add(layers.Dense(1, activation = 'sigmoid'))
        print(model.summary())
        image = layers.Input(shape = self.image_shape)
        validity = model(image)
        return models.Model(image, validity)

    def train(self, epochs, batch_size = 256, save_interval = 50):
        valid_labels = np.ones((batch_size , 1))
        fake_labels = np.zeros((batch_size, 1))
        for epoch in range(epochs):
            # ----------------------------
            # Training The Discriminator
            # ----------------------------
            # Randomly select half of the images
            random_images = MNIST_images_train[np.random.randint(0, MNIST_images_train.shape[0],
                                                                 batch_size)]
            # Sample noise and generate a batch of new images
            noise = np.random.normal(0, 1, (batch_size, self.latent_dimension))
            generated_images = self.generator.predict(noise)
            discriminative_loss_actual = self.discriminator.train_on_batch(random_images, valid_labels)
            discriminative_loss_fake = self.discriminator.train_on_batch(generated_images, fake_labels)
            discriminative_loss = 0.5 * np.add(discriminative_loss_actual, discriminative_loss_fake)

            # ------------------------
            # Training The Generator
            # ------------------------
            generative_loss = self.combined_model.train_on_batch(noise, valid_labels)

            print('Epoch {}: Discriminative Loss = {:.5f}, Generative Loss = {:.5f}, Accuracy = {:.2f}%'.format(
                epoch + 1, discriminative_loss[0], generative_loss, discriminative_loss[1]*100))
            if (epoch + 1) % save_interval == 0:
                self.save_images(epoch + 1)

    def save_images(self, epoch):
        noise = np.random.normal(0, 1, (5 * 5, self.latent_dimension))
        # Rescale the images to a range [0, 1]
        generated_images = (0.5 * self.generator.predict(noise)) + 0.5
        fig, axes = plt.subplots(5, 5)
        counter = 0
        for j in range(5):
            for k in range(5):
                axes[j, k].imshow(generated_images[counter, :, :, 0], cmap = 'gray')
                axes[j, k].axis('off')
                counter += 1
        fig.savefig('Images/DCGAN_Generated_MNIST_Images_After_{}_Epochs.png'.format(epoch))

MNIST_DCGAN = DCGAN(28, 28, 1)
MNIST_DCGAN.train(epochs = 3000, batch_size = 256, save_interval = 50)