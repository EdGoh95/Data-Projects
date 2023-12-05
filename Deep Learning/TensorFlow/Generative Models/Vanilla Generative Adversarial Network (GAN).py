#!/usr/bin/env python3
"""
Deep Learning With TensorFlow and Keras Third Edition (Packt Publishing) Chapter 9: Generative Models
"""
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras import datasets, layers, models, initializers

#%% Load And Normalize The MNIST Dataset
(MNIST_images_train, _), (_, _)= datasets.mnist.load_data()
# Normalize the pixel values to within a range [-1, 1]
MNIST_images_train = ((MNIST_images_train.astype(np.float32) - 127.5)/127.5).reshape(60000, 784)

#%% Defining The Generator
MNIST_images_generator = models.Sequential()
MNIST_images_generator.add(layers.Dense(256, input_dim = 100))
MNIST_images_generator.add(layers.LeakyReLU(alpha = 0.2))
MNIST_images_generator.add(layers.Dense(512))
MNIST_images_generator.add(layers.LeakyReLU(alpha = 0.2))
MNIST_images_generator.add(layers.Dense(1024))
MNIST_images_generator.add(layers.LeakyReLU(alpha = 0.2))
MNIST_images_generator.add(layers.Dense(784, activation = 'tanh'))

#%% Defining The Discriminator
MNIST_images_discriminator = models.Sequential()
MNIST_images_discriminator.add(layers.Dense(1024, input_dim = 784,
                                            kernel_initializer = initializers.RandomNormal(stddev = 0.02)))
MNIST_images_discriminator.add(layers.LeakyReLU(alpha = 0.2))
MNIST_images_discriminator.add(layers.Dropout(0.3))
MNIST_images_discriminator.add(layers.Dense(512))
MNIST_images_discriminator.add(layers.LeakyReLU(alpha = 0.2))
MNIST_images_discriminator.add(layers.Dropout(0.3))
MNIST_images_discriminator.add(layers.Dense(256))
MNIST_images_discriminator.add(layers.LeakyReLU(alpha = 0.2))
MNIST_images_discriminator.add(layers.Dropout(0.3))
MNIST_images_discriminator.add(layers.Dense(1, activation = 'sigmoid'))
MNIST_images_discriminator.compile(loss = 'binary_crossentropy', optimizer = 'adam')
MNIST_images_discriminator.trainable = False

GAN_input = layers.Input(shape = (100,))
MNIST_images_generated = MNIST_images_generator(GAN_input)
GAN = models.Model(inputs = GAN_input, outputs = MNIST_images_discriminator(MNIST_images_generated))
GAN.compile(loss = 'binary_crossentropy', optimizer = 'adam')

discriminator_losses = []
generator_losses = []

def plot_loss(epoch):
    plt.figure(figsize = (10, 8))
    plt.plot(discriminator_losses, label = 'Discriminative Loss')
    plt.plot(generator_losses, label = 'Generative Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig('Images/GAN_Loss_After_{}_Epochs.png'.format(epoch))

def save_generated_images(epoch, examples = 100, dimensions = (10, 10), figsize = (10, 10)):
    noise = np.random.normal(0, 1, size = [examples, 100])
    generated_images = MNIST_images_generator.predict(noise).reshape(examples, 28, 28)
    plt.figure(figsize = figsize)
    for i in range(generated_images.shape[0]):
        plt.subplot(dimensions[0], dimensions[1], i+1)
        plt.imshow(generated_images[i], interpolation = 'nearest', cmap = 'gray_r')
        plt.axis('off')
    plt.tight_layout()
    plt.savefig('Images/Generated_Image_After_{}_Epochs.png'.format(epoch))

def train(epochs = 1, batch_size = 128):
    batch_count = int(MNIST_images_train.shape[0]/batch_size)
    print('Epochs:', epochs)
    print('Batch Size:', batch_size)
    print('Batches Per Epoch:', batch_count)
    for epoch in range(epochs):
        print('Epoch {}...'.format(epoch+1))
        for _ in range(batch_count):
            noise1 = np.random.normal(0, 1, size = [batch_size, 100])
            image_batch = MNIST_images_train[np.random.randint(0, MNIST_images_train.shape[0],
                                                               size = batch_size)]
            generated_images = MNIST_images_generator.predict(noise1)
            MNIST_fake_images = np.concatenate([image_batch, generated_images])
            MNIST_labels = np.zeros(2*batch_size)
            # One-sided label smoothing
            MNIST_labels[:batch_size] = 0.9
            # Train the discriminator
            MNIST_images_discriminator.trainable = True
            discriminator_loss = MNIST_images_discriminator.train_on_batch(MNIST_fake_images, MNIST_labels)
            # Train the generator
            noise2 = np.random.normal(0, 1, size = [batch_size, 100])
            MNIST_fake_labels = np.ones(batch_size)
            MNIST_images_discriminator.trainable = False
            generator_loss = GAN.train_on_batch(noise2, MNIST_fake_labels)

        # Store the loss of the most recent batch from the current epoch
        discriminator_losses.append(discriminator_loss)
        generator_losses.append(generator_loss)
        if epoch == 1 or (epoch + 1) % 20 == 0:
            save_generated_images(epoch + 1)
            plot_loss(epoch + 1)

train(200, 128)