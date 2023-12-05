#!/usr/bin/env python3
"""
Deep Learning With TensorFlow and Keras Third Edition (Packt Publishing) Chapter 9: Generative Models
"""
import os
import time
import matplotlib.pyplot as plt
import tensorflow as tf
import pix2pix
from IPython.display import clear_output

#%% Load The Yosemite Summer And Winter Datasets
yosemite_trainA = tf.data.Dataset.list_files('Data/summer2winter_yosemite/train*/*', shuffle = False)
yosemite_trainB = tf.data.Dataset.list_files('Data/summer2winter_yosemite/trainB/*', shuffle = False)
yosemite_testA = tf.data.Dataset.list_files('Data/summer2winter_yosemite/testA/*', shuffle = False)
yosemite_testB = tf.data.Dataset.list_files('Data/summer2winter_yosemite/testB/*', shuffle = False)

def load_images(path):
    image = tf.io.read_file(path)
    image = tf.io.decode_png(image, channels = 3)
    label = tf.strings.split(path, os.path.sep)[-2]
    return image, label

yosemite_summer_train = yosemite_trainA.map(load_images)
yosemite_winter_train = yosemite_trainB.map(load_images)
yosemite_summer_test = yosemite_testA.map(load_images)
yosemite_winter_test = yosemite_testB.map(load_images)

#%% Data Preparation & Preprocessing
def normalize(input_image, label):
    input_image = (tf.cast(input_image, tf.float32) - 127.5)/127.5
    return input_image, label

def preprocess_train_image(image, label):
    image = random_jitter(image)
    image = normalize(image, label)
    return image, label

def preprocess_test_image(image, label):
    image = normalize(image, label)
    return image, label

#%% Data Augmentation
def random_crop(image):
    cropped_image = tf.image.random_crop(image, size = [256, 256, 3])
    return cropped_image

def random_jitter(image):
    # Resizing the image to dimension 286x286x3
    image = tf.image.resize(image, [286, 286], method = tf.image.ResizeMethod.NEAREST_NEIGHBOR)
    image = random_crop(image)
    # Random mirroring of image
    image = tf.image.random_flip_left_right(image)
    return image

#%% Save The Generated Images
def generate_images(model, input_image):
  prediction = model(input_image)

  plt.figure(figsize = (12, 12))

  display_list = [input_image[0], prediction[0]]
  title = ['Input Image', 'Predicted Image']
  for index in range(2):
    plt.subplot(1, 2, index+1)
    plt.title(title[index])
    plt.imshow((display_list[index] + 1)/2)
    plt.axis('off')
  plt.savefig('Images/CycleGAN Image.png')

yosemite_summer_train = yosemite_summer_train.cache().map(
    preprocess_train_image, num_parallel_calls = tf.data.AUTOTUNE).prefetch(tf.data.AUTOTUNE)\
    .shuffle(1000).batch(1)
yosemite_winter_train = yosemite_winter_train.cache().map(
    preprocess_train_image, num_parallel_calls = tf.data.AUTOTUNE).prefetch(tf.data.AUTOTUNE)\
    .shuffle(1000).batch(1)
yosemite_summer_test = yosemite_summer_test.map(
    preprocess_test_image, num_parallel_calls = tf.data.AUTOTUNE).prefetch(tf.data.AUTOTUNE)\
    .shuffle(1000).batch(1)
yosemite_winter_test = yosemite_winter_test.map(
    preprocess_test_image, num_parallel_calls = tf.data.AUTOTUNE).prefetch(tf.data.AUTOTUNE)\
    .shuffle(1000).batch(1)

f_generator = pix2pix.unet_generator(3, norm_type = 'instancenorm')
g_generator = pix2pix.unet_generator(3, norm_type = 'instancenorm')

image_discriminator = pix2pix.discriminator(norm_type = 'instancenorm', target = False)
label_discriminator = pix2pix.discriminator(norm_type = 'instancenorm', target = False)

#%% Visualize Some Of The Images
summer_sample = iter(yosemite_summer_train).get_next()[0][0].numpy()
winter_sample = iter(yosemite_winter_train).get_next()[0][0].numpy()
to_winter = g_generator(summer_sample)
to_summer = f_generator(winter_sample)
images = [summer_sample, to_winter, winter_sample, to_summer]
title = ['Summer', 'To Winter', 'Winter', 'To Summer']

plt.figure(figsize = (8, 8))
for index in range(len(images)):
    plt.subplot(2, 2, index + 1)
    plt.title(title[index])
    if index % 2 == 0:
        plt.imshow((images[index][0] + 1)/2)
    else:
        plt.imshow(((images[index][0] * 8) + 1)/2)

loss_objective = tf.keras.losses.BinaryCrossentropy(from_logits = True)
f_generator_optimizer = tf.keras.optimizers.Adam(1e-4, 0.5)
g_generator_optimizer = tf.keras.optimizers.Adam(1e-4, 0.5)
image_discriminator_optimizer = tf.keras.optimizers.Adam(1e-4, 0.5)
label_discriminator_optimizer = tf.keras.optimizers.Adam(1e-4, 0.5)

def discriminator_loss(actual, generated):
    actual_loss = loss_objective(tf.ones_like(actual), actual)
    generated_loss = loss_objective(tf.zeros_like(generated), generated)
    return (actual_loss + generated_loss)/2

def generator_loss(generated):
    return loss_objective(tf.ones_like(generated), generated)

def calculate_cycle_loss(actual_image, cycled_image):
    return 10 * tf.reduce_mean(tf.abs(actual_image - cycled_image))

def identity_loss(actual_image, same_image):
    return (10 * tf.reduce_mean(tf.abs(actual_image - same_image)))/2

@tf.function
def train_step(actual_image, actual_label):
    # persistent = True: tape will be used more than once to calculate the gradients
    with tf.GradientTape(persistent = True) as tape:
        fake_label = g_generator(actual_image, training = True)
        cycled_image = f_generator(fake_label, training = True)

        fake_image = f_generator(actual_label, training = True)
        cycled_label = g_generator(fake_image, training = True)

        # For calculating the identity loss
        same_image = f_generator(actual_image, training = True)
        same_label = g_generator(actual_label, training = True)

        discriminator_actual_image = image_discriminator(actual_image, training = True)
        discriminator_actual_label = label_discriminator(actual_label, training = True)

        discriminator_fake_image = image_discriminator(fake_image, training = True)
        discriminator_fake_label = label_discriminator(fake_label, training = True)

        # Calculate both generator losses
        f_generator_loss = generator_loss(discriminator_fake_image)
        g_generator_loss = generator_loss(discriminator_fake_label)

        total_cycle_loss = calculate_cycle_loss(actual_image, cycled_image) + calculate_cycle_loss(
            actual_label, cycled_label)
        total_f_generator_loss = f_generator_loss + total_cycle_loss + identity_loss(actual_image,
                                                                                     same_image)
        total_g_generator_loss = g_generator_loss + total_cycle_loss + identity_loss(actual_label,
                                                                                     same_label)

        discriminator_image_loss = discriminator_loss(discriminator_actual_image, discriminator_fake_image)
        discriminator_label_loss = discriminator_loss(discriminator_actual_label, discriminator_fake_label)

        # Calculate the gradients for both the generator and discriminator
        f_generator_gradients = tape.gradient(total_f_generator_loss, f_generator.trainable_variables)
        g_generator_gradients = tape.gradient(total_g_generator_loss, g_generator.trainable_variables)

        image_discriminator_gradients = tape.gradient(discriminator_image_loss,
                                                      image_discriminator.trainable_variables)
        label_discriminator_gradients = tape.gradient(discriminator_label_loss,
                                                      label_discriminator.trainable_variables)

        # Apply the gradients to the optimizer
        f_generator_optimizer.apply_gradients(zip(f_generator_gradients, f_generator.trainable_variables))
        g_generator_optimizer.apply_gradients(zip(g_generator_gradients, g_generator.trainable_variables))

        image_discriminator_optimizer.apply_gradients(zip(image_discriminator_gradients,
                                                          image_discriminator.trainable_variables))
        label_discriminator_optimizer.apply_gradients(zip(label_discriminator_gradients,
                                                          label_discriminator.trainable_variables))

checkpoint = tf.train.Checkpoint(
    f_generator = f_generator, g_generator = g_generator, image_discriminator = image_discriminator,
    label_discriminator = label_discriminator, f_generator_optimizer = f_generator_optimizer,
    g_generator_optimizer = g_generator_optimizer,
    image_discriminator_optimizer = image_discriminator_optimizer,
    label_discriminiator_optimizer = label_discriminator_optimizer)
checkpoint_manager = tf.train.CheckpointManager(checkpoint, 'Checkpoints/train', max_to_keep = 5)
#  If a checkpoint exists, restore the latest checkpoint
if checkpoint_manager.latest_checkpoint:
    checkpoint.restore(checkpoint_manager.latest_checkpoint)
    print('Latest checkpoint has been restored!')

for epoch in range(100):
    start = time.perf_counter()
    counter = 0
    for image, label in tf.data.Dataset.zip((yosemite_summer_train, yosemite_winter_train)):
        train_step(image[0][0], label[0][0])
        if counter % 10 == 0:
            print('.', end = '')
        counter += 1

    clear_output(wait = True)
    generate_images(g_generator, summer_sample)

    if (epoch + 1) % 5 == 0:
        checkpoint_location = checkpoint_manager.save()
        print('Saving checkpoint for epoch {} to {}'.format(epoch + 1, checkpoint_location))

    print('Time taken to complete epoch {}: {:.3f}s\n'.format(epoch + 1, time.perf_counter() - start))

to_winter = g_generator(summer_sample)
to_summer = f_generator(winter_sample)
images = [summer_sample, to_winter, winter_sample, to_summer]
title = ['Summer', 'To Winter', 'Winter', 'To Summer']

# plt.figure(figsize = (8, 8))
# for index in range(len(images)):
#     plt.subplot(2, 2, index + 1)
#     plt.title(title[index])
#     if index % 2 == 0:
#         plt.imshow((images[index][0] + 1)/2)
#     else:
#         plt.imshow(((images[index][0] * 8) + 1)/2)