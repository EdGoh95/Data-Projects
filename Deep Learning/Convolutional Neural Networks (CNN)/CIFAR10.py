#!/usr/bin/env python3
"""
Deep Learning With TensorFlow and Keras Third Edition (Packt Publishing)
Chapter 3: Convolutional Neural Networks
"""
import numpy as np
import tensorflow as tf
import cv2
from tensorflow.keras import datasets, layers, utils, models, optimizers, callbacks, preprocessing

def build_model(input_shape, classes):
    model = models.Sequential()
    # First block
    model.add(layers.Convolution2D(32, (3, 3), padding = 'same', activation = 'elu',
                                   input_shape = input_shape))
    model.add(layers.BatchNormalization())
    model.add(layers.Convolution2D(32, (3, 3), padding = 'same', activation = 'elu'))
    model.add(layers.BatchNormalization())
    model.add(layers.MaxPooling2D(pool_size = (2, 2)))
    model.add(layers.Dropout(0.3))
    # Second block
    model.add(layers.Convolution2D(64, (3, 3), padding = 'same', activation = 'elu'))
    model.add(layers.BatchNormalization())
    model.add(layers.Convolution2D(64, (3, 3), padding = 'same', activation = 'elu'))
    model.add(layers.BatchNormalization())
    model.add(layers.MaxPooling2D(pool_size = (2, 2)))
    model.add(layers.Dropout(0.4))
    # Fourth block
    model.add(layers.Convolution2D(128, (3, 3), padding = 'same', activation = 'elu'))
    model.add(layers.BatchNormalization())
    model.add(layers.Convolution2D(128, (3, 3), padding = 'same', activation = 'elu'))
    model.add(layers.BatchNormalization())
    model.add(layers.MaxPooling2D(pool_size = (2, 2)))
    model.add(layers.Dropout(0.5))
    # Dense layer
    model.add(layers.Flatten())
    model.add(layers.Dense(classes, activation = 'softmax'))
    return model

def load_data():
    (CIFAR10_images_train, CIFAR10_labels_train), (CIFAR10_images_test, CIFAR10_labels_test) = datasets.cifar10.load_data()
    CIFAR10_images_train = CIFAR10_images_train.astype('float32')
    CIFAR10_images_test = CIFAR10_images_test.astype('float32')

    # Normalize
    mean = np.mean(CIFAR10_images_train, axis = (0, 1, 2, 3))
    stddev = np.std(CIFAR10_images_train, axis = (0, 1, 2, 3))
    CIFAR10_images_train = (CIFAR10_images_train - mean)/(stddev + 1e-7)
    CIFAR10_images_test = (CIFAR10_images_test - mean)/(stddev + 1e-7)

    CIFAR10_labels_train = utils.to_categorical(CIFAR10_labels_train, 10)
    CIFAR10_labels_test = utils.to_categorical(CIFAR10_labels_test, 10)
    return CIFAR10_images_train, CIFAR10_labels_train, CIFAR10_images_test, CIFAR10_labels_test

CIFAR10_images_train, CIFAR10_labels_train, CIFAR10_images_test, CIFAR10_labels_test = load_data()
CIFAR10_model = build_model(input_shape = CIFAR10_images_train.shape[1:], classes = CIFAR10_labels_train.shape[1])
lr_schedule = optimizers.schedules.ExponentialDecay(initial_learning_rate = 1e-3, decay_steps = 1e6,
                                                    decay_rate = 0.99)
CIFAR10_model.compile(loss = 'categorical_crossentropy',
                      optimizer = optimizers.RMSprop(learning_rate = lr_schedule), metrics = ['accuracy'])
CIFAR10_model.summary()

callbacks = [callbacks.TensorBoard(log_dir = 'logs')]

# CIFAR10_model.fit(CIFAR10_images_train, CIFAR10_labels_train, batch_size = 64, epochs = 50,
#                   steps_per_epoch = CIFAR10_images_train.shape[0]//64, validation_split = 0.2,
#                   validation_data = (CIFAR10_images_test, CIFAR10_labels_test), verbose = 1, callbacks = callbacks)

# CIFAR10_score = CIFAR10_model.evaluate(CIFAR10_images_test, CIFAR10_labels_test, batch_size = 64,
#                                        verbose = 1)
# print('\nTest score: {:.4f}'.format(CIFAR10_score[0]))
# print('Test accuracy: {:.4f}'.format(CIFAR10_score[1]))

#%% Image Augmentation
data_generator = preprocessing.image.ImageDataGenerator(
    rotation_range = 15, width_shift_range = 0.1, height_shift_range = 0.1, horizontal_flip = True)
data_generator.fit(CIFAR10_images_train)
CIFAR10_model.fit_generator(data_generator.flow(CIFAR10_images_train, CIFAR10_labels_train, batch_size = 64),
                            epochs = 200, steps_per_epoch = CIFAR10_images_train.shape[0]//64,
                            verbose = 1, validation_data = (CIFAR10_images_test, CIFAR10_labels_test))

# Saving the model to disk
CIFAR10_model_json = CIFAR10_model.to_json()
with open('CIFAR10/CIFAR10_model.json', 'w') as json_file:
    json_file.write(CIFAR10_model_json)
CIFAR10_model.save_weights('CIFAR10/CIFAR10_weights.h5')

CIFAR10_augmented_score = CIFAR10_model.evaluate(CIFAR10_images_test, CIFAR10_labels_test, batch_size = 64,
                                                 verbose = 1)
print('\nTest loss: {:.4f}'.format(CIFAR10_augmented_score[0]))
print('Test accuracy: {:.3f}%'.format(CIFAR10_augmented_score[1]*100))

#%% Prediction Using CIFAR10_model
# Load the pre-trained model
pretrained_CIFAR10_model = models.model_from_json(open('CIFAR10/CIFAR10_model.json').read())
pretrained_CIFAR10_model.load_weights('CIFAR10/CIFAR10_weights.h5')

# Load images
image_location = ['Images/cat-standing.jpg', 'Images/dog2.jpeg']
images = [cv2.resize(cv2.imread(image), (32, 32)) for image in image_location]
images = np.array(images)/255.0

pretrained_CIFAR10_model.compile(
    loss = 'categorical_crossentropy', optimizer = optimizers.RMSprop(learning_rate = lr_schedule),
    metrics = ['accuracy'])
predicted_labels = np.argmax(pretrained_CIFAR10_model.predict(images), axis = 1)
print(predicted_labels)