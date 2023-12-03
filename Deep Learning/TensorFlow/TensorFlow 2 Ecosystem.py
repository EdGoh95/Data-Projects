#!/usr/bin/env python3
"""
Deep Learning With TensorFlow and Keras Third Edition (Packt Publishing)
Chapter 19: TensorFlow 2 Ecosystem
"""
import time
import numpy as np
import matplotlib.pyplot as plt
import requests
import tensorflow as tf
import tensorflow_hub as tf_hub
import tensorflow_datasets as tfds
from PIL import Image
from io import BytesIO

#%% TensorFlow Hub
def load_image_from_url(image_url, image_size):
    """
    Get the image from the specified url. Returns an image with shape [1, height, width, num_channels]
    """
    response = requests.get(image_url, headers = {'User-agent': 'Colab Sample (https://tensorflow.org)'})
    image = np.array(Image.open(BytesIO(response.content)))
    # Reshape the image
    image_reshaped = tf.reshape(image, [1, image.shape[0], image.shape[1], image.shape[2]])
    # Normalize the image by converting it to a float within [0, 1]
    image = tf.image.convert_image_dtype(image_reshaped, tf.float32)
    image_padded = tf.image.resize_with_pad(image, image_size, image_size)
    return image_padded, image

def show_image(image, title = ''):
    image_size = image.shape[1]
    width = (image_size * 6)//320
    plt.figure(figsize = (width, width))
    plt.imshow(image[0], aspect = 'equal')
    plt.axis('off')
    plt.title(title)

print('Images will be converted to {}x{}'.format(330, 330))
scaled_image, original_image = load_image_from_url(
    'https://upload.wikimedia.org/wikipedia/commons/c/c6/Okonjima_Lioness.jpg', 330)
show_image(scaled_image, 'Scaled Image')

labels_file = tf.keras.utils.get_file(
    'label.txt', origin = 'https://storage.googleapis.com/download.tensorflow.org/data/ImageNetLabels.txt')
with open(labels_file) as file:
    labels = file.readlines()
    classes = [label.strip() for label in labels]

classifier = tf_hub.load('https://tfhub.dev/tensorflow/efficientnet/b2/classification/1')
probabilities = tf.nn.softmax(classifier(scaled_image)).numpy()
top_5 = tf.argsort(probabilities, axis = -1, direction = 'DESCENDING')[0][:5].numpy()
show_image(scaled_image, '{}: {:.5f}'.format(classes[top_5[0] + 1].capitalize(),
                                             probabilities[0][top_5][0]))

#%% TensorFlow Datasets (TFDS)
start = time.perf_counter()
CNN_model = tf.keras.models.Sequential([
    tf.keras.layers.Convolution2D(16, (3, 3), activation = 'relu', input_shape = (300, 300, 3)),
    tf.keras.layers.MaxPooling2D(2, 2), tf.keras.layers.Convolution2D(32, (3, 3), activation = 'relu'),
    tf.keras.layers.MaxPooling2D(2, 2), tf.keras.layers.Convolution2D(64, (3, 3), activation = 'relu'),
    tf.keras.layers.MaxPooling2D(2, 2), tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(256, activation = 'relu'), tf.keras.layers.Dense(1, activation = 'sigmoid')])
CNN_model.compile(optimizer = 'Adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

def normalize_image(image, label):
    """
    Normalize the images from 'uint8' to 'float32'
    """
    return tf.cast(image, tf.float32)/255.0, label

def augment_image(image, label):
    image, label = normalize_image(image, label)
    image = tf.image.random_flip_left_right(image)
    return image, label

training_dataset = tfds.load('horses_or_humans', split = 'train', as_supervised = True).cache()
training_dataset = training_dataset.map(augment_image, num_parallel_calls = tf.data.AUTOTUNE)
training_data = training_dataset.shuffle(1024).batch(32).prefetch(tf.data.AUTOTUNE)
validation_dataset = tfds.load('horses_or_humans', split = 'test', as_supervised = True)
validation_dataset = validation_dataset.map(augment_image, num_parallel_calls = tf.data.AUTOTUNE)
validation_data = validation_dataset.batch(32).cache().prefetch(tf.data.AUTOTUNE)

CNN_model_fit = CNN_model.fit(training_data, epochs = 10, validation_data = validation_data,
                              validation_steps = 1)
print('Duration: {:.3f}s'.format(time.perf_counter() - start))