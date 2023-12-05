#!/usr/bin/env python3
"""
Deep Learning With TensorFlow And Keras Third Edition (Packt Publishing)
Chapter 3: Convolutional Neural Networks
"""
import tensorflow as tf
import cv2
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras import layers, models, applications, preprocessing

def VGG16(weights_path = None):
    # First block
    model = models.Sequential()
    model.add(layers.ZeroPadding2D((1, 1), input_shape = (224, 224, 3)))
    model.add(layers.Convolution2D(64, (3, 3), activation = 'relu'))
    model.add(layers.ZeroPadding2D((1, 1)))
    model.add(layers.Convolution2D(64, (3, 3), activation = 'relu'))
    model.add(layers.MaxPooling2D((2, 2), strides = (2, 2)))
    # Second block
    model.add(layers.ZeroPadding2D((1, 1)))
    model.add(layers.Convolution2D(128, (3, 3), activation = 'relu'))
    model.add(layers.ZeroPadding2D((1, 1)))
    model.add(layers.Convolution2D(128, (3, 3), activation = 'relu'))
    model.add(layers.MaxPooling2D((2, 2), strides = (2, 2)))
    # Third block
    model.add(layers.ZeroPadding2D((1, 1)))
    model.add(layers.Convolution2D(256, (3, 3), activation = 'relu'))
    model.add(layers.ZeroPadding2D((1, 1)))
    model.add(layers.Convolution2D(256, (3, 3), activation = 'relu'))
    model.add(layers.ZeroPadding2D((1, 1)))
    model.add(layers.Convolution2D(256, (3, 3), activation = 'relu'))
    model.add(layers.MaxPooling2D((2, 2), strides = (2, 2)))
    # Fourth block
    model.add(layers.ZeroPadding2D((1, 1)))
    model.add(layers.Convolution2D(512, (3, 3), activation = 'relu'))
    model.add(layers.ZeroPadding2D((1, 1)))
    model.add(layers.Convolution2D(512, (3, 3), activation = 'relu'))
    model.add(layers.ZeroPadding2D((1, 1)))
    model.add(layers.Convolution2D(512, (3, 3), activation = 'relu'))
    model.add(layers.MaxPooling2D((2, 2), strides = (2, 2)))
    # Fifth layer
    model.add(layers.ZeroPadding2D((1, 1)))
    model.add(layers.Convolution2D(512, (3, 3), activation = 'relu'))
    model.add(layers.ZeroPadding2D((1, 1)))
    model.add(layers.Convolution2D(512, (3, 3), activation = 'relu'))
    model.add(layers.ZeroPadding2D((1, 1)))
    model.add(layers.Convolution2D(512, (3, 3), activation = 'relu'))
    model.add(layers.MaxPooling2D((2, 2), strides = (2, 2)))
    # Top layer of the VGG Network
    model.add(layers.Flatten())
    model.add(layers.Dense(4096, activation = 'relu'))
    model.add(layers.Dropout(0.5))
    model.add(layers.Dense(4096, activation = 'relu'))
    model.add(layers.Dropout(0.5))
    model.add(layers.Dense(1000, activation = 'softmax'))
    # Importing the weights for the model if the file containing the weights for the pre-trained model
    # is available
    if weights_path:
        model.load_weights(weights_path)
    return model

#%% Recognizing Cats Using a VGG16 Network
image = cv2.resize(cv2.imread('Images/cat.jpg'), (224, 224)).astype(np.float32)
image = np.expand_dims(image, axis = 0)

tf.keras.applications.vgg16.VGG16(include_top = True, weights = 'imagenet', input_tensor = None,
                                  input_shape = None, pooling = None, classes = 1000)
VGG16_pretrained = VGG16('vgg16_weights_tf_dim_ordering_tf_kernels.h5')
VGG16_pretrained.summary()
VGG16_pretrained.compile(optimizer = 'sgd', loss = 'categorical_crossentropy')
prediction = VGG16_pretrained.predict(image)
print(np.argmax(prediction))

#%% Using Keras' Built-In VGG16 Network
VGG16_keras = applications.vgg16.VGG16(weights = 'imagenet', include_top = True)
VGG16_keras.compile(optimizer = 'sgd', loss = 'categorical_crossentropy')

test_image = cv2.resize(cv2.imread('Images/steam-locomotive.jpg'), (224, 224))
test_image = np.expand_dims(test_image, axis = 0)
test_prediction = VGG16_keras.predict(test_image)
print(np.argmax(test_prediction))
plt.plot(test_prediction.ravel())

#%% Recycling A Pre-Trained VGG16 Network For Feature Extraction
for i, layer in enumerate(VGG16_keras.layers):
    print(i, layer.name, layer.output_shape)

# Extract features from the block4_pool block
block = models.Model(inputs = VGG16_keras.input, outputs = VGG16_keras.get_layer('block4_pool').output)
cat_image = preprocessing.image.load_img('Images/cat.jpg', target_size = (224, 224))
cat_image = np.expand_dims(preprocessing.image.img_to_array(cat_image), axis = 0)
cat_image = applications.vgg16.preprocess_input(cat_image)
features = block.predict(cat_image)
print(features)