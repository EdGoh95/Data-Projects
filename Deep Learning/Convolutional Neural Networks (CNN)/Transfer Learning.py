#!/usr/bin/env python3
"""
Deep Learning With TensorFlow And Keras Third Edition (Packt Publishing)
Chapter 3: Convolutional Neural Networks
"""
import tensorflow as tf
from tensorflow.keras import layers, applications, models, optimizers

#%% Deep Inception V3
InceptionV3_pretrained = applications.inception_v3.InceptionV3(weights = 'imagenet', include_top = False)

# Use the final fully-connected layer as the first layer of the new model
first_layer = layers.Dense(1024, 'relu')(InceptionV3_pretrained.output)
predictions = layers.Dense(200, activation = 'softmax')(first_layer)
new_model = models.Model(inputs = InceptionV3_pretrained.input, outputs = predictions)

for layer in InceptionV3_pretrained.layers[:172]:
    layer.trainable = False
for layer in InceptionV3_pretrained.layers[172:]:
    layer.trainable = True

InceptionV3_pretrained.compile(optimizer = optimizers.SGD(lr = 1e-4, momentum = 0.9),
                               loss = 'categorical_crossentropy')

#%% Style Transfer
def get_content_loss(base_content, target):
    '''
    Calculate the content distance (the distance in the feature space defined by a layer l for a
    VGGG19 or any suitable network receiving the 2 images as input)
    '''
    return tf.reduce_mean(tf.square(base_content - target))

def gram_matrix(input_tensor):
    channels = int(input_tensor.shape[-1])
    a = tf.reshape(input_tensor, [-1, channels])
    n = tf.shape(a)[0]
    gram = tf.matmul(a, a, transpose_a = True)
    return gram/tf.cast(n, tf.float32)

def get_style_loss(base_style, gram_target):
    '''
    Calculate the style distance (total style loss across levels)
    '''
    # filters of each layer
    height, width, channels = base_style.get_shape().as_list()
    gram_style = gram_matrix(base_style)
    return tf.reduce_mean(tf.square(gram_style - gram_target))