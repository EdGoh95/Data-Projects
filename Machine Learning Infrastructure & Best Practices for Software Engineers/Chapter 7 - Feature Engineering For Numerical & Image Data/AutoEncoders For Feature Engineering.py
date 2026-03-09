#!/usr/bin/env python3
"""
Machine Learning Infrastructure And Best Practices For Software Engineers (Packt Publishing)
Chapter 7: Feature Engineering for Numerical and Image Data
"""
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, BatchNormalization, LeakyReLU, Flatten, Reshape
from tensorflow.keras.datasets import mnist

ant_df = pd.read_excel("../Chapter 6 - Processing Data In Machine Learning Systems/Regenerated PROMISE and BPD Datasets.xlsx",
                       sheet_name = 'ant_1_3', index_col = 0)
ant_features = ant_df.drop(['Defect'], axis = 1)
ant_target = ant_df['Defect']
ant_features_train, ant_features_test, ant_target_train, ant_target_test = train_test_split(
    ant_features, ant_target, test_size = 0.33, random_state = 1)

# Scaling the data using MinMax Scaler
scaler = MinMaxScaler().fit(ant_features_train)
ant_features_train_scaled = scaler.transform(ant_features_train)
ant_features_test_scaled = scaler.transform(ant_features_test)

num_cols = ant_features.shape[1]
input_layer = Input(shape = (num_cols,))

#%% Encoders
#### First Encoder Layer
encoder = Dense(num_cols * 2)(input_layer)
encoder = BatchNormalization()(encoder)
encoder = LeakyReLU()(encoder)

#### Second Encoder Layer
encoder = Dense(num_cols)(encoder)
encoder = BatchNormalization()(encoder)
encoder = LeakyReLU()(encoder)

#%% Bottleneck
bottleneck = Dense(3, name = 'bottleneck')(encoder)

#%% Decoders
#### First Decoder Layer
decoder = Dense(num_cols)(bottleneck)
decoder = BatchNormalization()(decoder)
decoder = LeakyReLU()(decoder)

#### Second Decoder Layer
decoder = Dense(num_cols * 2)(decoder)
decoder = BatchNormalization()(decoder)
decoder = LeakyReLU()(decoder)

output_layer = Dense(num_cols, activation = 'linear')(decoder)

ant_model = Model(inputs = input_layer, outputs = output_layer)
ant_model.compile(optimizer = 'adam', loss = 'mse')
ant_model_history = ant_model.fit(ant_features_train_scaled, ant_features_train_scaled, epochs = 100,
                                  batch_size = 16, verbose = 2,
                                  validation_data = (ant_features_test_scaled, ant_features_test_scaled))

ant_bottleneck_layer = Model(ant_model.inputs, ant_model.get_layer('bottleneck').output)
ant_bottleneck_values = ant_bottleneck_layer.predict(ant_features_train_scaled)

#%% Feature Engineering For Image Data
(MNIST_images_train, MNIST_labels_train), (MNIST_images_test, MNIST_labels_test) = mnist.load_data()
MNIST_images_train = MNIST_images_train/255.0
MNIST_images_test = MNIST_images_test/255.0

num_pixels = MNIST_images_train.shape[1]
MNIST_input_layer = Input(shape = (num_pixels, num_pixels,))

#### First Encoder Layer
MNIST_encoder = Flatten(input_shape = (num_pixels, num_pixels))(MNIST_input_layer)
MNIST_encoder = LeakyReLU()(MNIST_encoder )
MNIST_encoder = Dense(num_pixels * 2)(MNIST_encoder)
MNIST_encoder = BatchNormalization()(MNIST_encoder)
MNIST_encoder = LeakyReLU()(MNIST_encoder)

#### Second Encoder Layer
MNIST_encoder = Dense(num_pixels)(MNIST_encoder)
MNIST_encoder = BatchNormalization()(MNIST_encoder)
MNIST_encoder = LeakyReLU()(MNIST_encoder)

#### Bottleneck
MNIST_bottleneck = Dense(32, name = 'MNIST_bottleneck')(MNIST_encoder)

#### First Decoder Layer
MNIST_decoder = Dense(num_pixels)(MNIST_bottleneck)
MNIST_decoder = BatchNormalization()(MNIST_decoder)
MNIST_decoder = LeakyReLU()(MNIST_decoder)

#### Second Decoder Layer
MNIST_decoder = Dense(num_pixels * 2)(MNIST_decoder)
MNIST_decoder = BatchNormalization()(MNIST_decoder)
MNIST_decoder = LeakyReLU()(MNIST_decoder)

MNIST_output_layer = Dense(num_pixels * num_pixels, activation = 'linear')(MNIST_decoder)
MNIST_output = Reshape((num_pixels, num_pixels))(MNIST_output_layer)

MNIST_model = Model(inputs = MNIST_input_layer, outputs = MNIST_output)
MNIST_model.compile(optimizer = 'adam', loss = 'mse')
MNIST_model_history = MNIST_model.fit(MNIST_images_train, MNIST_images_train, epochs = 100,
                                      batch_size = 16, verbose = 2,
                                      validation_data = (MNIST_images_test, MNIST_images_test))

MNIST_bottleneck_layer = Model(MNIST_model.inputs, MNIST_bottleneck)
MNIST_bottleneck_values = MNIST_bottleneck_layer.predict(MNIST_images_train)