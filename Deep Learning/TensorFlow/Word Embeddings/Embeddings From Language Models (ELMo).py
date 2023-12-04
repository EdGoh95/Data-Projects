#!/usr/bin/env python3
"""
Deep Learning With TensorFlow and Keras Third Edition (Packt Publishing) Chapter 4: Word Embeddings
"""
import tensorflow as tf
import tensorflow_hub as tfhub

#%% Dynamic Embeddings
elmo = tfhub.load('https://tfhub.dev/google/elmo/3')
text_inputs = ['i like green eggs and ham', 'would you eat them in a box']
elmo_embeddings = elmo.signatures['default'](tf.constant(text_inputs))['elmo']
print(elmo_embeddings.shape)
print('\u2500'*110)

elmo_embedding_layer = tfhub.KerasLayer('https://tfhub.dev/google/elmo/3', input_shape = [],
                                        dtype = tf.string)
model = tf.keras.Sequential([elmo_embedding_layer])
adapted_embeddings = model.predict(text_inputs)
print(adapted_embeddings.shape)
print('\u2500'*110)

#%% Sentence and Paragraph Embeddings
universal_sentence_encoder = tfhub.load('https://tfhub.dev/google/universal-sentence-encoder-large/4')
sentence_embeddings = universal_sentence_encoder(text_inputs)['outputs']
print(sentence_embeddings.shape)
print('\u2500'*110)