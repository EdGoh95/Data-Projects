#!/usr/bin/env python3
"""
Deep Learning With TensorFlow and Keras Third Edition (Packt Publishing)
Chapter 5: Recurrent Neural Networks
"""
import os
import re
import numpy as np
import tensorflow as tf

def download_and_read(urls):
    texts = []
    for index, url in enumerate(urls):
        p = tf.keras.utils.get_file('ex1-{}.txt'.format(index), url, cache_subdir = 'Data', cache_dir = '.')
        text = open(p, 'r').read()
        # Removing the byte order marks
        text = text.replace('\ufeff', '')
        # Removing new lines
        text = text.replace('\n', ' ')
        text = re.sub(r'\s+', ' ', text)
        texts.extend(text)
    return texts

gutenberg_texts = download_and_read(['https://www.gutenberg.org/cache/epub/28885/pg28885.txt',
                                      'https://www.gutenberg.org/files/12/12-0.txt'])

# Create the vocabulary
vocab = sorted(set(gutenberg_texts))
vocab_size = len(vocab)
print('Vocabulary Size:', vocab_size)
# Create a mapping from characters to indices
char2idx = {c: i for i, c in enumerate(vocab)}
idx2char = {i: c for c, i in char2idx.items()}

texts_as_ints = np.array([char2idx[c] for c in gutenberg_texts])
gutenberg_data = tf.data.Dataset.from_tensor_slices(texts_as_ints)
sequences = gutenberg_data.batch(101, drop_remainder = True)

def split_train_labels(sequence):
    input_sequence = sequence[0:-1]
    output_sequence = sequence[1:]
    return input_sequence, output_sequence

sequences = sequences.map(split_train_labels)
gutenberg_dataset = sequences.shuffle(10000).batch(64, drop_remainder = True)

class CharGenModel(tf.keras.Model):
    def __init__(self, vocab_size, num_timesteps, embedding_dimension, **kwargs):
        super(CharGenModel, self).__init__(**kwargs)
        self.embedding_layer = tf.keras.layers.Embedding(vocab_size, embedding_dimension)
        self.rnn_layer = tf.keras.layers.GRU(num_timesteps, recurrent_initializer = 'glorot_uniform',
                                             recurrent_activation = 'sigmoid', stateful = True,
                                             return_sequences = True)
        self.dense_layer = tf.keras.layers.Dense(vocab_size)

    def call(self, x):
        x = self.embedding_layer(x)
        x = self.rnn_layer(x)
        x = self.dense_layer(x)
        return x

def loss(labels, predictions):
    return tf.losses.sparse_categorical_crossentropy(labels, predictions, from_logits = True)

gutenberg_model = CharGenModel(vocab_size, 100, 256)
gutenberg_model.build(input_shape = (64, 100))
gutenberg_model.summary()
gutenberg_model.compile(optimizer = tf.optimizers.Adam(), loss = loss)

def generate_text(model, prefix_string, char2idx, idx2char, num_chars_to_generate = 1000, temperature = 1.0):
    input = tf.expand_dims([char2idx[s] for s in prefix_string], axis = 0)
    generated_text = []
    model.reset_states()
    for i in range(num_chars_to_generate):
        prediction = tf.squeeze(model(input), axis = 0)/temperature
        # Predict the character returned by the model
        prediction_id = tf.random.categorical(prediction, num_samples = 1)[-1, 0].numpy()
        generated_text.append(idx2char[prediction_id])
        # Pass the prediction as the next input to the model
        input = tf.expand_dims([prediction_id], 0)
    return prefix_string + ''.join(generated_text)

for j in range(50//10):
    gutenberg_model.fit(gutenberg_dataset.repeat(), epochs = 10,
                        steps_per_epoch = len(gutenberg_texts)//100//64)
    checkpoint_file = os.path.join('Data/checkpoints', 'model_epoch_{}'.format(j+1))
    gutenberg_model.save_weights(checkpoint_file)
    generative_model = CharGenModel(vocab_size, 100, 256)
    generative_model.load_weights(checkpoint_file)
    generative_model.build(input_shape = (1, 100))
    print('After epoch {}:'.format((j+1)*5))
    print(generate_text(generative_model, 'Alice ', char2idx, idx2char))
    print('\u2500'*110)