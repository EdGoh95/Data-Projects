#!/usr/bin/env python3
"""
Deep Learning With TensorFlow and Keras Third Edition (Packt Publishing)
Chapter 5: Recurrent Neural Networks
"""
import os
import numpy as np
import re
import tensorflow as tf
import unicodedata
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction

def preprocess_sentence(sentence):
    sentence = ''.join([char for char in unicodedata.normalize('NFD', sentence)
                        if unicodedata.category(char) != 'Mn'])
    sentence = re.sub(r'([!.?])', r'\1', sentence)
    sentence = re.sub(r'[^a-zA-Z!.?]+', r' ', sentence)
    sentence = re.sub(r'\s+', ' ', sentence)
    sentence = sentence.lower()
    return sentence

def download_and_read():
    english_sentences = []
    french_sentences_in = []
    french_sentences_out = []
    local_file = os.path.join('Data/fra-eng', 'fra.txt')
    with open(local_file, 'r') as file_in:
        for index, line in enumerate(file_in):
            english_sentence = line.strip().split('\t')[0]
            french_sentence = line.strip().split('\t')[1]
            english_sentence = [word for word in preprocess_sentence(english_sentence).split()]
            english_sentences.append(english_sentence)
            french_sentence = preprocess_sentence(french_sentence)
            french_sentence_in = [word for word in ('BOS ' + french_sentence).split()]
            french_sentences_in.append(french_sentence_in)
            french_sentence_out = [word for word in (french_sentence + ' EOS').split()]
            french_sentences_out.append(french_sentence_out)
            if index >= 30000 - 1:
                break
    return english_sentences, french_sentences_in, french_sentences_out

english_sentences, french_sentences_in, french_sentences_out = download_and_read()

english_tokenizer = tf.keras.preprocessing.text.Tokenizer(filters = '', lower = False)
english_tokenizer.fit_on_texts(english_sentences)
english_dataset = english_tokenizer.texts_to_sequences(english_sentences)
english_dataset = tf.keras.preprocessing.sequence.pad_sequences(english_dataset, padding = 'post')

french_tokenizer = tf.keras.preprocessing.text.Tokenizer(filters = '', lower = False)
french_tokenizer.fit_on_texts(french_sentences_in)
french_tokenizer.fit_on_texts(french_sentences_out)
french_dataset_in = french_tokenizer.texts_to_sequences(french_sentences_in)
french_dataset_in = tf.keras.preprocessing.sequence.pad_sequences(french_dataset_in, padding = 'post')
french_dataset_out = french_tokenizer.texts_to_sequences(french_sentences_out)
french_dataset_out = tf.keras.preprocessing.sequence.pad_sequences(french_dataset_out, padding = 'post')

english_vocab_size = len(english_tokenizer.word_index)
english_word2idx = english_tokenizer.word_index
english_idx2word = {index: word for word, index in english_word2idx.items()}
french_vocab_size = len(french_tokenizer.word_index)
french_word2idx = french_tokenizer.word_index
french_idx2word = {index: word for word, index in french_word2idx.items()}
print('Vocab size: {} (English), {} (French)'.format(english_vocab_size, french_vocab_size))

english_max_sentence_len = english_dataset.shape[1]
french_max_sentence_len = french_dataset_out.shape[1]
print('Max sentence length: {} (English), {} (French)'.format(english_max_sentence_len, french_max_sentence_len))

combined_dataset = tf.data.Dataset.from_tensor_slices((english_dataset, french_dataset_in,
                                                       french_dataset_out)).shuffle(10000)
test_dataset = combined_dataset.take(30000//4).batch(64, drop_remainder = True)
training_dataset = combined_dataset.skip(30000//4).batch(64, drop_remainder = True)

class BahdanauAttention(tf.keras.layers.Layer):
    def __init__(self, num_units):
        super(BahdanauAttention, self).__init__()
        self.W1 = tf.keras.layers.Dense(num_units)
        self.W2 = tf.keras.layers.Dense(num_units)
        self.V = tf.keras.layers.Dense(1)

    def call(self, query, values):
        '''
        query: Decoder state at time step j (query.shape: (batch_size, num_units))
        value: Encoder states at every time step i (values.shape: batch_size, num_timesteps, num_units)
        Add time axis to query: (batch_size, 1, num_units)
        '''
        query_with_time_axis = tf.expand_dims(query, axis = 1)
        score = self.V(tf.keras.activations.tanh(self.W1(values) + self.W2(query_with_time_axis)))
        alignment = tf.nn.softmax(score, axis = 1)
        # Compute the attended output
        context = tf.reduce_sum(tf.linalg.matmul(tf.linalg.matrix_transpose(alignment), values),
                                axis = 1)
        context = tf.expand_dims(context, axis = 1)
        return context, alignment

class LuongAttention(tf.keras.layers.Layer):
    def __init__(self, num_units):
        super(LuongAttention, self).__init__()
        self.W = tf.keras.layers.Dense(num_units)

    def call(self, query, values):
        # Add time axis to query
        query_with_time_axis = tf.expand_dims(query, axis = 1)
        score = tf.linalg.matmul(query_with_time_axis, self.W(values), transpose_b = True)
        alignment = tf.nn.softmax(score, axis = 2)
        context = tf.matmul(alignment, values)
        return context, alignment

class Encoder(tf.keras.Model):
    def __init__(self, vocab_size, num_timesteps, embedding_dimension, encoder_dimension, **kwargs):
        super(Encoder, self).__init__(**kwargs)
        self.encoder_dimension = encoder_dimension
        self.embedding = tf.keras.layers.Embedding(vocab_size, embedding_dimension, input_length = num_timesteps)
        self.rnn = tf.keras.layers.GRU(encoder_dimension, return_sequences = True, return_state = True)

    def call(self, x, state):
        x = self.embedding(x)
        x, state = self.rnn(x, initial_state = state)
        return x, state

    def init_state(self, batch_size):
        return tf.zeros((batch_size, self.encoder_dimension))

class Decoder(tf.keras.Model):
    def __init__(self, vocab_size, num_timesteps, embedding_dimension, decoder_dimension, **kwargs):
        super(Decoder, self).__init__(**kwargs)
        self.decoder_dimension = decoder_dimension
        # self.attention = BahdanauAttention(embedding_dimension)
        self.attention = LuongAttention(embedding_dimension)
        self.embedding = tf.keras.layers.Embedding(vocab_size, embedding_dimension, input_length = num_timesteps)
        self.rnn = tf.keras.layers.GRU(decoder_dimension, return_sequences = True, return_state = True)
        self.Wc = tf.keras.layers.Dense(decoder_dimension, activation = 'tanh')
        self.Ws = tf.keras.layers.Dense(vocab_size)

    def call(self, x, state, encoder_out):
        x = self.embedding(x)
        context, alignment = self.attention(x, encoder_out)
        x = tf.expand_dims(tf.concat([x, tf.squeeze(context, axis = 1)], axis = 1), axis = 1)
        x, state = self.rnn(x, state)
        x = self.Wc(x)
        x = self.Ws(x)
        return x, state, alignment

encoder = Encoder(english_vocab_size + 1, 256, english_max_sentence_len, 1024)
decoder = Decoder(french_vocab_size + 1, 256, french_max_sentence_len, 1024)

for encoder_in, decoder_in, decoder_out in training_dataset:
    encoder_state = encoder.init_state(64)
    encoder_out, encoder_state = encoder(encoder_in, encoder_state)
    decoder_state = encoder_state
    decoder_predictions = []
    decoder_state = decoder_state
    for t in range(decoder_out.shape[1]):
        decoder_in_t = decoder_in[:, t]
        decoder_prediction_t, decoder_state, _ = decoder(decoder_in_t, decoder_state, encoder_out)
        decoder_predictions.append(decoder_prediction_t.numpy())
    decoder_predictions = tf.squeeze(np.array(decoder_predictions), axis = 2)
    break

print('Encoder Input:', encoder_in.shape)
print('Encoder Output: {}, Encoder State: {}'.format(encoder_out.shape, encoder_state.shape))
print('Decoder Output (Logits): {}, Decoder State: {}'.format(decoder_predictions.shape, decoder_state.shape))
print('Decoder Output (Labels):', decoder_out.shape)

query = np.random.random(size = (64, 1024))
values = np.random.random(size = (64, 8, 1024))
Bahdanau_attn = BahdanauAttention(1024)
Bahdanau_context, Bahdanau_alignments = Bahdanau_attn(query, values)
print('Bahdanau Attention - Context Shape: {}, Alignment Shape: {}'.format(Bahdanau_context.shape,
                                                                           Bahdanau_alignments.shape))
luong_attn = LuongAttention(1024)
luong_context, luong_alignments = luong_attn(query, values)
print('Luong Attention - Context Shape: {}, Alignment Shape: {}'.format(luong_context.shape, luong_alignments.shape))

def loss_fn(label, prediction):
    loss_function = tf.keras.losses.SparseCategoricalCrossentropy(from_logits = True)
    mask = tf.cast(tf.math.logical_not(tf.math.equal(label, 0)), dtype = tf.int64)
    loss = loss_function(label, prediction, sample_weight = mask)
    return loss

@tf.function
def train_step(encoder_in, decoder_in, decoder_out, encoder_state):
    with tf.GradientTape() as tape:
        encoder_out, encoder_state = encoder(encoder_in, encoder_state)
        decoder_state = encoder_state
        loss = 0
        for t in range(decoder_out.shape[1]):
            decoder_in_t = decoder_in[:, t]
            decoder_prediction_t, decoder_state, _ = decoder(decoder_in_t, decoder_state, encoder_out)
            loss += loss_fn(decoder_out[:, t], decoder_prediction_t)

    variables = encoder.trainable_variables + decoder.trainable_variables
    gradients = tape.gradient(loss, variables)
    optimizer.apply_gradients(zip(gradients, variables))
    return loss/decoder_out.shape[1]

def predict(encoder, decoder, batch_size, english_sentences, english_dataset, french_sentences_out,
            french_word2idx, french_idx2word):
    random_id = np.random.choice(len(english_sentences))
    print('Input:', ' '.join(english_sentences[random_id]))
    print('Label:', ' '.join(french_sentences_out[random_id]))

    encoder_in = tf.expand_dims(english_dataset[random_id], axis = 0)
    decoder_out = tf.expand_dims(french_sentences_out[random_id], axis = 0)

    encoder_state = encoder.init_state(1)
    encoder_out, encoder_state = encoder(encoder_in, encoder_state)
    decoder_state = encoder_state
    decoder_in = tf.expand_dims(tf.constant(french_word2idx['BOS']), axis = 0)

    french_sentence_prediction = []
    while True:
        decoder_prediction, decoder_state, _ = decoder(decoder_in, decoder_state, encoder_out)
        decoder_prediction = tf.argmax(decoder_prediction, axis = -1)
        predicted_word = french_idx2word[decoder_prediction.numpy()[0][0]]
        french_sentence_prediction.append(predicted_word)
        if predicted_word == 'EOS':
            break
        decoder_in = tf.squeeze(decoder_prediction, axis = 1)

    print('Prediction:', ' '.join(french_sentence_prediction))

def evaluate_bleu_score(encoder, decoder, test_dataset, french_word2idx, french_idx2word):
    bleu_scores = []
    smooth_fn = SmoothingFunction()
    for encoder_in, decoder_in, decoder_out in test_dataset:
        encoder_state = encoder.init_state(64)
        encoder_out, encoder_state = encoder(encoder_in, encoder_state)
        decoder_state = encoder_state

        reference_sentence_ids = np.zeros_like(decoder_out)
        candidate_sentence_ids = np.zeros_like(decoder_out)
        for t in range(decoder_out.shape[1]):
            decoder_in_t = decoder_in[:, t]
            decoder_out_t = decoder_out[:, t]
            decoder_prediction_t, decoder_state, _ = decoder(decoder_in_t, decoder_state, encoder_out)
            decoder_prediction_t = tf.argmax(decoder_prediction_t, axis = -1)
            for b in range(decoder_prediction_t.shape[0]):
                reference_sentence_ids[b, t] = decoder_out_t.numpy()[0]
                candidate_sentence_ids[b, t] = decoder_prediction_t.numpy()[0][0]

        for i in range(reference_sentence_ids.shape[0]):
            reference_sentence = [french_idx2word[j] for j in reference_sentence_ids[i] if j > 0]
            candidate_sentence = [french_idx2word[k] for k in candidate_sentence_ids[i] if k > 0]
            # Remove any trailing EOS
            reference_sentence = reference_sentence[:-1]
            candidate_sentence = candidate_sentence[:-1]
            bleu_score = sentence_bleu([reference_sentence], candidate_sentence,
                                       smoothing_function = smooth_fn.method1)
            bleu_scores.append(bleu_score)
    return np.mean(np.array(bleu_scores))

optimizer = tf.keras.optimizers.Adam()
checkpoint_prefix = os.path.join('Data', 'checkpoints')
checkpoint = tf.train.Checkpoint(optimizer = optimizer, encoder = encoder, decoder = decoder)
for epoch in range(50):
    encoder_state = encoder.init_state(64)
    for batch, data in enumerate(training_dataset):
        encoder_in, decoder_in, decoder_out = data
        loss = train_step(encoder_in, decoder_in, decoder_out, encoder_state)

    print('Epoch {}: Loss = {:.4f}'.format(epoch + 1, loss.numpy()))
    if epoch % 10 == 0:
        checkpoint.save(file_prefix = checkpoint_prefix)

    predict(encoder, decoder, 64, english_sentences, english_dataset, french_sentences_out,
            french_word2idx, french_idx2word)
    evaluation_score = evaluate_bleu_score(encoder, decoder, test_dataset, french_word2idx,
                                           french_idx2word)
    print('Evaluation Score (BLEU): {:.3e}'.format(evaluation_score))
    print('\u2500'*120)

checkpoint.save(file_prefix = checkpoint_prefix)