#!/usr/bin/env python3
"""
Deep Learning With TensorFlow and Keras Third Edition (Packt Publishing) Chapter 4: Word Embeddings
"""
import os
import gensim.downloader as api
import numpy as np
import tensorflow as tf
from gensim.models import Word2Vec, KeyedVectors
from sklearn.metrics import accuracy_score, confusion_matrix

#%% Downloading A Pre-Trained Word2Vec Model Using Gensim
# text8_dataset = api.load('text8')
# word2vec_model = Word2Vec(text8_dataset)
# word2vec_model.save('Data/text8-word2vec.bin')
print('Pre-trained embedding models in gensim:', list(api.info()['models'].keys()), sep = '\n')
print('\u2500'*110)

#%% Exploring The Word2Vec Embedding Space With Gensim
print('Exploring Word2Vec Embedding Space...')
word2vec_model = KeyedVectors.load('Data/text8-word2vec.bin')
word_vectors = word2vec_model.wv
words = word_vectors.index_to_key
print('First 10 words in the corpus:', [word for index, word in enumerate(words) if index < 10])
assert('king' in words)
print('\u2500'*110)

def print_most_similar(word_conf_pairs, k):
    for index, (word, conf) in enumerate(word_conf_pairs):
        print('{:.3f} {:s}'.format(conf, word))
        if index >=k-1:
            break
    if k < len(word_conf_pairs):
        print('\u2500'*110)

print_most_similar(word_vectors.most_similar('king'), 5)
print_most_similar(word_vectors.most_similar_cosmul(positive = ['france', 'berlin'], negative = ['paris']), 1)
print(word_vectors.doesnt_match(['hindus', 'buddhists', 'singapore', 'christians']))
print('\u2500'*110)

for word in ['woman', 'dog', 'whale', 'tree']:
    print('Similarity({}, {}) = {:.3f}'.format('man', word, word_vectors.similarity('man', word)))
print('\u2500'*110)

print_most_similar(word_vectors.similar_by_word('singapore'), 5)
print('Distance(singapore, malaysia) = {:.3f}'.format(word_vectors.distance('singapore', 'malaysia')))

song_vec = word_vectors['song']
song_vec2 = word_vectors.get_vector('song', norm = True)
print('\u2500'*110)

#%% Using Word Embeddings For Spam Detection
print('Using Word Embeddings For Spam Detection...')
def download_and_read(url):
    local_file = url.split('/')[-1]
    p = tf.keras.utils.get_file(local_file, url, extract = True, cache_dir = '.')
    labels = []
    texts = []
    local_file = os.path.join('datasets', 'SMSSpamCollection')
    with open(local_file, 'r') as file:
        for line in file:
            label, text = line.strip().split('\t')
            labels.append(1 if label == 'spam' else 0)
            texts.append(text)
    return texts, labels

SMS_texts, SMS_labels = download_and_read('https://archive.ics.uci.edu/ml/machine-learning-databases/00228/smsspamcollection.zip')

SMS_tokenizer = tf.keras.preprocessing.text.Tokenizer()
SMS_tokenizer.fit_on_texts(SMS_texts)
SMS_text_sequences = SMS_tokenizer.texts_to_sequences(SMS_texts)
SMS_text_sequences = tf.keras.preprocessing.sequence.pad_sequences(SMS_text_sequences)
num_records = len(SMS_text_sequences)
max_sentence_len = len(SMS_text_sequences[0])
print('Number of sentences: {}, Max Length: {}'.format(num_records, max_sentence_len))

categories = tf.keras.utils.to_categorical(SMS_labels, num_classes = 2)

# Obtaining a dictionary of vocabulary words from their index positions from the vocabulary
word2idx = SMS_tokenizer.word_index
idx2word = {word: index for word, index in word2idx.items()}
word2idx['PAD'] = 0
idx2word[0] = 'PAD'
vocab_size = len(word2idx)
print('Size of vocabulary: ', vocab_size)

SMS_dataset = tf.data.Dataset.from_tensor_slices((SMS_text_sequences, categories)).shuffle(10000)
test_size = num_records//4
test_dataset = SMS_dataset.take(test_size)
validation_size = (num_records - test_size)//10
validation_dataset = SMS_dataset.skip(test_size).take(validation_size)
training_dataset = SMS_dataset.skip(test_size + validation_size)
test_dataset = test_dataset.batch(128, drop_remainder = True)
validation_dataset = validation_dataset.batch(128, drop_remainder = True)
training_dataset = training_dataset.batch(128, drop_remainder = True)

# Building the embedding matrix
def build_embedding_matrix(sequences, word2idx, embedding_model, embedding_dimension, embedding_file):
    if os.path.exists(embedding_file):
        embedding = np.load(embedding_file)
    else:
        vocab_size = len(word2idx)
        embedding = np.zeros((vocab_size, embedding_dimension))
        word_vectors = api.load(embedding_model)
        for word, index in word2idx.items():
            try:
                embedding[index] = word_vectors.word_vec(word)
            except KeyError:
                # When word is not present in the embedding_model
                pass
        np.save(embedding_file, embedding)
    return embedding

embedding_matrix = build_embedding_matrix(SMS_text_sequences, word2idx, 'glove-wiki-gigaword-300',
                                          300, os.path.join('Data', 'Embedding.npy'))
print('Shape of embedding matrix:', embedding_matrix.shape)

# Defining the sapm classifier
class SpamClassifier(tf.keras.Model):
    def __init__(self, vocab_size, embedding_size, input_length, num_filters, kernel_size, output_size,
                 run_mode, embedding_weights, **kwargs):
        super(SpamClassifier, self).__init__(**kwargs)
        if run_mode == 'scratch':
            self.embedding = tf.keras.layers.Embedding(
                vocab_size, embedding_size, input_length = input_length, trainable = True)
        elif run_mode == 'vectorizer':
            self.embedding = tf.keras.layers.Embedding(
                vocab_size, embedding_size, input_length = input_length, weights = [embedding_weights],
                trainable = False)
        else:
            self.embedding = tf.keras.layers.Embedding(
                vocab_size, embedding_size, input_length = input_length, weights = [embedding_weights],
                trainable = True)
        self.convolution = tf.keras.layers.Convolution1D(filters = num_filters, kernel_size = kernel_size,
                                                         activation = 'relu')
        self.dropout = tf.keras.layers.SpatialDropout1D(0.2)
        self.pool = tf.keras.layers.GlobalMaxPooling1D()
        self.dense = tf.keras.layers.Dense(output_size, activation = 'softmax')

    def call(self, x):
        x = self.embedding(x)
        x  =self.convolution(x)
        x = self.dropout(x)
        x = self.pool(x)
        x = self.dense(x)
        return x

spam_classifier = SpamClassifier(vocab_size, 300, max_sentence_len, 256, 3, 2, 'finetune', embedding_matrix)
spam_classifier.build(input_shape = (None, max_sentence_len))
spam_classifier.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['accuracy'])
spam_classifier.summary()

spam_classifier.fit(training_dataset, epochs = 3, validation_data = validation_dataset,
                    class_weight = {0: 1, 1: 8})

labels = []
predictions = []
for SMS_text_test, SMS_label_test in test_dataset:
    predicted_label = spam_classifier.predict_on_batch(SMS_text_test)
    labels.extend(np.argmax(SMS_label_test, axis = 1).tolist())
    predictions.extend(np.argmax(predicted_label, axis = 1).tolist())

print('Test accuracy:', accuracy_score(labels, predictions))
print('Confusion matrix:', confusion_matrix(labels, predictions), sep = '\n')