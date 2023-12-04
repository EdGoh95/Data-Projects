#!/usr/bin/env python3
"""
Deep Learning With TensorFlow and Keras Third Edition (Packt Publishing) Chapter 8: AutoEncoders
"""
import collections
import re
import numpy as np
import nltk
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from tensorflow.keras import preprocessing, models, layers
# nltk.download('punkt')
# nltk.download('reuters')

def is_number(n):
    return re.sub('[.,-/]', ' ', n).isdigit()

# Parsing sentences and building vocabulary
word_frequencies = collections.Counter()
documents = nltk.corpus.reuters.fileids()
sentences = []
sentence_lens = []
num_reads = 1
for i in range(len(documents)):
    if num_reads % 100 == 0:
        print('Building features from {} documents'.format(num_reads))
    # Skipping documents without a specified topic
    title = nltk.corpus.reuters.raw(documents[i]).lower()
    if len(title) == 0:
        continue
    num_reads += 1
    # Convert to list of word indices
    title = re.sub('\n', '', title)
    for sentence in nltk.sent_tokenize(title):
        for word in nltk.word_tokenize(sentence):
            if is_number(word):
                word = '9'
            word = word.lower()
            word_frequencies[word] += 1
        sentences.append(sentence)
        sentence_lens.append(len(sentence))

print('Total number of sentences in reuters corpus:', len(sentences))
print('Distribution of sentence lengths throughout the corpus: Min = {}, Max = {}, Mean = {:.3f}, Median = {:.3f}'.format(
    np.min(sentence_lens), np.max(sentence_lens), np.mean(sentence_lens), np.median(sentence_lens)))
print('Vocab Size (Full):', len(word_frequencies))

word2idx = {}
word2idx['PAD'] = 0
word2idx['UNK'] = 1
for count, (word, _) in enumerate(word_frequencies.most_common(5000 - 2)):
    word2idx[word] = count + 2
idx2word = {index: word for word, index in word2idx.items()}

def lookup_word2idx(word):
    try:
        return word2idx[word]
    except KeyError:
        return word2idx['UNK']

def load_glove_vectors(glove_file, word2idx, embedding_size):
    embedding = np.zeros((len(word2idx), embedding_size))
    for line in open(glove_file, 'rb'):
        cols = line.strip().split()
        word = cols[0].decode('utf-8')
        if embedding_size == 0:
            embedding_size = len(cols) - 1
        if word in word2idx:
            vector = np.array([float(v) for v in cols[1:]])
        embedding[lookup_word2idx(word)] = vector
    embedding[word2idx['PAD']] = np.zeros((embedding_size))
    embedding[word2idx['UNK']] = np.random.uniform(-1, 1, 50)
    return embedding

sentence_ids = [[lookup_word2idx(word) for word in s.split()] for s in sentences]
sentence_ids = preprocessing.sequence.pad_sequences(sentence_ids, 50)
# Load the GloVe vectors into the weights matrix
embeddings = load_glove_vectors('Data/glove.6B/glove.6B.{}d.txt'.format(50), word2idx, 50)

def sentence_generator(X, embeddings, batch_size):
    '''
    Produce batches of tensors with shape (batch_size, sentence_len, embedding_size)
    '''
    while True:
        # Loop once per epoch
        indices = np.random.permutation(np.arange(X.shape[0]))
        num_batches = X.shape[0]//batch_size
        for batch_id in range(num_batches):
            sentence_ids = indices[batch_id * batch_size: (batch_id + 1) * batch_size]
            yield embeddings[X[sentence_ids, :]], embeddings[X[sentence_ids, :]]

sentence_ids_train, sentence_ids_test = train_test_split(sentence_ids, train_size = 0.7)
training_generator = sentence_generator(sentence_ids_train, embeddings, 64)
test_generator = sentence_generator(sentence_ids_test, embeddings, 64)

# Defining the LSTM-Based AutoEncoder
inputs = layers.Input(shape = (50, 50), name = 'input')
encoded = layers.Bidirectional(layers.LSTM(512), merge_mode = 'sum', name = 'encoder_LSTM')(inputs)
decoded = layers.RepeatVector(50, name = 'repeater')(encoded)
decoded = layers.Bidirectional(layers.LSTM(50, return_sequences = True), merge_mode = 'sum',
                               name = 'decoder_LSTM')(decoded)
LSTM_autoencoder = models.Model(inputs, decoded)
LSTM_autoencoder.compile(optimizer = 'adam', loss = 'mse')

LSTM_autoencoder_fit = LSTM_autoencoder.fit_generator(
    training_generator, steps_per_epoch = len(sentence_ids_train)//64, epochs = 20,
    validation_data = test_generator, validation_steps = len(sentence_ids_test)//64)

plt.figure()
plt.plot(LSTM_autoencoder_fit.history['loss'], label = 'Training Loss')
plt.plot(LSTM_autoencoder_fit.history['val_loss'], label = 'Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

LSTM_encoder = models.Model(LSTM_autoencoder.input, LSTM_autoencoder.get_layer('encoder_LSTM').output)

def compute_cosine_similarity(x, y):
    return np.dot(x, y)/(np.linalg.norm(x, 2) * np.linalg.norm(y, 2))

cos_sims = np.zeros((500))
j = 0
for batch_id in range(len(sentence_ids_test)//64):
    sentences_test, labels_test = next(test_generator)
    predictions = LSTM_autoencoder.predict(sentences_test)
    sentences_vector = LSTM_encoder.predict(sentences_test)
    prediction_vector = LSTM_encoder.predict(predictions)
    for row_id in range(sentences_vector.shape[0]):
        if j >= 500:
            break
        cos_sims[j] = compute_cosine_similarity(sentences_vector[row_id], prediction_vector[row_id])
        if j <= 10:
            print(cos_sims[j])
        j += 1
    if j >= 500:
        break

plt.figure()
plt.hist(cos_sims, bins = 20, density = True)
plt.xlabel('Cosine Similarity')
plt.ylabel('Counts')