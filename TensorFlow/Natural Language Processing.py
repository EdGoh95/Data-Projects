#!/usr/bin/env python3
"""
TensorFlow Machine Learning Cookbook Second Edition (Packt Publishing) Chapter 7:
Natural Language Processing (NLP)
"""
import string
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
import pickle
from termcolor import colored
from collections import Counter
from tensorflow.python.framework import ops
from tensorflow.keras.preprocessing.text import Tokenizer
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk import word_tokenize
from nltk.corpus import stopwords

start = time.perf_counter()
#%% Bag-Of-Words (BOW) Embeddings
ops.reset_default_graph()
BOW_session= tf.Session()

# Load the SMS Spam Collection dataset
SMS_spam_df = pd.read_csv('Data/SMS Spam Dataset', sep = '\t', header = None,
                          names = ['Label', 'SMS Text'])
# 'spam': 1 and 'ham'(non-spam): 0
SMS_spam_df['Target'] = [1 if label == 'spam' else 0 for label in SMS_spam_df['Label']]

cleaned_text = []
for text in SMS_spam_df['SMS Text']:
    text = text.lower()
    # Remove any punctuations
    text = ''.join(s for s in text if s not in string.punctuation)
    # Remove any numbers
    text = ''.join(s for s in text if s not in '0123456789')
    # Remove any extra whitespaces
    text = ' '.join(text.split())
    cleaned_text.append(text)
SMS_spam_df['SMS Text (Cleaned)'] = cleaned_text

text_length = []
for text in SMS_spam_df['SMS Text (Cleaned)']:
    length = len(text.split())
    if length < 50:
        text_length.append(length)
plt.figure()
plt.hist(text_length, bins = 25, ec = 'k', alpha = 0.9)
plt.title('Histogram of Length of Each SMS')
plt.xlabel('Number of Words In Each SMS')
plt.xlim([0, 50])
plt.ylabel('Frequency')
plt.ylim([0, 900])

vocabulary_processor = Tokenizer(num_words = 25)
vocabulary_processor.fit_on_texts(cleaned_text)
cleaned_text_array = vocabulary_processor.texts_to_sequences(cleaned_text)
embedding_size = len(np.unique(cleaned_text_array))

# Split the dataset into training (80%) and testing (20%) sets
BOW_train_indices = np.random.choice(len(cleaned_text), round(len(cleaned_text) * 0.8),
                                     replace = False)
BOW_test_indices = np.array(list(set(range(len(cleaned_text))) - set(BOW_train_indices)))
train_SMS_text = [text for index, text in enumerate(cleaned_text) if index in BOW_train_indices]
test_SMS_text = [text for index, text in enumerate(cleaned_text) if index in BOW_test_indices]
train_SMS_target = [target for index, target in enumerate(SMS_spam_df['Target'])
                    if index in BOW_train_indices]
test_SMS_target = [target for index, target in enumerate(SMS_spam_df['Target'])
                   if index in BOW_test_indices]

embedding_matrix = tf.diag(tf.ones(shape = [embedding_size]))

A_SMS = tf.Variable(tf.random_normal(shape = [embedding_size, 1]))
b_SMS = tf.Variable(tf.random_normal(shape = [1, 1]))
features = tf.placeholder(shape = [25], dtype = tf.int32)
target = tf.placeholder(shape = [1, 1], dtype = tf.float32)

# One-hot encoding of the indices of the words in the sentence
features_embedding = tf.nn.embedding_lookup(embedding_matrix, features)
# Summing up the columns of the embedding matrix
features_sum = tf.reduce_sum(features_embedding, 0)
features_sum_2D = tf.expand_dims(features_sum, 0)

BOW_output = tf.add(tf.matmul(features_sum_2D, A_SMS), b_SMS)
BOW_loss = tf.reduce_mean(
    tf.nn.sigmoid_cross_entropy_with_logits(logits = BOW_output, labels = target))
BOW_prediction = tf.sigmoid(BOW_output)

BOW_optimizer = tf.train.GradientDescentOptimizer(0.001)
BOW_train_step = BOW_optimizer.minimize(BOW_loss)
init = tf.global_variables_initializer()
BOW_session.run(init)

loss = []
overall_train_accuracy, average_train_accuracy = [], []
vocabulary_processor.fit_on_texts(train_SMS_text)
print(colored('\u2500'*38 + ' Bag-Of-Words (BOW) Embeddings ' + '\u2500'*38, 'red',
              attrs = ['bold']))
print(colored('Starting Training For {} Sentences'.format(len(train_SMS_text)), 'blue'))
for i, text in enumerate(vocabulary_processor.texts_to_matrix(train_SMS_text)):
    BOW_session.run(BOW_train_step, feed_dict = {features: list(text),
                                                 target: [[train_SMS_target[i]]]})
    temp_loss = BOW_session.run(BOW_loss, feed_dict = {features: list(text),
                                                       target: [[train_SMS_target[i]]]})
    loss.append(temp_loss)
    if (i + 1) % 50 == 0:
        print('Training Observation {}: Loss = {:.5e}'.format(i+1, temp_loss))
    [[temp_prediction]] = BOW_session.run(
        BOW_prediction, feed_dict = {features: list(text),
                                     target: [[train_SMS_target[i]]]})
    train_temp_accuracy = train_SMS_target[i] == np.round(temp_prediction)
    overall_train_accuracy.append(train_temp_accuracy)
    if len(overall_train_accuracy) >= 50:
        average_train_accuracy.append(np.mean(overall_train_accuracy[-50:]))

overall_test_accuracy = []
print(colored('\nCalculating Test Accuracy For {} Sentences'.format(len(test_SMS_text)),
              'green'))
for j, text in enumerate(vocabulary_processor.texts_to_matrix(test_SMS_text)):
    if (j + 1) % 50 == 0:
        print('Testing Observation {}'.format(j+1))
    [[temp_prediction]] = BOW_session.run(
        BOW_prediction, feed_dict = {features: list(text), target: [[test_SMS_target[j]]]})
    test_temp_accuracy = test_SMS_target[j] == np.round(temp_prediction)
    overall_test_accuracy.append(test_temp_accuracy)
print('Average Test Accuracy = {:.5f}'.format(np.mean(overall_test_accuracy)))

plt.figure()
plt.plot(range(len(average_train_accuracy)), average_train_accuracy, 'k-')
plt.title('Average Training Accuracy For Every 50 Iterations (Bag-Of-Words)')
plt.xlabel('Iteration')
plt.xlim(left = 0)
plt.ylabel('Accuracy')

#%% Text Frequency - Inverse Document Frequency (TF-IDF)
ops.reset_default_graph()
TFIDF_session = tf.Session()

TFIDF = TfidfVectorizer(tokenizer = word_tokenize, stop_words = 'english',
                        max_features = 1000)
SMS_sparse_array = TFIDF.fit_transform(cleaned_text)
TFIDF_train_indices = np.random.choice(SMS_sparse_array.shape[0],
                                       round(SMS_sparse_array.shape[0] * 0.8),
                                       replace = False)
TFIDF_test_indices = np.array(list(set(range(SMS_sparse_array.shape[0])) -
                                   set(TFIDF_train_indices)))
TFIDF_train_features = SMS_sparse_array[TFIDF_train_indices]
TFIDF_test_features = SMS_sparse_array[TFIDF_test_indices]
TFIDF_train_target = np.array([target for index, target in enumerate(SMS_spam_df['Target'])
                               if index in TFIDF_train_indices])
TFIDF_test_target = np.array([target for index, target in enumerate(SMS_spam_df['Target'])
                              if index in TFIDF_test_indices])

A_TFIDF = tf.Variable(tf.random_normal(shape = [1000, 1]))
b_TFIDF = tf.Variable(tf.random_normal(shape = [1, 1]))
features = tf.placeholder(shape = [None, 1000], dtype = tf.float32)
target = tf.placeholder(shape = [None, 1], dtype = tf.float32)

TFIDF_output = tf.add(tf.matmul(features, A_TFIDF), b_TFIDF)
TFIDF_loss = tf.reduce_mean(
    tf.nn.sigmoid_cross_entropy_with_logits(logits = TFIDF_output, labels = target))
TFIDF_prediction = tf.cast(tf.equal(tf.round(tf.sigmoid(TFIDF_output)), target), tf.float32)
TFIDF_accuracy = tf.reduce_mean(TFIDF_prediction)

TFIDF_optimizer = tf.train.GradientDescentOptimizer(0.0025)
TFIDF_train_step = TFIDF_optimizer.minimize(TFIDF_loss)
init = tf.global_variables_initializer()
TFIDF_session.run(init)

TFIDF_train_loss, TFIDF_test_loss = [], []
TFIDF_train_accuracy, TFIDF_test_accuracy = [], []
TFIDF_iteration = []
print(colored('\u2500'*27 + ' Text Frequency - Inverse Document Frequency (TF-IDF) ' +
              '\u2500'*27, 'yellow', attrs = ['bold']))
for k in range(10000):
    rand_index = np.random.choice(TFIDF_train_features.shape[0], size = 200)
    rand_x = TFIDF_train_features[rand_index].todense()
    rand_y = np.transpose([TFIDF_train_target[rand_index]])
    TFIDF_session.run(TFIDF_train_step, feed_dict = {features: rand_x, target: rand_y})
    if (k + 1) % 100 == 0:
        TFIDF_iteration.append(k+1)
        train_temp_loss = TFIDF_session.run(TFIDF_loss,
                                            feed_dict = {features: rand_x, target: rand_y})
        TFIDF_train_loss.append(train_temp_loss)
        test_temp_loss = TFIDF_session.run(
            TFIDF_loss, feed_dict = {features: TFIDF_test_features.todense(),
                                     target: np.transpose([TFIDF_test_target])})
        TFIDF_test_loss.append(test_temp_loss)
        train_temp_accuracy = TFIDF_session.run(
            TFIDF_accuracy, feed_dict = {features: rand_x, target: rand_y})
        TFIDF_train_accuracy.append(train_temp_accuracy)
        test_temp_accuracy = TFIDF_session.run(
            TFIDF_accuracy, feed_dict = {features: TFIDF_test_features.todense(),
                                         target: np.transpose([TFIDF_test_target])})
        TFIDF_test_accuracy.append(test_temp_accuracy)
    if (k + 1) % 500 == 0:
        print('Iteration {}:'.format(k+1))
        print('Loss - Training = {:.5f}, Testing = {:.5f}'.format(train_temp_loss,
                                                                  test_temp_loss))
        print('Accuracy - Training = {:.5f}, Testing = {:.5f}'.format(train_temp_accuracy,
                                                                      test_temp_accuracy))

plt.figure()
plt.plot(TFIDF_iteration, TFIDF_train_loss, 'k-', label = 'Training Loss')
plt.plot(TFIDF_iteration, TFIDF_test_loss, 'r--', label = 'Testing Loss')
plt.title('Cross Entropy Loss Variations Across Iterations (TF-IDF)')
plt.xlabel('Iteration')
plt.xlim([0, 10000])
plt.ylabel('Loss')
plt.legend(loc = 'upper right')

plt.figure()
plt.plot(TFIDF_iteration, TFIDF_train_accuracy, 'k-', label = 'Training Accuracy')
plt.plot(TFIDF_iteration, TFIDF_test_accuracy, 'r--', label = 'Testing Accuracy')
plt.title('Accuracies For Both Training & Testing Sets (TF-IDF)')
plt.xlabel('Iteration')
plt.xlim([0, 10000])
plt.ylabel('Accuracy')
plt.legend(loc = 'upper right')

#%% Skip-Gram Embedding
ops.reset_default_graph()
skip_gram_session = tf.Session()

stop_words = stopwords.words('english')

# Load the movie review dataset
positive_reviews_df = pd.read_csv('Data/Movie Review/rt-polarity.pos', sep = '\t',
                                 header = None, names = ['Review'], encoding = 'latin-1')
positive_reviews_df['Target'] = 1
negative_reviews_df = pd.read_csv('Data/Movie Review/rt-polarity.neg', sep = '\t',
                                 header = None, names = ['Review'], encoding = 'latin-1')
negative_reviews_df['Target'] = 0

def normalize_text(df, stop_words):
    normalized_text = []
    for text in df['Review']:
        text = text.lower()
        # Remove any punctuations
        text = ''.join(char for char in text if char not in string.punctuation)
        # Remove any numbers
        text = ''.join(char for char in text if char not in '0123456789')
        # Remove any stopwords
        text = ' '.join(word for word in text.split() if word not in stop_words)
        # Remove any extra whitespaces
        text = ' '.join(text.split())
        normalized_text.append(text)
    df['Review (Normalized)'] = normalized_text
    filtered_review = []
    filtered_target = []
    # Filter out non-informative reviews whose lengths are shorter than 4 words
    for index, text in enumerate(normalized_text):
        if len(text.split()) > 3:
            filtered_review.append(text)
            filtered_target.append(df['Target'][index])
    # Arrange the dataframe for easy viewing
    df = df[['Review', 'Review (Normalized)', 'Target']]
    filtered_df = pd.DataFrame({'Review': filtered_review, 'Target': filtered_target})
    return df, filtered_df

positive_reviews_df, positive_filtered_df = normalize_text(positive_reviews_df, stop_words)
negative_reviews_df, negative_filtered_df = normalize_text(negative_reviews_df, stop_words)
combined_filtered_df = pd.concat([positive_filtered_df, negative_filtered_df],
                                 ignore_index = True)
# Shuffle the dataframe
combined_filtered_df = combined_filtered_df.sample(frac = 1, ignore_index = True)

def build_dictionary(df, vocabulary_size):
    words_list = []
    for review in df['Review']:
        for word in review.split():
            words_list.append(word)
    # Initialize a list containing [word, word_count] for each word with 'RARE'
    # 'RARE': Labelling for any word uncommon enough to meet the vocabulary_size cut-off
    counts = [['RARE', -1]]
    # Add the most frequent words, limited to a frequency of N = vocabulary_size
    counts.extend(Counter(words_list).most_common(vocabulary_size - 1))
    words_dict = {}
    # Key = Word, Value = Current Dictionary Length
    for word, word_count in counts:
        words_dict[word] = len(words_dict)
    return words_dict

def text_to_numbers(df, words_dict):
    indices = []
    for review in df['Review']:
        # Use either the selected index or the rare word index for each word
        indices.append([words_dict[word] if word in words_dict else 0
                        for word in review.split(' ')])
    return indices

words_dict = build_dictionary(combined_filtered_df, 5000)
reversed_words_dict = dict(zip(words_dict.values(), words_dict.keys()))
word_indices = text_to_numbers(combined_filtered_df, words_dict)

validation_words = ['cliche', 'love', 'hate', 'silly', 'sad']
validation_examples = [words_dict[w] for w in validation_words]

def generate_batch(word_indices, batch_size, window_size, method = 'skip-gram'):
    batch_list, label_list = [], []
    while len(batch_list) < batch_size:
        rand_indices = word_indices[int(np.random.choice(len(word_indices), size = 1))]
        sequences_window = [rand_indices[max((index - window_size), 0):
                                         (index + window_size + 1)]
                             for index in range(len(rand_indices))]
        label_indices = [index if index < window_size else window_size
                         for index in range(len(sequences_window))]
        batch, label = [], []
        if method == 'skip-gram':
            batch_with_label = [(w[i], w[:i] + w[(i+1):])
                                for w, i in zip(sequences_window, label_indices)]
            batch_with_label_unpacked = [(b, l) for b, label in batch_with_label
                                                for l in label]
            if len(batch_with_label_unpacked) > 0:
                batch, label = [c for c in zip(*batch_with_label_unpacked)]
        elif method == 'CBOW':
            batch_with_label = [(w[:i] + w[(i+1):], w[i])
                                for w, i in zip(sequences_window, label_indices)]
            batch_with_label = [(b, l) for b, l in batch_with_label
                                       if len(b) == (2*window_size)]
            if len(batch_with_label) > 0:
                batch, label = [c for c in zip(*batch_with_label)]
        elif method == 'doc2vec':
            batch_with_label = [(rand_indices[i:(i+window_size)], rand_indices[i+window_size])
                                for i in range(0, (len(rand_indices) - window_size))]
            batch, label = [c for c in zip(*batch_with_label)]
            batch = [b + [int(np.random.choice(len(word_indices), size = 1))] for b in batch]
        else:
            raise ValueError('{} method has not implemented yet'.format(method))
        batch_list.extend(batch[:batch_size])
        label_list.extend(label[:batch_size])
    batch_list = np.array(batch_list[:batch_size])
    label_list = np.transpose(np.array([label_list[:batch_size]]))
    return batch_list, label_list

skip_gram_embedding_matrix = tf.Variable(tf.random_uniform([5000, 100], -1.0, 1.0))
features = tf.placeholder(shape = [100], dtype = tf.int32)
target = tf.placeholder(shape = [100, 1], dtype = tf.int32)
validation_dataset = tf.constant(validation_examples, dtype = tf.int32)

features_embedding = tf.nn.embedding_lookup(skip_gram_embedding_matrix, features)

# NCE: Noise-Contrastive Error
NCE_weights = tf.Variable(tf.truncated_normal([5000, 100], stddev = 1.0/np.sqrt(100)))
NCE_biases = tf.Variable(tf.zeros([5000]))
# NCE loss generates a binary prediction of the word class VS random noise predictions
NCE_loss = tf.reduce_mean(tf.nn.nce_loss(weights = NCE_weights, biases = NCE_biases,
                                         inputs = features_embedding, labels = target,
                                         num_sampled = int(100/2), num_classes = 5000))

normalized_embedding = skip_gram_embedding_matrix/tf.sqrt(
    tf.reduce_mean(tf.square(skip_gram_embedding_matrix), 1, keepdims = True))
validation_embedding = tf.nn.embedding_lookup(normalized_embedding, validation_dataset)
similiarity = tf.matmul(validation_embedding, normalized_embedding, transpose_b = True)

skip_gram_optimizer = tf.train.GradientDescentOptimizer(1.0)
skip_gram_train_step = skip_gram_optimizer.minimize(NCE_loss)
init = tf.global_variables_initializer()
skip_gram_session.run(init)

skip_gram_loss = []
skip_gram_iteration = []
print(colored('\u2500'*43 + ' Skip-Gram Embeddings ' + '\u2500'*43, 'cyan',
              attrs = ['bold']))
for l in range(100000):
    skip_gram_batch_inputs, skip_gram_batch_labels = generate_batch(word_indices, 100, 2)
    skip_gram_session.run(skip_gram_train_step, feed_dict = {features: skip_gram_batch_inputs,
                                                             target: skip_gram_batch_labels})
    if (l + 1) % 500 == 0:
        temp_loss = skip_gram_session.run(
            NCE_loss, feed_dict = {features: skip_gram_batch_inputs,
                                   target: skip_gram_batch_labels})
        skip_gram_loss.append(temp_loss)
        skip_gram_iteration.append(l+1)
        print('Iteration {}: Loss = {:.5f}'.format(l+1, temp_loss))
    if (l + 1) % 10000 == 0:
        temp_similiarity = skip_gram_session.run(
            similiarity, feed_dict = {features: skip_gram_batch_inputs,
                                      target: skip_gram_batch_labels})
        for m in range(len(validation_words)):
            validation_word = reversed_words_dict[validation_examples[m]]
            nearest_neighbours = (-temp_similiarity[m, :]).argsort()[1:(5+1)]
            closest_words = []
            for K in range(5):
                closest_words.append(reversed_words_dict[nearest_neighbours[K]])
            print('5 Nearest Word To {}: {},'.format(validation_word.capitalize(),
                                                     closest_words))

#%% Continuous Bag-Of-Words (CBOW) Embedding
ops.reset_default_graph()
CBOW_session = tf.Session()

CBOW_words_dict = build_dictionary(combined_filtered_df, 2000)
CBOW_reversed_words_dict = dict(zip(CBOW_words_dict.values(), CBOW_words_dict.keys()))
CBOW_word_indices = text_to_numbers(combined_filtered_df, CBOW_words_dict)

CBOW_validation_words = ['love', 'hate', 'happy', 'sad', 'man', 'woman']
CBOW_validation_examples = [CBOW_words_dict[w] for w in CBOW_validation_words]

CBOW_embedding_matrix = tf.Variable(tf.random_uniform([2000, 200], -1.0, 1.0))
features = tf.placeholder(shape = [200, (2*3)], dtype = tf.int32)
target = tf.placeholder(shape = [200, 1], dtype = tf.int32)
CBOW_validation_dataset = tf.constant(CBOW_validation_examples, dtype = tf.int32)

CBOW_features_embedding = tf.zeros([200, 200])
# CBOW model adds up the embeddings of the context window
for size in range(2*3):
    CBOW_features_embedding += tf.nn.embedding_lookup(CBOW_embedding_matrix,
                                                      features[:, size])

CBOW_NCE_weights = tf.Variable(tf.truncated_normal([2000, 200], stddev = 1.0/np.sqrt(200)))
CBOW_NCE_biases = tf.Variable(tf.zeros([2000]))
CBOW_NCE_loss = tf.reduce_mean(
    tf.nn.nce_loss(weights = CBOW_NCE_weights, biases = CBOW_NCE_biases,
                   inputs = CBOW_features_embedding, labels = target,
                   num_sampled = int(200/2), num_classes = 2000))

CBOW_normalized_embedding = CBOW_embedding_matrix/tf.sqrt(
    tf.reduce_mean(tf.square(CBOW_embedding_matrix), 1, keepdims = True))
CBOW_validation_embedding = tf.nn.embedding_lookup(CBOW_normalized_embedding,
                                                   CBOW_validation_dataset)
CBOW_similiarity = tf.matmul(CBOW_validation_embedding, CBOW_normalized_embedding,
                             transpose_b = True)

CBOW_embeddings = tf.train.Saver({'Embedding': CBOW_embedding_matrix})

CBOW_optimizer = tf.train.GradientDescentOptimizer(0.05)
CBOW_train_step = CBOW_optimizer.minimize(CBOW_NCE_loss)
init = tf.global_variables_initializer()
CBOW_session.run(init)

CBOW_loss = []
CBOW_iteration = []
print(colored('\u2500'*32 + ' Continuous Bag-Of-Words (CBOW) Embeddings ' + '\u2500'*32,
              'magenta', attrs = ['bold']))
for a in range(50000):
    CBOW_batch_inputs, CBOW_batch_labels = generate_batch(CBOW_word_indices, 200, 3,
                                                          method = 'CBOW')
    CBOW_session.run(CBOW_train_step, feed_dict = {features: CBOW_batch_inputs,
                                                   target: CBOW_batch_labels})
    if (a + 1) % 500 == 0:
        temp_loss = CBOW_session.run(CBOW_NCE_loss, feed_dict = {features: CBOW_batch_inputs,
                                                                 target: CBOW_batch_labels})
        CBOW_loss.append(temp_loss)
        CBOW_iteration.append(a+1)
        print('Iteration {}: Loss = {:.5f}'.format(a+1, temp_loss))
    if (a + 1) % 5000 == 0:
        temp_similiarity = CBOW_session.run(
            CBOW_similiarity, feed_dict = {features: CBOW_batch_inputs,
                                           target: CBOW_batch_labels})
        for c in range(len(CBOW_validation_words)):
            CBOW_validation_word = CBOW_reversed_words_dict[CBOW_validation_examples[c]]
            nearest_neighbours = (-temp_similiarity[c, :]).argsort()[1:(5+1)]
            closest_words = []
            for K in range(5):
                closest_words.append(CBOW_reversed_words_dict[nearest_neighbours[K]])
            print('5 Nearest Word To {}: {},'.format(CBOW_validation_word.capitalize(),
                                                     closest_words))

    if (a + 1) % 5000 == 0:
        with open('Embeddings/CBOW Word Dictionary (Movie).pkl', 'wb') as file:
            pickle.dump(CBOW_words_dict, file)

        save_filepath = CBOW_embeddings.save(
            CBOW_session, 'Embeddings/CBOW Embeddings (Movie).ckpt')
        print('Embeddings saved in "{}"'.format(save_filepath))

plt.figure()
plt.plot(CBOW_iteration, CBOW_loss, 'k')
plt.title('Noise-Contrastive Error (NCE) Loss Variations Across Iterations (Continuous Bag-Of-Words)')
plt.xlabel('Iteration')
plt.xlim([0, 50000])
plt.ylabel('Loss')

#%% Word2vec Using CBOW Embeddings
ops.reset_default_graph()
word2vec_session = tf.Session()

# Split the combined dataframe into training (80%) and testing (20%) datasets
word2vec_train_indices = np.random.choice(
    len(combined_filtered_df), round(0.8 * len(combined_filtered_df)), replace = False)
word2vec_test_indices = np.array(list(
    set(range(len(combined_filtered_df))) - set(word2vec_train_indices)))
combined_train_df = combined_filtered_df.iloc[word2vec_train_indices]
combined_test_df = combined_filtered_df.iloc[word2vec_test_indices]

# Load the word dictionary created from the CBOW embeddings
word2vec_words_dict = pickle.load(open('Embeddings/CBOW Word Dictionary (Movie).pkl',
                                       'rb'))
train_word_indices = text_to_numbers(combined_train_df, word2vec_words_dict)
test_word_indices = text_to_numbers(combined_test_df, word2vec_words_dict)
# Standardize length of all word indices to 100 (Pad word indices with length < 100 with zeroes)
train_word_indices = np.array(
    [index[0:100] for index in [word_index + ([0]*100) for word_index in train_word_indices]])
test_word_indices = np.array(
    [index[0:100] for index in [word_index + ([0]*100) for word_index in test_word_indices]])

word2vec_embedding_matrix = tf.Variable(tf.random_uniform([2000, 200], -1.0, 1.0))
A_word2vec = tf.Variable(tf.random_normal(shape = [200, 1]))
b_word2vec = tf.Variable(tf.random_normal(shape = [1, 1]))
features = tf.placeholder(shape = [None, 100], dtype = tf.int32)
target = tf.placeholder(shape = [None, 1], dtype = tf.float32)

# Taking the average embedding
word2vec_features_embedding = tf.reduce_mean(
    tf.nn.embedding_lookup(word2vec_embedding_matrix, features), 1)

word2vec_output = tf.add(tf.matmul(word2vec_features_embedding, A_word2vec), b_word2vec)
word2vec_loss = tf.reduce_mean(
    tf.nn.sigmoid_cross_entropy_with_logits(logits = word2vec_output, labels = target))
word2vec_prediction = tf.cast(
    tf.equal(tf.round(tf.sigmoid(word2vec_output)), target), tf.float32)
word2vec_accuracy = tf.reduce_mean(word2vec_prediction)

word2vec_optimizer = tf.train.AdagradOptimizer(0.005)
word2vec_train_step = word2vec_optimizer.minimize(word2vec_loss)
init = tf.global_variables_initializer()
word2vec_session.run(init)

word2vec_embeddings = tf.train.Saver({'Embedding': word2vec_embedding_matrix})
word2vec_embeddings.restore(word2vec_session, 'Embeddings/CBOW Embeddings (Movie).ckpt')

word2vec_train_loss, word2vec_test_loss = [], []
word2vec_train_accuracy, word2vec_test_accuracy = [], []
word2vec_iteration = []
print(colored('\u2500'*38 + ' Word2vec Using CBOW Embeddings ' + '\u2500'*38, 'grey',
              attrs = ['bold']))
for c in range(10000):
    rand_index = np.random.choice(train_word_indices.shape[0], size = 100)
    rand_x = train_word_indices[rand_index]
    rand_y = np.transpose([combined_train_df['Target'].iloc[rand_index]])
    word2vec_session.run(word2vec_train_step, feed_dict = {features: rand_x, target: rand_y})
    if (c + 1) % 100 == 0:
        word2vec_iteration.append(c+1)
        train_temp_loss = word2vec_session.run(word2vec_loss, feed_dict = {features: rand_x,
                                                                           target: rand_y})
        word2vec_train_loss.append(train_temp_loss)
        test_temp_loss = word2vec_session.run(
            word2vec_loss, feed_dict = {features: test_word_indices,
                                        target: np.transpose([combined_test_df['Target']])})
        word2vec_test_loss.append(test_temp_loss)
        train_temp_accuracy = word2vec_session.run(
            word2vec_accuracy, feed_dict = {features: rand_x, target: rand_y})
        word2vec_train_accuracy.append(train_temp_accuracy)
        test_temp_accuracy = word2vec_session.run(
            word2vec_accuracy, feed_dict = {features: test_word_indices,
                                            target: np.transpose([combined_test_df['Target']])})
        word2vec_test_accuracy.append(test_temp_accuracy)

    if (c + 1) % 500 == 0:
        print('Iteration {}:'.format(c+1))
        print('Loss - Training = {:.5f}, Testing = {:.5f}'.format(train_temp_loss,
                                                                  test_temp_loss))
        print('Accuracy - Training = {:.5f}, Testing = {:.5f}'.format(train_temp_accuracy,
                                                                      test_temp_accuracy))

plt.figure()
plt.plot(word2vec_iteration, word2vec_train_loss, 'k-', label = 'Training Loss')
plt.plot(word2vec_iteration, word2vec_test_loss, 'r--', label = 'Testing Loss', linewidth = 3)
plt.title('Cross Entropy Loss Variations Across Iterations (Word2vec Using CBOW Embeddings)')
plt.xlabel('Iteration')
plt.ylabel('Loss')
plt.legend(loc = 'best')

plt.figure()
plt.plot(word2vec_iteration, word2vec_train_accuracy, 'k-', label = 'Training Accuracy')
plt.plot(word2vec_iteration, word2vec_test_accuracy, 'r--', label = 'Testing Accuracy',
         linewidth = 3)
plt.title('Accuracies For Both Training & Testing Sets (Word2vec Using CBOW Embeddings)')
plt.xlabel('Iteration')
plt.ylabel('Accuracy')
plt.legend(loc = 'best')

#%% Doc2vec For Sentiment Analysis
#### Train The Word and Document Embeddings
ops.reset_default_graph()
doc2vec_session = tf.Session()

doc2vec_words_dict = build_dictionary(combined_filtered_df, 7500)
doc2vec_reversed_words_dict = dict(zip(doc2vec_words_dict.values(), doc2vec_words_dict.keys()))
doc2vec_word_indices = text_to_numbers(combined_filtered_df, doc2vec_words_dict)
doc2vec_validation_examples = [doc2vec_words_dict[w] for w in CBOW_validation_words]

word_embedding_matrix = tf.Variable(tf.random_uniform([7500, 200], -1.0, 1.0))
document_embedding_matrix = tf.Variable(tf.random.uniform([len(combined_filtered_df), 100],
                                                          -1.0, 1.0))
features = tf.placeholder(shape = [None, (3+1)], dtype = tf.int32)
target = tf.placeholder(shape = [None, 1], dtype = tf.int32)
doc2vec_validation_dataset = tf.constant(doc2vec_validation_examples, dtype = tf.int32)

word_embedding = tf.zeros([500, 200])
for size in range(3):
    word_embedding += tf.nn.embedding_lookup(word_embedding_matrix, features[:, size])
document_indices = tf.slice(features, [0, 3], [500, 1])
document_embedding = tf.nn.embedding_lookup(document_embedding_matrix, document_indices)
combined_embedding = tf.concat(axis = 1,
                               values = [word_embedding, tf.squeeze(document_embedding)])

doc2vec_NCE_weights = tf.Variable(tf.truncated_normal([7500, (200+100)],
                                                      stddev = 1.0/np.sqrt(200+100)))
doc2vec_NCE_biases = tf.Variable(tf.zeros([7500]))
doc2vec_NCE_loss = tf.reduce_mean(
    tf.nn.nce_loss(weights = doc2vec_NCE_weights, biases = doc2vec_NCE_biases, labels = target,
                   inputs = combined_embedding, num_sampled = int(500/2), num_classes = 7500))

normalized_word_embedding = word_embedding_matrix/tf.sqrt(
    tf.reduce_sum(tf.square(word_embedding_matrix), 1, keepdims = True))
validation_word_embedding = tf.nn.embedding_lookup(normalized_word_embedding,
                                                   doc2vec_validation_dataset)
doc2vec_similiarity = tf.matmul(validation_word_embedding, normalized_word_embedding,
                                transpose_b = True)

doc2vec_embeddings = tf.train.Saver({'Word Embeddings': word_embedding_matrix,
                                    'Document Embeddings': document_embedding_matrix})

doc2vec_optimizer = tf.train.GradientDescentOptimizer(0.001)
doc2vec_train_step = doc2vec_optimizer.minimize(doc2vec_NCE_loss)
init = tf.global_variables_initializer()
doc2vec_session.run(init)

doc2vec_loss = []
train_iteration = []
print(colored('\u2500'*29 + " Movie Reviews' Sentiment Analysis Using Doc2vec " + '\u2500'*29,
              'blue', attrs = ['bold']))
print(colored('Training The Word and Document Embeddings:', 'magenta'))
for d in range(100000):
    doc2vec_batch_inputs, doc2vec_batch_labels = generate_batch(doc2vec_word_indices, 500, 3,
                                                                method = 'doc2vec')
    doc2vec_session.run(doc2vec_train_step, feed_dict = {features: doc2vec_batch_inputs,
                                                         target: doc2vec_batch_labels})
    if (d + 1) % 500 == 0:
        temp_loss = doc2vec_session.run(
            doc2vec_NCE_loss, feed_dict = {features: doc2vec_batch_inputs,
                                           target: doc2vec_batch_labels})
        doc2vec_loss.append(temp_loss)
        train_iteration.append(d+1)
        print('Iteration {}: Loss = {:.5f}'.format(d+1, temp_loss))
    if (d + 1) % 5000 == 0:
        temp_similiarity = doc2vec_session.run(
            doc2vec_similiarity, feed_dict = {features: doc2vec_batch_inputs,
                                              target: doc2vec_batch_labels})
        for e in range(len(CBOW_validation_words)):
            doc2vec_validation_word = doc2vec_reversed_words_dict[doc2vec_validation_examples[e]]
            nearest_neighbours = (-temp_similiarity[e, :]).argsort()[1:(5+1)]
            closest_words = []
            for K in range(5):
                closest_words.append(doc2vec_reversed_words_dict[nearest_neighbours[K]])
            print('5 Nearest Word To {}: {},'.format(doc2vec_validation_word.capitalize(),
                                                     closest_words))

    if (d + 1) % 5000 == 0:
        with open('Embeddings/Doc2vec Word Dictionary (Movie).pkl', 'wb') as file:
            pickle.dump(doc2vec_words_dict, file)

        doc2vec_save_filepath = doc2vec_embeddings.save(
            doc2vec_session, 'Embeddings/Doc2vec Embeddings (Movie).ckpt')
        print('Embeddings saved in "{}"'.format(doc2vec_save_filepath))

#### Using The Combined Word and Document Embeddings To Predict Movie Reviews' Sentiments
# Split the combined dataframe into training (80%) and testing (20%) datasets
doc2vec_train_indices = np.sort(np.random.choice(
    len(combined_filtered_df), round(0.8 * len(combined_filtered_df)), replace = False))
doc2vec_test_indices = np.sort(np.array(list(
    set(range(len(combined_filtered_df))) - set(doc2vec_train_indices))))
doc2vec_combined_train_df = combined_filtered_df.iloc[doc2vec_train_indices]
doc2vec_combined_test_df = combined_filtered_df.iloc[doc2vec_test_indices]

doc2vec_train_word_indices = text_to_numbers(doc2vec_combined_train_df, doc2vec_words_dict)
doc2vec_test_word_indices = text_to_numbers(combined_test_df, doc2vec_words_dict)
# Standardize length of all word indices to 20
doc2vec_train_word_indices = np.array(
    [index[0:20] for index in [word_index + ([0]*20)
                              for word_index in doc2vec_train_word_indices]])
doc2vec_test_word_indices = np.array(
    [index[0:20] for index in [word_index + ([0]*20)
                               for word_index in doc2vec_test_word_indices]])

A_doc2vec = tf.Variable(tf.random_normal(shape = [(200+100), 1]))
b_doc2vec = tf.Variable(tf.random_normal(shape = [1, 1]))
logistic_features = tf.placeholder(shape = [None, (20+1)], dtype = tf.int32)
logistic_target = tf.placeholder(shape = [None, 1], dtype = tf.int32)

logistic_word_embedding = tf.zeros([500, 200])
for size in range(20):
    logistic_word_embedding += tf.nn.embedding_lookup(word_embedding_matrix,
                                                      logistic_features[:, size])
logistic_document_indices = tf.slice(logistic_features, [0, 20], [500, 1])
logistic_document_embedding = tf.nn.embedding_lookup(document_embedding_matrix,
                                                     logistic_document_indices)
logistic_combined_embedding = tf.concat(
    axis = 1, values = [logistic_word_embedding, tf.squeeze(logistic_document_embedding)])

doc2vec_output = tf.add(tf.matmul(logistic_combined_embedding, A_doc2vec), b_doc2vec)
doc2vec_logistic_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
    logits = doc2vec_output, labels = tf.cast(logistic_target, tf.float32)))
doc2vec_prediction = tf.cast(tf.equal(tf.round(tf.sigmoid(doc2vec_output)),
                                      tf.cast(logistic_target, tf.float32)), tf.float32)
doc2vec_accuracy = tf.reduce_mean(doc2vec_prediction)

doc2vec_logistic_optimizer = tf.train.GradientDescentOptimizer(0.0025)
doc2vec_logistic_train_step = doc2vec_logistic_optimizer.minimize(
    doc2vec_logistic_loss, var_list = [A_doc2vec, b_doc2vec])
init = tf.global_variables_initializer()
doc2vec_session.run(init)

doc2vec_train_loss, doc2vec_test_loss = [], []
doc2vec_train_accuracy, doc2vec_test_accuracy = [], []
prediction_iteration = []
print(colored("Using The Combined Embeddings In A Logistic Regression To Predict The Movie Review's Sentiments:", 'cyan'))
for e in range(10000):
    train_rand_index = np.random.choice(doc2vec_train_word_indices.shape[0], size = 500)
    train_rand_x = np.hstack((doc2vec_train_word_indices[train_rand_index],
                              np.transpose([doc2vec_train_indices[train_rand_index]])))
    train_rand_y = np.transpose([doc2vec_combined_train_df['Target'].iloc[train_rand_index]])
    doc2vec_session.run(doc2vec_logistic_train_step,
                        feed_dict = {logistic_features: train_rand_x,
                                     logistic_target: train_rand_y})
    if (e + 1) % 100 == 0:
        test_rand_index = np.random.choice(doc2vec_test_word_indices.shape[0], size = 500)
        test_rand_x = np.hstack((doc2vec_test_word_indices[test_rand_index],
                                 np.transpose([doc2vec_test_indices[test_rand_index]])))
        test_rand_y = np.transpose([doc2vec_combined_test_df['Target'].iloc[test_rand_index]])
        prediction_iteration.append(e+1)
        train_temp_loss = doc2vec_session.run(doc2vec_logistic_loss,
                                              feed_dict = {logistic_features: train_rand_x,
                                                           logistic_target: train_rand_y})
        doc2vec_train_loss.append(train_temp_loss)
        test_temp_loss = doc2vec_session.run(doc2vec_logistic_loss,
                                             feed_dict = {logistic_features: test_rand_x,
                                                          logistic_target: test_rand_y})
        doc2vec_test_loss.append(test_temp_loss)
        train_temp_accuracy = doc2vec_session.run(doc2vec_accuracy,
                                                  feed_dict = {logistic_features: train_rand_x,
                                                               logistic_target: train_rand_y})
        doc2vec_train_accuracy.append(train_temp_accuracy)
        test_temp_accuracy = doc2vec_session.run(doc2vec_accuracy,
                                                 feed_dict = {logistic_features: test_rand_x,
                                                              logistic_target: test_rand_y})
        doc2vec_test_accuracy.append(test_temp_accuracy)

    if (e + 1) % 500 == 0:
        print('Iteration {}:'.format(e+1))
        print('Loss - Training = {:.5f}, Testing = {:.5f}'.format(train_temp_loss,
                                                                  test_temp_loss))
        print('Accuracy - Training = {:.5f}, Testing = {:.5f}'.format(train_temp_accuracy,
                                                                      test_temp_accuracy))

plt.figure()
plt.plot(prediction_iteration, doc2vec_train_loss, 'k-', label = 'Training Loss')
plt.plot(prediction_iteration, doc2vec_test_loss, 'r--', label = 'Testing Loss', linewidth = 3)
plt.title('Cross Entropy Loss Variations Across Iterations (Doc2vec Using Word & Document Embeddings)')
plt.xlabel('Iteration')
plt.xlim([0, 10000])
plt.ylabel('Loss')
plt.legend(loc = 'best')

plt.figure()
plt.plot(prediction_iteration, doc2vec_train_accuracy, 'k-', label = 'Training Accuracy')
plt.plot(prediction_iteration, doc2vec_test_accuracy, 'r--', label = 'Testing Accuracy',
         linewidth = 3)
plt.title('Accuracies For Both Training & Testing Sets (Doc2vec Using Word & Document Embeddings)')
plt.xlabel('Iteration')
plt.xlim([0, 10000])
plt.ylabel('Accuracy')
plt.legend(loc = 'best')

stop = time.perf_counter()
print('\u2550'*105)
duration = stop - start
hours = divmod(divmod(duration, 60), 60)[0][0]
minutes = divmod(divmod(duration, 60), 60)[1][0]
seconds = divmod(divmod(duration, 60), 60)[1][1]
print(colored('Execution Duration: {:.2f}s ({:.1f}hrs, {:.1f}mins, {:.2f}s)'.format(
    duration, hours, minutes, seconds), 'red'))