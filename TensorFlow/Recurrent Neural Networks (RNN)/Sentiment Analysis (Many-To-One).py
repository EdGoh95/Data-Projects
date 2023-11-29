#!/usr/bin/env python3
"""
Deep Learning With TensorFlow and Keras Third Edition (Packt Publishing)
Chapter 5: Recurrent Neural Networks
"""
import os
import numpy as np
import tensorflow as tf
from sklearn.metrics import accuracy_score, confusion_matrix

def download_and_read(url):
    local_file = url.split('/')[-1]
    local_file = local_file.replace('%20', ' ')
    p = tf.keras.utils.get_file(local_file, url, extract = True, cache_subdir = 'Data', cache_dir = '.')
    local_folder = os.path.join('Data', local_file.split('.')[0])
    labeled_sentences = []
    for labeled_filename in os.listdir(local_folder):
        if labeled_filename.endswith('_labelled.txt'):
            with open(os.path.join(local_folder, labeled_filename), 'r') as file:
                for line in file:
                    sentence, label = line.strip().split('\t')
                    labeled_sentences.append((sentence, label))
    return labeled_sentences

labeled_reviews = download_and_read('https://archive.ics.uci.edu/ml/machine-learning-databases/00331/sentiment%20labelled%20sentences.zip')
reviews = [s for (s, l) in labeled_reviews]
labels = [int(l) for (s, l) in labeled_reviews]

tokenizer = tf.keras.preprocessing.text.Tokenizer()
tokenizer.fit_on_texts(reviews)
vocab_size = len(tokenizer.word_counts)
print('Vocabulary size:', vocab_size)
word2idx = tokenizer.word_index
idx2word = {index: word for (word, index) in word2idx.items()}

review_lengths = np.array([len(review.split()) for review in reviews])
print([(p, np.percentile(review_lengths, p)) for p in [75, 80, 90, 95, 99, 100]])

reviews_as_ints = tokenizer.texts_to_sequences(reviews)
reviews_as_ints = tf.keras.preprocessing.sequence.pad_sequences(reviews_as_ints, maxlen = 64)
labels_as_ints = np.array(labels)
reviews_dataset = tf.data.Dataset.from_tensor_slices((reviews_as_ints, labels_as_ints))
reviews_dataset = reviews_dataset.shuffle(10000)
test_size = len(reviews)//3
validation_size = (len(reviews) - test_size)//10
test_dataset = reviews_dataset.take(test_size).batch(64)
validation_dataset = reviews_dataset.skip(test_size).take(validation_size).batch(64)
training_dataset = reviews_dataset.skip(test_size + validation_size).batch(64)

class SentimentAnalysisModel(tf.keras.Model):
    def __init__(self, vocab_size, max_sentence_len, **kwargs):
        super(SentimentAnalysisModel, self).__init__(**kwargs)
        self.embedding = tf.keras.layers.Embedding(vocab_size, max_sentence_len)
        self.bidirectional_lstm = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(max_sentence_len))
        self.dense = tf.keras.layers.Dense(64, activation = 'relu')
        self.out = tf.keras.layers.Dense(1, activation = 'sigmoid')
    def call(self, x):
        x = self.embedding(x)
        x = self.bidirectional_lstm(x)
        x = self.dense(x)
        x = self.out(x)
        return x

sentiment_analysis = SentimentAnalysisModel(vocab_size + 1, 64)
sentiment_analysis.build(input_shape = (64, 64))
sentiment_analysis.summary()
sentiment_analysis.compile(loss = 'binary_crossentropy', optimizer = 'adam', metrics = ['accuracy'])
best_model_file = os.path.join('Data/Logs (Sentiment Analysis)', 'best_model.h5')
checkpoint = tf.keras.callbacks.ModelCheckpoint(best_model_file, save_weights_only = True, save_best_only = True)
tensorboard = tf.keras.callbacks.TensorBoard(log_dir = 'Data/Logs (Sentiment Analysis)')
history = sentiment_analysis.fit(training_dataset, epochs = 10, validation_data = validation_dataset,
                                 callbacks = [checkpoint, tensorboard])

# Loading the best model from the checkpoint callback
best_model = SentimentAnalysisModel(vocab_size + 1, 64)
best_model.build(input_shape = (64, 64))
best_model.load_weights(best_model_file)
best_model.compile(loss = 'binary_crossentropy', optimizer = 'adam', metrics = ['accuracy'])
test_loss, test_accuracy = best_model.evaluate(test_dataset)
print('Test loss: {:.3f}, Test accuracy: {:.3f}'.format(test_loss, test_accuracy))

review_labels = []
predictions = []
idx2word[0] = 'PAD'
is_first_batch = True
for test_batch in test_dataset:
    inputs_best, labels_best = test_batch
    prediction_batch = best_model.predict(inputs_best)
    predictions.extend([(1 if p > 0.5 else 0) for p in prediction_batch])
    review_labels.extend([l for l in labels_best])
    if is_first_batch:
        # Print the first batch of label, prediction and review
        for rowid in range(inputs_best.shape[0]):
            words = [idx2word[index] for index in inputs_best[rowid].numpy()]
            words = [word for word in words if word != 'PAD']
            sentence = ' '.join(words)
            print('{}\t{}\t{}'.format(labels[rowid], predictions[rowid], sentence))
        is_first_batch = False

print('Accuracy score:', accuracy_score(review_labels, predictions))
print('Confusion matrix:', confusion_matrix(review_labels, predictions), sep = '\n')