#!/usr/bin/env python3
"""
Deep Learning With TensorFlow and Keras Third Edition (Packt Publishing)
Chapter 5: Recurrent Neural Networks
"""
import os
import numpy as np
import tensorflow as tf
import nltk
nltk.download('treebank')

def read_treebank(num_pairs = None):
    sentences = []
    pos_tags = []
    for sentence in nltk.corpus.treebank.tagged_sents():
        sentence, pos_tag = map(list, zip(*sentence))
        sentences.append(' '.join([word for word in sentence]))
        pos_tags.append(' '.join([tag for tag in pos_tag]))
    return sentences, pos_tags

treebank_sentences, treebank_tags = read_treebank()
assert(len(treebank_sentences) == len(treebank_tags))
print('Number of records in the Treebank corpus:', len(treebank_sentences))

def tokenize_and_build_vocab(texts, vocab_size = None, lower = True):
    if vocab_size is None:
        tokenizer = tf.keras.preprocessing.text.Tokenizer(lower = lower)
    else:
        tokenizer = tf.keras.preprocessing.text.Tokenizer(num_words = vocab_size + 1, oov_token = 'UNK',
                                                          lower = lower)
    tokenizer.fit_on_texts(texts)
    if vocab_size is not None:
        tokenizer.word_index = {element: index for element, index in tokenizer.word_index.items()
                                if index <= vocab_size + 1}
    word2idx = tokenizer.word_index
    idx2word = {index: word for word, index in word2idx.items()}
    return word2idx, idx2word, tokenizer

word2idx_source, idx2word_source, tokenizer_source = tokenize_and_build_vocab(
treebank_sentences, vocab_size = 9000)
word2idx_target, idx2word_target, tokenizer_target = tokenize_and_build_vocab(
    treebank_tags, vocab_size = 38, lower = False)
source_vocab_size = len(word2idx_source)
target_vocab_size = len(word2idx_target)
print('Vocab sizes: {} (Source), {} (Target)'.format(source_vocab_size, target_vocab_size))

sentence_lengths = np.array([len(sentence.split()) for sentence in treebank_sentences])
print([(p, np.percentile(sentence_lengths, p)) for p in [75, 80, 90, 95, 99, 100]])

# Convert the sentences into a sequence of integers
sentences_as_ints = tokenizer_source.texts_to_sequences(treebank_sentences)
sentences_as_ints = tf.keras.preprocessing.sequence.pad_sequences(sentences_as_ints, maxlen = 271,
                                                                  padding = 'post')

# Convert the POS tags into a sequence of integers
POS_as_ints = tokenizer_target.texts_to_sequences(treebank_tags)
POS_as_ints = tf.keras.preprocessing.sequence.pad_sequences(POS_as_ints, maxlen = 271, padding = 'post')

POS_as_categories = []
for p in POS_as_ints:
    POS_as_categories.append(tf.keras.utils.to_categorical(p, num_classes = target_vocab_size,
                                                           dtype = 'int32'))
POS_as_categories = tf.keras.preprocessing.sequence.pad_sequences(POS_as_categories, maxlen = 271)
treebank_dataset = tf.data.Dataset.from_tensor_slices((sentences_as_ints, POS_as_categories))
idx2word_source[0] = 'PAD'
idx2word_target[0] = 'PAD'

# Split the dataset into training, validation and test sets
treebank_dataset = treebank_dataset.shuffle(10000)
test_size = len(treebank_sentences)//3
validation_size = (len(treebank_sentences) - test_size)//10
test_dataset = treebank_dataset.take(test_size).batch(128)
validation_dataset = treebank_dataset.skip(test_size).take(validation_size).batch(128)
training_dataset = treebank_dataset.skip(test_size + validation_size).batch(128)

class POSTagger(tf.keras.Model):
    def __init__(self, source_vocab_size, target_vocab_size, embedding_dimension, max_sentence_len,
                 rnn_output_dimension, **kwargs):
        super(POSTagger, self).__init__(**kwargs)
        self.embedding = tf.keras.layers.Embedding(source_vocab_size, embedding_dimension,
                                                   input_length = max_sentence_len)
        self.dropout = tf.keras.layers.SpatialDropout1D(0.2)
        self.rnn = tf.keras.layers.Bidirectional(tf.keras.layers.GRU(rnn_output_dimension,
                                                                    return_sequences = True))
        self.dense = tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(target_vocab_size))
        self.activation = tf.keras.layers.Activation('softmax')

    def call(self, x):
        x = self.embedding(x)
        x = self.dropout(x)
        x = self.rnn(x)
        x = self.dense(x)
        x = self.activation(x)
        return x

def masked_accuracy():
    def masked_accuracy_fn(label, prediction):
        label = tf.compat.v1.keras.backend.argmax(label, axis = -1)
        prediction = tf.compat.v1.keras.backend.argmax(prediction, axis = -1)
        mask = tf.compat.v1.keras.backend.cast(tf.keras.backend.not_equal(prediction, 0), tf.int32)
        matches = tf.compat.v1.keras.backend.cast(tf.compat.v1.keras.backend.equal(label, prediction), tf.int32) * mask
        accuracy = tf.compat.v1.keras.backend.sum(matches)/tf.compat.v1.keras.backend.maximum(
            tf.compat.v1.keras.backend.sum(mask), 1)
        return accuracy
    return masked_accuracy_fn

POS_tagging_model = POSTagger(source_vocab_size, target_vocab_size, 128, 271, 256)
POS_tagging_model.build(input_shape = (128, 271))
POS_tagging_model.summary()
POS_tagging_model.compile(loss = 'categorical_crossentropy', optimizer = 'adam',
                          metrics = ['accuracy', masked_accuracy()])

best_model_file = os.path.join('Logs/POS Tagging)', 'best_model.h5')
checkpoint = tf.keras.callbacks.ModelCheckpoint(best_model_file, save_weights_only = True,
                                                save_best_only = True)
tensorboard = tf.keras.callbacks.TensorBoard(log_dir = 'Logs/POS Tagging)')
POS_tagging_model_fit = POS_tagging_model.fit(
    training_dataset, epochs = 50, validation_data = validation_dataset, callbacks = [checkpoint, tensorboard])