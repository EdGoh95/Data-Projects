#!/usr/bin/env python3
"""
Deep Learning With TensorFlow and Keras Third Edition (Packt Publishing) Chapter 6: Transformers
"""
import seaborn as sns
import textwrap
import tensorflow_hub as tfhub
from sklearn.metrics import pairwise

pretrained_model = tfhub.KerasLayer('https://tfhub.dev/google/nnlm-en-dim128/2')
pretrained_embeddings = pretrained_model(['The rain in Spain.', 'falls', 'mainly', 'In the plain!'])
print(pretrained_embeddings.shape)

sentences = [
    "Do not pity the dead, Harry. Pity the living, and, above all those who live without love.",
    "It is impossible to manufacture or imitate love",
    "Differences of habit and language are nothing at all if our aims are identical and our hearts are open.",
    "What do I care how he looks? I am good-looking enough for both of us, I theek! All these scars show is zat my husband is brave!",
    "Love as powerful as your mother's for you leaves it's own mark. To have been loved so deeply,even though the person who loved us is gone, will give us some protection forever.",
    "Family...Whatever yeh say, blood's important....",
    "I cared more for your happiness than your knowing the truth, more for your peace of mind than my plan, more for your life than the lives that might be lost if the plan failed."
    ]

bert_model = tfhub.load('https://tfhub.dev/google/experts/bert/wiki_books/2')
preprocessing = tfhub.load('https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3')
inputs = preprocessing(sentences)
outputs = bert_model(inputs)

def plot_similarity(features, labels):
    '''
    Plot the similarity matrix of the embeddings
    '''
    cos_sim = pairwise.cosine_similarity(features)
    sns.set(font_scale = 1.2)
    cbar_kws = dict(use_gridspec = False, location = 'right')
    heatmap = sns.heatmap(cos_sim, xticklabels = labels, yticklabels = labels, vmin = 0, vmax = 1,
                          cmap = 'Blues', cbar_kws = cbar_kws)
    labels = [textwrap.fill(label.get_text(), 30) for label in heatmap.get_xticklabels()]
    heatmap.set_xticklabels(labels, size = 9, rotation = 0)
    heatmap.set_yticklabels(labels, size = 11)
    heatmap.set_title('Semantic Textual Similarity', size = 30, pad = 30, weight = 'bold')

plot_similarity(outputs['pooled_output'], sentences)