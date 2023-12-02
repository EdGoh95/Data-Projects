#!/usr/bin/env python3
"""
Deep Learning With TensorFlow And Keras Third Edition (Packt Publishing) Chapter 6: Transformers
"""
import torch
import numpy as np
from transformers import pipeline, AutoModelForSequenceClassification, AutoTokenizer, \
    DefaultDataCollator, TrainingArguments, Trainer
from datasets import load_dataset, load_metric

#%% Text Generation
text_generator = pipeline(task = 'text-generation')
# From Einstein's Theory of Relativity
print(text_generator(' '.join(['The original theory of relativity is based upon the premise that all',
                               'coordinate systems in relative uniform translatory motion to each other',
                               'are equally valid and equivalent to one another'])))
print('\n')
# From Harry Potter
print(text_generator('It takes a great deal of bravery to stand up to our enemies'))
print('\u2550'*130)

#%% Model Auto-Selection & Auto-Tokenization
pretrained_model = AutoModelForSequenceClassification.from_pretrained('distilbert-base-uncased')
tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
sequence = 'The original theory of relativity is based on the premise that all coordinate systems'
print(tokenizer(sequence))
print('\u2550'*130)

#%% Named Entity Recognition (NER)
ner_pipeline = pipeline('ner')
ner_sequence = '''Mr. and Mrs. Dusley, of number four, Privet Drive, were proud to say that they were
                  perfectly normal, thank you very much.'''
for entity in ner_pipeline(ner_sequence):
    print(entity)
print('\u2550'*130)

#%% Text Summarization
text_summarizer = pipeline('summarization', model = 't5-base')
article = '''Mr. and Mrs. Dusley, of number four, Privet Drive, were proud to say that they were
             perfectly normal, thank you very much.

             They were the last people you'd expect to be involved in anything strange or mysterious,
             because they just didn't hold with such nonsense.

             Mr. Dursley was the director of a firm called Grunnings, which made drills. He was a big,
             beefy man with hardly any neck, although he did have a very large moustache.

             Mrs. Dusley was thin and blonde and had nearly twice the usual amount of neck, which came
             in very useful as she spent so much of her time craning over garden fences, spying on the
             neighbours.

             The Dursleys had a small son called Dudly and in their opinion there was no finer boy
             anywhere.'''
print(text_summarizer(article, max_length = 130, min_length = 30, do_sample = False))
print('\u2550'*130)