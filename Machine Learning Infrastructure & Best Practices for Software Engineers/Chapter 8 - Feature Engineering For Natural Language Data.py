#!/usr/bin/env python3
"""
Machine Learning Infrastructure And Best Practices For Software Engineers (Packt Publishing)
Chapter 8: Feature Engineering for Natural Language Data
"""
from tokenizers import BertWordPieceTokenizer, Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import Whitespace
from sentencepiece import SentencePieceTrainer, SentencePieceProcessor
from gensim.models import word2vec, FastText

#%% Tokenizers
#### WordPiece Tokenizer
wordpiece_tokenizer = BertWordPieceTokenizer(clean_text = True, handle_chinese_chars = False,
                                             strip_accents = False, lowercase = True)
wordpiece_tokenizer.train(files = 'nx_ip_checksum_compute.c', vocab_size = 30_000, min_frequency = 1,
                          limit_alphabet = 1000, wordpieces_prefix = '##',
                          special_tokens = ['[PAD]', '[UNK]', '[CLS]', '[SEP]', '[MASK]'])
CProgram_string = """
int main(int argc, void **argc)
{
     printf('%s', 'Hello World!\n');
     return 0;
}
"""
wordpiece_tokenized_text = wordpiece_tokenizer.encode(CProgram_string)

#### Byte-Pair Encoding (BPE) Algorithm
BPE_tokenizer = Tokenizer(BPE(unk_token = '[UNK]'))
BPE_tokenizer.pre_tokenizer = Whitespace()
BPE_trainer = BpeTrainer(special_tokens = ['[UNK]', '[CLS]', '[SEP]', '[PAD]', '[MASK]'])
BPE_tokenizer.train(['nx_ip_checksum_compute.c'], BPE_trainer)

#### SentencePiece Tokenizer
SentencePieceTrainer.train('--input="nx_ip_checksum_compute.c" --model_prefix=m --vocab_size=200')
SPP = SentencePieceProcessor()
SPP.load('m.model')
sentencepiece_encoded = SPP.encode_as_pieces(CProgram_string)

#%% Word Embeddings
#### Word2Vec
with open('nx_ip_checksum_compute.c', 'r') as code:
    lines = code.readlines()

print('The file (and thus the text corpus) contains {:d} lines of code'.format(len(lines)))
tokenized_sentences = [sentence.split() for sentence in lines]
word2vec_model = word2vec.Word2Vec(tokenized_sentences, vector_size = 10, window = 1, min_count = 0,
                                   workers = 10)

#%% FastText
FastText_model = FastText(vector_size = 4, window = 3, min_count = 1)
FastText_model.build_vocab(corpus_iterable = tokenized_sentences)
FastText_model.train(corpus_iterable = tokenized_sentences, total_examples = len(tokenized_sentences),
                     epochs = 10)