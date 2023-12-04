#!/usr/bin/env python3
"""
Deep Learning With TensorFlow and Keras Third Edition (Packt Publishing) Chapter 4: Word Embeddings
"""
import os
import logging
import numpy as np
import gensim
import tensorflow as tf
from scipy.sparse import csr_matrix
from sklearn.metrics.pairwise import cosine_similarity
logging.basicConfig(format = '%(asctime)s : %(levelname)s : %(message)s', level = logging.INFO)

#%% node2vec
def download_and_read(url):
    local_file = url.split('/')[-1]
    p = tf.keras.utils.get_file(local_file, url, cache_dir = '.')
    file = open(p ,'r')
    row_ids = []
    col_ids = []
    data = []
    rid = 0
    for line in file:
        line = line.strip()
        if line.startswith("\"\","):
            continue
        # Compute the non-zero elements in the current row
        counts = np.array([int(x) for x in line.split(',')[1:]])
        nonzero_col_ids = np.nonzero(counts)[0]
        nonzero_data = counts[nonzero_col_ids]
        nonzero_row_ids = np.repeat(rid, len(nonzero_col_ids))
        rid += 1

        row_ids.extend(nonzero_row_ids.tolist())
        col_ids.extend(nonzero_col_ids.tolist())
        data.extend(nonzero_data.tolist())
    file.close()
    term_document_matrix = csr_matrix((np.array(data), (np.array(row_ids), np.array(col_ids))),
                                       shape = (rid, counts.shape[0]))
    return term_document_matrix

TD_matrix = download_and_read('https://archive.ics.uci.edu/ml/machine-learning-databases/00371/NIPS_1987-2015.csv')
embedding = TD_matrix.T * TD_matrix
# Sparse binarized adjacency matrix
embedding[embedding > 0] = 1

def construct_random_walks(embedding, n, alpha, l, outfile):
    if os.path.exists(outfile):
        print('Random walk has already been generated, skipping...')
        return
    file = open(outfile, 'w')
    for i in range(embedding.shape[0]):
        # Each vertex
        if i % 100 == 0:
            print('{} random walks have been generated from {} vertices'.format(n*i, i))
        for j in range(n):
            # Constructing n random walks
            current = i
            walk = [current]
            target_nodes = np.nonzero(embedding[current])[1]
            for k in range (l):
                if np.random.random() < alpha and len(walk) > 5:
                    break
                try:
                    # Select another outgoingedge and append to walk
                    current = np.random.choice(target_nodes)
                    walk.append(current)
                    target_nodes = np.nonzero(embedding[current])[1]
                except ValueError:
                    continue
            file.write('{}\n'.format(' '.join([str(x) for x in walk])))
    print('{} random walks have been generated from {} vertices, COMPLETE!'.format(n*i, i))
    file.close()

construct_random_walks(embedding, 32, 0.15, 40, 'Data/random-walks.txt')

class Documents(object):
    def __init__(self, input_file):
        self.input_file = input_file

    def __iter__(self):
        with open(self.input_file, 'r') as file:
            for index, line in enumerate(file):
                if index % 1000 == 0:
                    logging.info('{} random walks extracted'.format(index))
                yield line.strip().split()

def train_word2vec_model(random_walks_file, model_file):
    if os.path.exists(model_file):
        print('Model located at {}, skipping training...'.format(model_file))
        return

    documents = Documents(random_walks_file)
    # sg: skip-gram model
    model = gensim.models.Word2Vec(documents, vector_size = 128, window = 10, sg = 1, min_count = 2,
                                   workers = 4)
    model.train(documents, total_examples = model.corpus_count, epochs = 50)
    model.save(model_file)

train_word2vec_model('Data/random-walks.txt', 'Data/word2vec-neurips-papers.model')

def evaluate_model(TD_matrix, model_file, source_id):
    model = gensim.models.Word2Vec.load(model_file).wv
    most_similar = model.most_similar(str(source_id))
    scores = [x[1] for x in most_similar]
    target_ids = [int(x[0]) for x in most_similar]
    # Compare the top 10 scores using cosine_similarity between the source and each target
    X = np.repeat(TD_matrix[source_id].todense(), 10, axis = 0)
    Y = TD_matrix[target_ids].todense()
    cos_sims = [cosine_similarity(np.array(X[i]), np.array(Y[i]))[0, 0] for i in range(10)]
    for j in range(10):
        print('{} {} {:.3f} {:.3f}'.format(source_id, target_ids[j], cos_sims[j], scores[j]))

source_id = np.random.choice(embedding.shape[0])
evaluate_model(TD_matrix, 'Data/word2vec-neurips-papers.model', source_id)