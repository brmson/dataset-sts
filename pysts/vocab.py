"""
Vocabulary that indexes words, can handle OOV words and integrates word
embeddings.
"""


from __future__ import print_function

from keras.layers.embeddings import Embedding
from keras.preprocessing.sequence import pad_sequences

import numpy as np


class Vocabulary:
    """ word-to-index mapping, token sequence mapping tools and
    embedding matrix construction tools """
    def __init__(self, sentences):
        """ build a vocabulary from given list of sentences """

	vocab = sorted(list(set([t for s in sentences for t in s])))
	self.word_idx = dict((w, i + 2) for i, w in enumerate(vocab))
        self.word_idx['_PAD_'] = 0
        self.word_idx['_OOV_'] = 1

        self.embcache = dict()

    def vectorize(self, slist, spad=60):
        """ build an spad-ed matrix of word indices from a list of
        token sequences """
        silist = [[self.word_idx.get(t, 1) for t in s] for s in slist]
        return pad_sequences(silist, maxlen=spad, truncating='post', padding='post') 

    def embmatrix(self, emb):
        """ generate index-based embedding matrix from embedding class emb
        (typically GloVe); pass as weights= argument of Keras' Embedding layer """
        if str(emb) in self.embcache:
            return self.embcache[str(emb)]
        embedding_weights = np.zeros((len(self.word_idx), emb.N))
        for word, index in self.word_idx.items():
            try:
                embedding_weights[index, :] = emb.g[word]
            except KeyError:
                embedding_weights[index, :] = np.random.uniform(-0.25, 0.25, emb.N)  # 0.25 is embedding SD
        self.embcache[str(emb)] = embedding_weights
        return embedding_weights

    def size(self):
        return len(self.word_idx)
