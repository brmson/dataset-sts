"""
Vocabulary that indexes words, can handle OOV words and integrates word
embeddings.
"""


from __future__ import print_function

from collections import defaultdict
import numpy as np
from operator import itemgetter

from keras.layers.embeddings import Embedding
from keras.preprocessing.sequence import pad_sequences


class Vocabulary:
    """ word-to-index mapping, token sequence mapping tools and
    embedding matrix construction tools """
    def __init__(self, sentences, count_thres=1):
        """ build a vocabulary from given list of sentences, but including
        only words occuring at least #count_thres times """

        # Counter() is superslow :(
        vocabset = defaultdict(int)
        for s in sentences:
            for t in s:
                vocabset[t] += 1

        vocab = sorted(list(map(itemgetter(0),
                                filter(lambda k: itemgetter(1)(k) >= count_thres,
                                       vocabset.items() ) )))
        self.word_idx = dict((w, i + 2) for i, w in enumerate(vocab))
        self.word_idx['_PAD_'] = 0
        self.word_idx['_OOV_'] = 1

        self.embcache = dict()

    def vectorize(self, slist, spad=60):
        """ build an spad-ed matrix of word indices from a list of
        token sequences """
        silist = [[self.word_idx.get(t, 1) for t in s] for s in slist]
        if spad is not None:
            return pad_sequences(silist, maxlen=spad, truncating='post', padding='post') 
        else:
            return silist

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
