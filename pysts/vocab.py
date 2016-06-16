"""
Vocabulary that indexes words, can handle OOV words and integrates word
embeddings.
"""


from __future__ import print_function

from collections import defaultdict
import numpy as np
from operator import itemgetter

from keras.preprocessing.sequence import pad_sequences


class Vocabulary:
    """ word-to-index mapping, token sequence mapping tools and
    embedding matrix construction tools """
    def __init__(self, sentences, count_thres=1, prune_N=None, icase=False):
        """ build a vocabulary from given list of sentences, but including
        only words occuring at least #count_thres times; prune_N, if set,
        denotes the only top N most occuring tokens to retain in vocab. """

        # Counter() is superslow :(
        vocabset = defaultdict(int)
        for s in sentences:
            for t in s:
                vocabset[t if not icase else t.lower()] += 1

        vocab = list(map(itemgetter(0),
                         sorted(filter(lambda k: itemgetter(1)(k) >= count_thres,
                                       vocabset.items()),
                                key=itemgetter(1, 0), reverse=True)))
        vocab_N = len(vocab)
        if prune_N is not None:
            vocab = vocab[:prune_N]
        self.word_idx = dict((w, i + 2) for i, w in enumerate(vocab))
        self.word_idx['_PAD_'] = 0
        self.word_idx['_OOV_'] = 1
        print('Vocabulary of %d words (adaptable: %d)' % (vocab_N, len(self.word_idx)))
        print(vocab)

        self.embcache = dict()
        self.icase = icase

    def add_word(self, word):
        if word not in self.word_idx:
            self.word_idx[word] = len(self.word_idx)

    def vectorize(self, slist, emb, spad=60):
        """ build an spad-ed matrix of word indices from a list of
        token sequences; returns an si, sj tuple of indices in vocab
        and emb respectively """
        silist = []
        sjlist = []
        for s in slist:
            si = []
            sj = []
            for t in s:
                if self.icase:
                    t = t.lower()
                if t in self.word_idx:
                    si.append(self.word_idx[t])
                    sj.append(0)
                elif emb is not None and t in emb.w:
                    si.append(0)
                    sj.append(emb.w[t])
                else:
                    si.append(1)  # OOV
                    sj.append(0)
            silist.append(si)
            sjlist.append(sj)
        if spad is not None:
            return (pad_sequences(silist, maxlen=spad, truncating='post', padding='post'),
                    pad_sequences(sjlist, maxlen=spad, truncating='post', padding='post'))
        else:
            return (silist, sjlist)

    def embmatrix(self, emb):
        """ generate index-based embedding matrix from embedding class emb
        (typically GloVe); pass as weights= argument of Keras' Embedding layer """
        if str(emb) in self.embcache:
            return self.embcache[str(emb)]
        embedding_weights = np.zeros((len(self.word_idx), emb.N))
        for word, index in self.word_idx.items():
            try:
                embedding_weights[index, :] = emb.g[emb.w[word]]
            except KeyError:
                if index == 0:
                    embedding_weights[index, :] = np.zeros(emb.N)
                else:
                    embedding_weights[index, :] = np.random.uniform(-0.25, 0.25, emb.N)  # 0.25 is embedding SD
        self.embcache[str(emb)] = embedding_weights
        return embedding_weights

    def size(self):
        return len(self.word_idx)
