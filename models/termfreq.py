"""
A non-neural model based just on word overlaps.

This general model can count TFIDF as well as Okapi BM25,
count (un)weighed overlaps as well as compute cosine distance,
deal with boolean as well as raw frequencies...

At least it's designed to, some TODO items remain.

Stopwords and punctuation are removed and terms are normalized
to lowercase whenever possible, based on Yih et al., 2013
(http://research.microsoft.com/pubs/192357/QA-SentSel-Updated-PostACL.pdf).

BM25 parameters (and the whole idea) come from Wang+Nyberg, 2015
(http://www.aclweb.org/anthology/P15-2116).

Defaults are tuned for anssel-wang devMRR.
"""

from __future__ import print_function
from __future__ import division

from collections import defaultdict, Counter
import h5py
from nltk.corpus import stopwords
import numpy as np
import re

import pysts.loader as loader

stop = stopwords.words('english')


def config(c):
    # disable GloVe
    c['embdim'] = None
    # disable Keras training
    c['ptscorer'] = None

    # weight terms by their inverse document frequency
    c['idf'] = True

    # frequency computation mode can be either:
    # * 'tf' for normal tf-idf with raw counts (N.B. raw/bool has almost no
    #   effect as there are few term repetitions in the sentences)
    # * 'BM25' for the Okapi BM25; c['K1'] and c['B'] parameters are used
    c['freq_mode'] = 'BM25'

    # Okapi BM25 parameters; from http://www.aclweb.org/anthology/P15-2116:
    c['K1'] = 1.2
    c['B'] = 0.75

    # scoring may be performed using either:
    # * 'overlap' (count overlapping words, possibly reweighting it)
    # * 'cos' (compute tfidf vectors for each sentence, then measure
    #   cosine distance)
    c['score_mode'] = 'overlap'


class TFVec:
    """ tf(idf) sparse vector (dict-based) """

    def __init__(self, s, idf, conf, avglen):
        """ idf is an (idfdict, oovidfval) tuple """
        self.w = dict()
        for w, c in Counter(s).items():
            if w == '':
                continue

            if conf['freq_mode'] == 'tf':
                x = c
            elif conf['freq_mode'] == 'BM25':
                x = (c * (conf['K1'] + 1)) / (c + conf['K1'] * (1 - conf['B'] + conf['B'] * len(s) / avglen))

            if idf is not None:
                x *= idf[0].get(w, idf[1])
            self.w[w] = x

    def norm(self):
        return np.sum(list(self.w.values()))

    def dot(self, v2):
        x = 0
        for w in set(self.w.keys()) & set(v2.w.keys()):
            x += self.w[w] * v2.w[w]
        return x

    def cos(self, v2):
        return self.dot(v2) / (self.norm() * v2.norm())

    def overlap(self, v2):
        """ sum tfidf scores of v2 words overlapping with v1 """
        return np.sum([score for w, score in v2.w.items() if w in self.w])


class TFModel:
    """ Quacks (a little) like a Keras model. """

    def __init__(self, c, output):
        self.c = c
        self.output = output

    def fit(self, gr, **kwargs):
        # our "fitting" is just computing the idf table; for BM25,
        # we also need average sentence length
        lens = []
        if self.c['idf']:
            self.N = len(gr['s0'])
            counter = defaultdict(float)
            for i in range(len(gr['s0'])):
                for k in ['s0', 's1']:
                    for w in gr[k][i]:
                        counter[self._norm(w)] += 1
                    lens.append(len(gr[k]))

            if self.c['freq_mode'] == 'tf':
                # basic idf
                for k, v in counter.items():
                    counter[k] = np.log(self.N / (v + 1))

            elif self.c['freq_mode'] == 'BM25':
                # Okapi idf
                for k, v in counter.items():
                    counter[k] = np.log((self.N - v + 0.5) / (v + 0.5)) if v < self.N else 0.1

            self.idf = counter

        self.avglen = np.mean(lens)

    def load_weights(self, f, **kwargs):
        h5 = h5py.File(f, "r")
        self.avglen = h5['avglen'].value
        self.N = h5['N'].value
        self.idf = defaultdict(float)
        for w, v in h5['idf'].items():
            self.idf[w.replace('__SL__', '/')] = v.value

    def save_weights(self, f, **kwargs):
        h5 = h5py.File(f, "w")
        h5.create_dataset('avglen', data=self.avglen)
        h5.create_dataset('N', data=self.N)
        for w, v in self.idf.items():
            if w == '':
                continue
            h5.create_dataset('idf/'+w.replace('/', '__SL__'), data=v)

    def predict(self, gr):
        scores = []
        for i in range(len(gr['s0'])):
            s0 = [self._norm(w) for w in gr['s0'][i]]
            s1 = [self._norm(w) for w in gr['s1'][i]]
            scores.append([self._score(s0, s1)])
        scores = np.array(scores)
        if self.output == 'score':
            return {'score': scores}
        elif self.output == 'binary':
            # XXX: we should tune the threshold to maximize accuracy
            scores0 = scores - np.min(scores)
            return {'score': scores0 * 5. / np.max(scores0)}
        elif self.output == 'classes':
            scores0 = scores - np.min(scores)
            return {'classes': loader.sts_labels2categorical(scores0 * 5. / np.max(scores0))}

    def _norm(self, w):
        """ map punctuation and stopwords to 0, non-lowercase words to lowercase indices """
        if w in stop:
            return ''
        if re.match('^[,.:;`\'"!?()/-]+$', w):
            return ''
        return w.lower()

    def _score(self, s0, s1):
        idf = (self.idf, np.log(self.N)) if self.c['idf'] else None
        tf0 = TFVec(s0, idf, self.c, self.avglen)
        tf1 = TFVec(s1, idf, self.c, self.avglen)
        if self.c['score_mode'] == 'cos':
            return tf0.cos(tf1)
        elif self.c['score_mode'] == 'overlap':
            return tf0.overlap(tf1)
        else:
            raise ValueError


def prep_model(vocab, c, output='score'):
    # TODO: the output parameter is there for sts, output='classes'
    return TFModel(c, output)
