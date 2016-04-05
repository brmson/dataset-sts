"""
A non-neural anssel-specific model that replicates the YodaQA baseline.

A trivial computation of score that is used as "PassScoreSimple" strategy
in the YodaQA system, simply a linear combination of matched keyword weight
sums.
"""

from __future__ import print_function
from __future__ import division

import numpy as np


def config(c):
    # disable GloVe
    c['embdim'] = None
    # disable Keras training
    c['ptscorer'] = None

    # weight term for aboutkwweights
    c['akw_c'] = 0.25


class YodaQAKWModel:
    """ Quacks (a little) like a Keras model. """

    def __init__(self, c, output):
        self.c = c
        self.output = output

    def fit(self, gr, **kwargs):
        # TODO: possibly fine-tune the weights
        pass

    def load_weights(self, f, **kwargs):
        pass

    def save_weights(self, f, **kwargs):
        pass

    def predict(self, gr):
        assert self.output == 'score'
        scores = []
        for kw, akw in zip(gr['kw'], gr['akw']):
            score = kw[0] + self.c['akw_c'] * akw[0]
            scores.append([score])
        scores = np.array(scores)
        return {'score': scores}


def prep_model(vocab, c, output='score'):
    return YodaQAKWModel(c, output)
