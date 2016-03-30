"""
KeraSTS interface for datasets of the Hypothesis Evaluation task,
i.e. producing an aggregate s0 classification based on the set of its
s1 evidence (typically mix of relevant and irrelevant).
See data/hypev/... for details and actual datasets.

Training example:
    tools/train.py cnn hypev data/hypev/argus/argus_train.csv data/hypev/argus/argus_test.csv dropout=0
"""

from __future__ import print_function
from __future__ import division

from keras.callbacks import EarlyStopping, ModelCheckpoint
import numpy as np

import pysts.eval as ev
from pysts.kerasts import graph_input_anssel
from pysts.kerasts.callbacks import HypEvCB
import pysts.loader as loader
import pysts.nlp as nlp
from pysts.vocab import Vocabulary

from . import AbstractTask


class HypEvTask(AbstractTask):
    def __init__(self):
        self.name = 'hypev'
        self.s0pad = 60
        self.s1pad = 60
        self.emb = None
        self.vocab = None

    def config(self, c):
        c['loss'] = 'binary_crossentropy'
        c['nb_epoch'] = 4
        c['epoch_fract'] = 1

    def load_set(self, fname, cache_dir=None):
        s0, s1, y = loader.load_hypev(fname)

        if self.vocab is None:
            vocab = Vocabulary(s0 + s1)
        else:
            vocab = self.vocab

        si0 = vocab.vectorize(s0, spad=self.s0pad)
        si1 = vocab.vectorize(s1, spad=self.s1pad)
        f0, f1 = nlp.sentence_flags(s0, s1, self.s0pad, self.s1pad)
        gr = graph_input_anssel(si0, si1, y, f0, f1, s0, s1)

        return (gr, y, vocab)

    def build_model(self, module_prep_model, optimizer='adam', fix_layers=[], do_compile=True):
        if self.c['ptscorer'] is None:
            # non-neural model
            return module_prep_model(self.vocab, self.c)

        # ranking losses require wide output domain
        oact = 'sigmoid' if self.c['loss'] == 'binary_crossentropy' else 'linear'

        model = self.prep_model(module_prep_model, oact=oact)

        for lname in fix_layers:
            model.nodes[lname].trainable = False

        if do_compile:
            model.compile(loss={'score': self.c['loss']}, optimizer=optimizer)
        return model

    def fit_callbacks(self, weightsf):
        return [HypEvCB(self.grv['si0'], self.grv),
                ModelCheckpoint(weightsf, save_best_only=True, monitor='acc', mode='max'),
                EarlyStopping(monitor='acc', mode='max', patience=4)]

    def eval(self, model):
        res = []
        for gr, fname in [(self.gr, self.trainf), (self.grv, self.valf), (self.grt, self.testf)]:
            if gr is None:
                res.append(None)
                continue
            ypred = model.predict(gr)['score'][:,0]
            res.append(ev.eval_hypev(ypred, gr['si0'], gr['score'], fname))
        return tuple(res)

    def res_columns(self, mres, pfx=' '):
        """ Produce README-format markdown table row piece summarizing
        important statistics """
        return('%s%.6f |%s%.6f |%s%.6f'
               % (pfx, mres[self.trainf]['QAccuracy'],
                  pfx, mres[self.valf]['QAccuracy'],
                  pfx, mres[self.testf].get('QAccuracy', np.nan)))


def task():
    return HypEvTask()
