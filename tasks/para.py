"""
KeraSTS interface for datasets of the Paraphrasing task.  Basically,
it's a lot like STS but with binary output.  See data/para/... for
details and actual datasets.

Training example:
    tools/train.py cnn para data/para/msr/msr-para-train.tsv data/para/msr/msr-para-val.tsv inp_e_dropout=1/2
"""

from __future__ import print_function
from __future__ import division

from keras.callbacks import EarlyStopping, ModelCheckpoint
import numpy as np

import pysts.eval as ev
from pysts.kerasts import graph_input_anssel
import pysts.loader as loader
import pysts.nlp as nlp
from pysts.vocab import Vocabulary

from . import AbstractTask


class ParaphrasingTask(AbstractTask):
    def __init__(self):
        self.name = 'para'
        self.spad = 60
        self.s0pad = self.spad
        self.s1pad = self.spad
        self.emb = None
        self.vocab = None

    def config(self, c):
        c['loss'] = 'binary_crossentropy'
        c['nb_epoch'] = 32

    def load_set(self, fname):
        s0, s1, y = loader.load_msrpara(fname)

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
            return module_prep_model(self.vocab, output='binary')

        model = self.prep_model(module_prep_model)

        for lname in fix_layers:
            model.nodes[lname].trainable = False

        if do_compile:
            model.compile(loss={'score': self.c['loss']}, optimizer=optimizer)
        return model

    def fit_callbacks(self, weightsf):
        return [ModelCheckpoint(weightsf, save_best_only=True),
                EarlyStopping(patience=3)]

    def eval(self, model):
        res = []
        for gr, fname in [(self.gr, self.trainf), (self.grv, self.valf), (self.grt, self.testf)]:
            if gr is None:
                res.append(None)
                continue
            ypred = model.predict(gr)['score'][:,0]
            res.append(ev.eval_para(ypred, gr['score'], fname))
        return tuple(res)

    def res_columns(self, mres, pfx=' '):
        """ Produce README-format markdown table row piece summarizing
        important statistics """
        return('%s%.6f  |%s%.6f |%s%.6f |%s%.6f |%s%.6f |%s%.6f'
               % (pfx, mres[self.trainf]['Accuracy'],
                  pfx, mres[self.trainf]['F1'],
                  pfx, mres[self.valf]['Accuracy'],
                  pfx, mres[self.valf]['F1'],
                  pfx, mres[self.testf].get('Accuracy', np.nan),
                  pfx, mres[self.testf].get('F1', np.nan)))


def task():
    return ParaphrasingTask()
