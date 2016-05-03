#!/usr/bin/python3
"""
KeraSTS interface for the SNLI dataset of the Textual Entailment task.

Training example:
	tools/train.py avg snli data/rte/snli/snli_1.0_train.pickle data/rte/snli/snli_1.0_dev.pickle vocabf="data/rte/snli/v1-vocab.pickle" inp_w_dropout=0.5

Before training, you must however run:

	tools/snli_preprocess.py --revocab data/rte/snli/snli_1.0/snli_1.0_train.jsonl data/rte/snli/snli_1.0/snli_1.0_dev.jsonl data/rte/snli/snli_1.0/snli_1.0_test.jsonl data/rte/snli/snli_1.0_train.pickle data/rte/snli/snli_1.0_dev.pickle data/rte/snli/snli_1.0_test.pickle data/rte/snli/v1-vocab.pickle
"""

from __future__ import print_function
from __future__ import division


from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.layers.core import Activation, Dropout
from keras.models import Graph

import pickle
import pysts.eval as ev
import numpy as np


from pysts.kerasts import graph_input_anssel
import pysts.kerasts.blocks as B
from pysts.kerasts.callbacks import RTECB

from .anssel import AbstractTask

class SnliTask(AbstractTask):
    def __init__(self):
        self.name = 'snli'
        self.spad = 60
        self.s0pad = self.spad
        self.s1pad = self.spad
        self.emb = None
        self.vocab = None

    def config(self, c):
        c['loss'] = 'categorical_crossentropy'
        c['nb_epoch'] = 32
        c['batch_size'] = 200
        c['epoch_fract'] = 1/4

    def load_vocab(self, vocabf):
        # use plain pickle because unicode
        self.vocab = pickle.load(open(vocabf, "rb"))
        return self.vocab

    def load_set(self, fname):
        si0, si1, f0, f1, labels = pickle.load(open(fname, "rb"))
        gr = graph_input_anssel(si0, si1, labels, f0, f1)
        return (gr, labels, self.vocab)

    def prep_model(self, module_prep_model):
        # Input embedding and encoding
        model = Graph()
        N = B.embedding(model, self.emb, self.vocab, self.s0pad, self.s1pad,
                        self.c['inp_e_dropout'], self.c['inp_w_dropout'], add_flags=self.c['e_add_flags'])

        # Sentence-aggregate embeddings
        final_outputs = module_prep_model(model, N, self.s0pad, self.s1pad, self.c)

        # Measurement

        kwargs = dict()
        if self.c['ptscorer'] == B.mlp_ptscorer:
            kwargs['sum_mode'] = self.c['mlpsum']

        model.add_node(name='scoreS0', input=self.c['ptscorer'](model, final_outputs, self.c['Ddim'], N, self.c['l2reg'], pfx="out0", **kwargs),
                       layer=Activation('sigmoid'))
        model.add_node(name='scoreS1', input=self.c['ptscorer'](model, final_outputs, self.c['Ddim'], N, self.c['l2reg'], pfx="out1", **kwargs),
                       layer=Activation('sigmoid'))
        model.add_node(name='scoreS2', input=self.c['ptscorer'](model, final_outputs, self.c['Ddim'], N, self.c['l2reg'], pfx="out2", **kwargs),
                       layer=Activation('sigmoid'))

        model.add_node(name='scoreV', inputs=['scoreS0', 'scoreS1', 'scoreS2'], merge_mode='concat', layer=Activation('softmax'))
        model.add_output(name='score', input='scoreV')
        return model

    def build_model(self, module_prep_model, do_compile=True):
        if self.c['ptscorer'] is None:
            # non-neural model
            return module_prep_model(self.vocab, self.c, output='binary')  # FIXME

        model = self.prep_model(module_prep_model)

        for lname in self.c['fix_layers']:
            model.nodes[lname].trainable = False

        if do_compile:
            model.compile(loss={'score': self.c['loss']}, optimizer=self.c['opt'])
        return model

    def fit_callbacks(self, weightsf):
        return [RTECB(self.grv),
                ModelCheckpoint(weightsf, save_best_only=True, monitor='acc', mode='max'),
                EarlyStopping(monitor='acc', mode='max', patience=6)]

    def eval(self, model):
        res = []
        for gr, fname in [(self.gr, self.trainf), (self.grv, self.valf), (self.grt, self.testf)]:
            if gr is None:
                res.append(None)
                continue
            ypred = model.predict(gr)['score']
            res.append(ev.eval_snli(ypred, gr['score'], fname))
        return tuple(res)

    def res_columns(self, mres, pfx=' '):
        """ Produce README-format markdown table row piece summarizing
        important statistics """
        return('%s%.6f |%s%.6f |%s%.6f'
               % (pfx, mres[self.trainf]['Accuracy'],
                  pfx, mres[self.valf]['Accuracy'],
                  pfx, mres[self.testf].get('Accuracy', np.nan)))

def task():
    return SnliTask()
