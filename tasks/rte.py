#!/usr/bin/python3
"""
KeraSTS interface for the Recognizing Textual Entailment task.

Training example:
	tools/train.py avg rte data/rte/sick2014/SICK_train.txt data/rte/sick2014/SICK_trial.txt inp_w_dropout=0.5
"""

from __future__ import print_function
from __future__ import division

from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.layers.core import Activation, Dense
from keras.models import Graph
from keras.regularizers import l2
import numpy as np

import pysts.eval as ev
from pysts.kerasts import graph_input_anssel
import pysts.kerasts.blocks as B
from pysts.kerasts.callbacks import RTECB
import pysts.loader as loader
import pysts.nlp as nlp
from pysts.vocab import Vocabulary

from .anssel import AbstractTask


class RTETask(AbstractTask):
    def __init__(self):
        self.name = 'rte'
        self.spad = 60
        self.s0pad = self.spad
        self.s1pad = self.spad
        self.emb = None
        self.vocab = None

    def config(self, c):
        c['loss'] = 'categorical_crossentropy'

    def load_set(self, fname):
        s0, s1, y = loader.load_sick2014(fname, mode='entailment')

        if self.vocab is None:
            vocab = Vocabulary(s0 + s1, prune_N=self.c['embprune'], icase=self.c['embicase'])
        else:
            vocab = self.vocab

        si0, sj0 = vocab.vectorize(s0, self.emb, spad=self.s0pad)
        si1, sj1 = vocab.vectorize(s1, self.emb, spad=self.s1pad)
        f0, f1 = nlp.sentence_flags(s0, s1, self.s0pad, self.s1pad)
        gr = graph_input_anssel(si0, si1, sj0, sj1, None, None, y, f0, f1, s0, s1)

        return (gr, y, vocab)

    def prep_model(self, module_prep_model):
        # Input embedding and encoding
        model = Graph()
        N = B.embedding(model, self.emb, self.vocab, self.s0pad, self.s1pad,
                        self.c['inp_e_dropout'], self.c['inp_w_dropout'], add_flags=self.c['e_add_flags'])

        # Sentence-aggregate embeddings
        final_outputs = module_prep_model(model, N, self.s0pad, self.s1pad, self.c)

        # Measurement

        if self.c['ptscorer'] == '1':
            # special scoring mode just based on the answer
            # (assuming that the question match is carried over to the answer
            # via attention or another mechanism)
            ptscorer = B.cat_ptscorer
            final_outputs = [final_outputs[1]]
        else:
            ptscorer = self.c['ptscorer']

        kwargs = dict()
        if ptscorer == B.mlp_ptscorer:
            kwargs['sum_mode'] = self.c['mlpsum']
            kwargs['Dinit'] = self.c['Dinit']

        model.add_node(name='scoreS0', input=ptscorer(model, final_outputs, self.c['Ddim'], N, self.c['l2reg'], pfx="out0", **kwargs),
                       layer=Activation('sigmoid'))
        model.add_node(name='scoreS1', input=ptscorer(model, final_outputs, self.c['Ddim'], N, self.c['l2reg'], pfx="out1", **kwargs),
                       layer=Activation('sigmoid'))
        model.add_node(name='scoreS2', input=ptscorer(model, final_outputs, self.c['Ddim'], N, self.c['l2reg'], pfx="out2", **kwargs),
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
        return [RTECB(self),
                ModelCheckpoint(weightsf, save_best_only=True, monitor='acc', mode='max'),
                EarlyStopping(monitor='acc', mode='max', patience=6)]

    def eval(self, model):
        res = []
        for gr, fname in [(self.gr, self.trainf), (self.grv, self.valf), (self.grt, self.testf)]:
            if gr is None:
                res.append(None)
                continue
            ypred = []
            for ogr in self.sample_pairs(gr, batch_size=len(gr), shuffle=False, once=True):
                ypred += list(model.predict(ogr)['score'])
            ypred = np.array(ypred)
            res.append(ev.eval_rte(ypred, gr['score'], fname))
        return tuple(res)

    def res_columns(self, mres, pfx=' '):
        """ Produce README-format markdown table row piece summarizing
        important statistics """
        return('%s%.6f |%s%.6f |%s%.6f'
               % (pfx, mres[self.trainf]['Accuracy'],
                  pfx, mres[self.valf]['Accuracy'],
                  pfx, mres[self.testf].get('Accuracy', np.nan)))

def task():
    return RTETask()
