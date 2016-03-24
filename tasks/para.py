"""
KeraSTS interface for datasets of the Paraphrasing task.  Basically,
it's a lot like STS but with binary output.  See data/para/... for
details and actual datasets.

Training example:
    tools/train.py cnn para data/para/msr/msr-para-train.tsv data/para/msr/msr-para-val.tsv inp_e_dropout=1/2
"""

from __future__ import print_function
from __future__ import division

from keras.callbacks import EarlyStopping
from keras.layers.core import Activation
from keras.models import Graph
import numpy as np

import pysts.eval as ev
from pysts.kerasts import graph_input_anssel
import pysts.kerasts.blocks as B
import pysts.loader as loader
import pysts.nlp as nlp
from pysts.vocab import Vocabulary


class ParaphrasingTask:
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

    def load_vocab(self, vocabf):
        _, _, self.vocab = self.load_set(vocabf)
        return self.vocab

    def load_data(self, trainf, valf, testf=None):
        self.trainf = trainf
        self.valf = valf
        self.testf = testf

        self.gr, self.y, self.vocab = self.load_set(trainf)
        self.grv, self.yv, _ = self.load_set(valf)
        if testf is not None:
            self.grt, self.yt, _ = self.load_set(testf)
        else:
            self.grt, self.yt = (None, None)

    def prep_model(self, module_prep_model, c, oact='sigmoid'):
        # Input embedding and encoding
        model = Graph()
        N = B.embedding(model, self.emb, self.vocab, self.s0pad, self.s1pad, c['inp_e_dropout'], c['inp_w_dropout'], add_flags=c['e_add_flags'])

        # Sentence-aggregate embeddings
        final_outputs = module_prep_model(model, N, self.s0pad, self.s1pad, c)

        # Measurement

        if c['ptscorer'] == '1':
            # special scoring mode just based on the answer
            # (assuming that the question match is carried over to the answer
            # via attention or another mechanism)
            ptscorer = B.cat_ptscorer
            final_outputs = final_outputs[1]
        else:
            ptscorer = c['ptscorer']

        kwargs = dict()
        if ptscorer == B.mlp_ptscorer:
            kwargs['sum_mode'] = c['mlpsum']
        model.add_node(name='scoreS', input=ptscorer(model, final_outputs, c['Ddim'], N, c['l2reg'], **kwargs),
                       layer=Activation(oact))
        model.add_output(name='score', input='scoreS')
        return model

    def build_model(self, module_prep_model, c, optimizer='adam', fix_layers=[], do_compile=True):
        if c['ptscorer'] is None:
            # non-neural model
            return module_prep_model(self.vocab, c, output='binary')

        model = self.prep_model(module_prep_model, c)

        for lname in fix_layers:
            model.nodes[lname].trainable = False

        if do_compile:
            model.compile(loss={'score': c['loss']}, optimizer=optimizer)
        return model

    def fit_callbacks(self):
        return [EarlyStopping(patience=3)]

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
        return('%c%.6f  |%c%.6f |%c%.6f |%c%.6f |%c%.6f |%c%.6f'
               % (pfx, mres[self.trainf]['Accuracy'],
                  pfx, mres[self.trainf]['F1'],
                  pfx, mres[self.valf]['Accuracy'],
                  pfx, mres[self.valf]['F1'],
                  pfx, mres[self.testf].get('Accuracy', np.nan),
                  pfx, mres[self.testf].get('F1', np.nan)))


def task():
    return ParaphrasingTask()
