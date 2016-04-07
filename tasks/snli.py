from __future__ import print_function
from __future__ import division

import importlib
import sys

from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.layers.core import Activation, Dropout
from keras.layers.recurrent import SimpleRNN, GRU, LSTM
from keras.models import Graph
import numpy as np

import pickle
import pysts.embedding as emb
import pysts.eval as ev
import pysts.loader as loader
import pysts.nlp as nlp
from pysts.hyperparam import hash_params
from pysts.vocab import Vocabulary

from pysts.kerasts import graph_input_anssel
import pysts.kerasts.blocks as B
from pysts.kerasts.callbacks import AnsSelCB
from pysts.kerasts.objectives import ranknet, ranksvm, cicerons_1504
import models  # importlib python3 compatibility requirement

from .anssel import AbstractTask

class SnliTask(AbstractTask):
    def __init__(self):
        self.name = 'snli'
        self.spad=60
        self.s0pad = self.spad
        self.s1pad= self.spad
        self.emb = None
        self.vocab = None

    def load_vocab(self, vocabf):
        # use plain pickle because unicode
        self.vocab = pickle.load(open(vocabf, "rb"))
        return self.vocab

    def load_set(self,fname, vocab):
        s0, s1, y = loader.load_snli(fname, vocab)
        si0 = vocab.vectorize(s0, spad=self.s0pad)
        si1 = vocab.vectorize(s1, spad=self.s1pad)
        f0, f1 = nlp.sentence_flags(s0, s1, self.s0pad, self.s1pad)
        gr = graph_input_anssel(si0, si1, y, f0, f1, s0, s1)
        return (si0, si1, y, gr)


    def config(self, c):
        c['loss'] = 'categorical_crossentropy'
        c['nb_epoch'] = 32
        c['batch_size'] = 160

    def build_model(self, module_prep_model, do_compile=True):
        if self.c['ptscorer'] is None:
            # non-neural model
            return module_prep_model(self.vocab, self.c, output='binary')

        model = self.prep_model(module_prep_model)

        for lname in self.c['fix_layers']:
            model.nodes[lname].trainable = False

        if do_compile:
            model.compile(loss={'score': self.c['loss']}, optimizer=self.c['opt'])
        return model

    def eval(self, model):
        res = [None]
        for gr, fname in [(self.grv, self.valf), (self.grt, self.testf)]:
            if gr is None:
                res.append(None)
                continue
            ypred = model.predict(gr)['score']
            res.append(ev.eval_snli(ypred, gr['score'], fname))
        return tuple(res)

    def eval(self, model):
        ev.eval_snli(model.predict(self.gr)['score'], self.gr['score'], 'Train')
        ev.eval_snli(model.predict(self.grv)['score'], self.grv['score'], 'Val')

    def prep_model(self,module_prep_model):
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
        model.add_node(name='scoreS0', input=self.c['ptscorer'](model, final_outputs, self.c['Ddim'], N, self.c['l2reg'],pfx="out0", **kwargs),
                       layer=Activation('sigmoid'))

        model.add_node(name='scoreS1', input=self.c['ptscorer'](model, final_outputs, self.c['Ddim'], N, self.c['l2reg'],pfx="out1", **kwargs),
                       layer=Activation('sigmoid'))

        model.add_node(name='scoreS2', input=self.c['ptscorer'](model, final_outputs, self.c['Ddim'], N, self.c['l2reg'],pfx="out2", **kwargs),
                       layer=Activation('sigmoid'))

        model.add_node(name='scoreV', inputs=['scoreS0', 'scoreS1', 'scoreS2'], merge_mode='concat', layer=Activation('softmax'))

        model.add_output(name='score', input='scoreV')
        return model


    def build_model(self, module_prep_model, do_compile=True):
        if self.c['ptscorer'] is None:
            # non-neural model
            return module_prep_model(self.vocab, self.c, output='binary', spad=self.spad)
        model = self.prep_model(module_prep_model)

        if do_compile:
            model.compile(loss={'score': self.c['loss']}, optimizer=self.c['opt'])
        return model

    def fit_callbacks(self, weightsf):
        return [ModelCheckpoint(weightsf, save_best_only=True),
                EarlyStopping(patience=3)]





def train_and_eval(runid, module_prep_model, c, glove, vocab, gr, grt, do_eval=True):
    print('Model')
    model = build_model(glove, vocab, module_prep_model, c)

    print('Training')
    if c.get('balance_class', False):
        one_ratio = np.sum(gr['score'] == 1) / len(gr['score'])
        class_weight = {'score': {0: one_ratio, 1: 0.5}}
    else:
        class_weight = {}
    # XXX: samples_per_epoch is in brmson/keras fork, TODO fit_generator()?
    model.fit(gr, validation_data=grt,
              callbacks=[ModelCheckpoint('snli-weights-'+runid+'-bestval.h5', save_best_only=True),
                         EarlyStopping(patience=3)],
              batch_size=c['batch_size'], nb_epoch=c['nb_epoch'])
    model.save_weights('snli-weights-'+runid+'-final.h5', overwrite=True)
    if c['ptscorer'] is None:
        model.save_weights('snli-weights-'+runid+'-bestval.h5', overwrite=True)
    model.load_weights('snli-weights-'+runid+'-bestval.h5')

    if do_eval:
        print('Predict&Eval (best val epoch)')
        ev.eval_snli(model.predict(gr)['score'], gr['score'], 'Train')
        ev.eval_snli(model.predict(grt)['score'], grt['score'], 'Val')
    return model