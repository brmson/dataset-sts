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
        self.emb = None

    def config(self, c):
        c['loss'] = 'binary_crossentropy'
        c['nb_epoch'] = 32

    def load_set(self, fname, vocab=None):
        s0, s1, y = loader.load_msrpara(fname)

        if vocab is None:
            vocab = Vocabulary(s0 + s1)

        si0 = vocab.vectorize(s0, spad=self.spad)
        si1 = vocab.vectorize(s1, spad=self.spad)
        f0, f1 = nlp.sentence_flags(s0, s1, self.spad, self.spad)
        gr = graph_input_anssel(si0, si1, y, f0, f1, s0, s1)

        return (gr, y, vocab)

    def load_data(self, trainf, valf, vocab=None):
        self.gr, self.y, self.vocab = self.load_set(trainf, vocab=vocab)
        self.grv, self.yv, _ = self.load_set(valf, self.vocab)

    def prep_model(self, module_prep_model, c):
        # Input embedding and encoding
        model = Graph()
        N = B.embedding(model, self.emb, self.vocab, self.spad, self.spad, c['inp_e_dropout'], c['inp_w_dropout'], add_flags=c['e_add_flags'])

        # Sentence-aggregate embeddings
        final_outputs = module_prep_model(model, N, self.spad, self.spad, c)

        # Measurement
        kwargs = dict()
        if c['ptscorer'] == B.mlp_ptscorer:
            kwargs['sum_mode'] = c['mlpsum']
        model.add_node(name='scoreS', input=c['ptscorer'](model, final_outputs, c['Ddim'], N, c['l2reg'], **kwargs),
                       layer=Activation('sigmoid'))
        model.add_output(name='score', input='scoreS')
        return model

    def build_model(self, module_prep_model, c, optimizer='adam', fix_layers=[]):
        if c['ptscorer'] is None:
            # non-neural model
            return module_prep_model(self.vocab, c, output='binary')

        model = self.prep_model(module_prep_model, c)

        for lname in fix_layers:
            model.nodes[lname].trainable = False

        model.compile(loss={'score': c['loss']}, optimizer=optimizer)
        return model

    def fit_callbacks(self):
        return [EarlyStopping(patience=3)]

    def eval(self, model):
        ev.eval_para(model.predict(self.gr)['score'][:,0], self.gr['score'], 'Train')
        ev.eval_para(model.predict(self.grv)['score'][:,0], self.grv['score'], 'Val')


def task():
    return ParaphrasingTask()
