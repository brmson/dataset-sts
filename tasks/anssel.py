"""
KeraSTS interface for datasets of the Answer Sentence Selection task,
i.e. bipartite ranking task of s1 candidates within a single s0 context.
See data/anssel/... for details and actual datasets.

Training example:
    tools/train.py cnn anssel data/anssel/wang/train-all.csv data/anssel/wang/dev.csv inp_e_dropout=1/2
"""

from __future__ import print_function
from __future__ import division

# TODO: cPickle fallfront?
import pickle

from keras.callbacks import EarlyStopping
from keras.layers.core import Activation
from keras.models import Graph
import numpy as np

import pysts.eval as ev
from pysts.kerasts import graph_input_anssel
import pysts.kerasts.blocks as B
from pysts.kerasts.callbacks import AnsSelCB
from pysts.kerasts.objectives import ranknet
import pysts.loader as loader
import pysts.nlp as nlp
from pysts.vocab import Vocabulary


class AnsSelTask:
    def __init__(self):
        self.name = 'anssel'
        self.s0pad = 60
        self.s1pad = 60
        self.emb = None
        self.vocab = None

    def config(self, c):
        c['loss'] = ranknet
        c['nb_epoch'] = 16
        c['epoch_fract'] = 1/4

    def load_set(self, fname, cache_dir=None):
        # TODO: Make the cache-handling generic,
        # and offer a way to actually pass cache_dir
        save_cache = False
        if cache_dir:
            import os.path
            fname_abs = os.path.abspath(fname)
            from hashlib import md5
            cache_filename = "%s/%s.p" % (cache_dir, md5(fname_abs.encode("utf-8")).hexdigest())
            try:
                with open(cache_filename, "rb") as f:
                    return pickle.load(f)
            except (IOError, TypeError, KeyError):
                save_cache = True

        s0, s1, y, t = loader.load_anssel(fname)
        # TODO: Make use of the t-annotations

        if self.vocab is None:
            vocab = Vocabulary(s0 + s1)
        else:
            vocab = self.vocab

        si0 = vocab.vectorize(s0, spad=self.s0pad)
        si1 = vocab.vectorize(s1, spad=self.s1pad)
        f0, f1 = nlp.sentence_flags(s0, s1, self.s0pad, self.s1pad)
        gr = graph_input_anssel(si0, si1, y, f0, f1, s0, s1)

        if save_cache:
            with open(cache_filename, "wb") as f:
                pickle.dump((s0, s1, y, vocab, gr), f)
                print("save")

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
            return module_prep_model(self.vocab, c)

        # ranking losses require wide output domain
        oact = 'sigmoid' if c['loss'] == 'binary_crossentropy' else 'linear'

        model = self.prep_model(module_prep_model, c, oact)

        for lname in fix_layers:
            model.nodes[lname].trainable = False

        if do_compile:
            model.compile(loss={'score': c['loss']}, optimizer=optimizer)
        return model

    def fit_callbacks(self):
        return [AnsSelCB(self.grv['s0'], self.grv),
                EarlyStopping(monitor='mrr', mode='max', patience=4)]

    def eval(self, model):
        res = []
        for gr, fname in [(self.gr, self.trainf), (self.grv, self.valf), (self.grt, self.testf)]:
            if gr is None:
                res.append(None)
                continue
            ypred = model.predict(gr)['score'][:,0]
            res.append(ev.eval_anssel(ypred, gr['s0'], gr['s1'], gr['score'], fname, MAP=True))
        return tuple(res)

    def res_columns(self, mres, pfx=' '):
        """ Produce README-format markdown table row piece summarizing
        important statistics """
        return('%c%.6f    |%c%.6f |%c%.6f |%c%.6f'
               % (pfx, mres[self.trainf]['MRR'],
                  pfx, mres[self.valf]['MRR'],
                  pfx, mres[self.testf].get('MAP', np.nan),
                  pfx, mres[self.testf].get('MRR', np.nan)))


def task():
    return AnsSelTask()
