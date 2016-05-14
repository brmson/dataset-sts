"""
KeraSTS interface for datasets of the Answer Sentence Selection task,
i.e. bipartite ranking task of s1 candidates within a single s0 context.
See data/anssel/... for details and actual datasets.

Training example:
    tools/train.py cnn anssel data/anssel/wang/train-all.csv data/anssel/wang/dev.csv inp_e_dropout=1/2

Specific config parameters:

    * prescoring_prune=N to prune all but top N pre-scored s1s before
      main scoring

    * prescoring_input='bm25' to add an extra input called 'bm25' to the
      graph, which can be then included as an additional scoring feature
      by the ``f_add`` option.
"""

from __future__ import print_function
from __future__ import division

# TODO: cPickle fallfront?
import pickle

from keras.callbacks import EarlyStopping, ModelCheckpoint
import numpy as np

import pysts.eval as ev
from pysts.kerasts import graph_input_anssel, graph_input_unprune
from pysts.kerasts.callbacks import AnsSelCB
from pysts.kerasts.objectives import ranknet
import pysts.loader as loader
import pysts.nlp as nlp
from pysts.vocab import Vocabulary

from . import AbstractTask


class AnsSelTask(AbstractTask):
    def __init__(self):
        self.name = 'anssel'
        self.s0pad = 60
        self.s1pad = 60
        self.emb = None
        self.vocab = None

        self.prescoring_task = AnsSelTask

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

        skip_oneclass = self.c.get('skip_oneclass', True)
        s0, s1, y, kw, akw, t = loader.load_anssel(fname, skip_oneclass=skip_oneclass)
        # TODO: Make use of the t-annotations

        if self.vocab is None:
            vocab = Vocabulary(s0 + s1, prune_N=self.c['embprune'], icase=self.c['embicase'])
        else:
            vocab = self.vocab

        si0, sj0 = vocab.vectorize(s0, self.emb, spad=self.s0pad)
        si1, sj1 = vocab.vectorize(s1, self.emb, spad=self.s1pad)
        f0, f1 = nlp.sentence_flags(s0, s1, self.s0pad, self.s1pad)
        gr = graph_input_anssel(si0, si1, sj0, sj1, None, None, y, f0, f1, s0, s1, kw=kw, akw=akw)

        if save_cache:
            with open(cache_filename, "wb") as f:
                pickle.dump((s0, s1, y, vocab, gr), f)
                print("save")

        return (gr, y, vocab)

    def build_model(self, module_prep_model, do_compile=True):
        if self.c['ptscorer'] is None:
            # non-neural model
            return module_prep_model(self.vocab, self.c)

        # ranking losses require wide output domain
        oact = 'sigmoid' if self.c['loss'] == 'binary_crossentropy' else 'linear'

        model = self.prep_model(module_prep_model, oact=oact)

        for lname in self.c['fix_layers']:
            model.nodes[lname].trainable = False

        if do_compile:
            model.compile(loss={'score': self.c['loss']}, optimizer=self.c['opt'])
        return model

    def fit_callbacks(self, weightsf):
        return [AnsSelCB(self, self.grv_p),
                ModelCheckpoint(weightsf, save_best_only=True, monitor='mrr', mode='max'),
                EarlyStopping(monitor='mrr', mode='max', patience=4)]

    def fit_model(self, model, **kwargs):
        # Prepare the pruned datasets
        gr_p = self.prescoring_apply(self.gr, skip_oneclass=True)
        self.grv_p = self.prescoring_apply(self.grv)  # for the callback

        # Recompute epoch_fract based on the new train set size
        if self.c['epoch_fract'] != 1:
            kwargs['samples_per_epoch'] = int(len(gr_p['si0']) * self.c['epoch_fract'])

        return AbstractTask.fit_model(self, model, **kwargs)

    def eval(self, model):
        res = []
        for gr, fname in [(self.gr, self.trainf), (self.grv, self.valf), (self.grt, self.testf)]:
            if gr is None:
                res.append(None)
                continue

            # In case of prescoring pruning, we want to predict only
            # on the prescoring subset, but evaluate on the complete
            # dataset, actually!  Therefore, we then unprune again.
            # TODO: Cache the pruning
            gr_p = self.prescoring_apply(gr)
            ypred = self.predict(model, gr_p)
            gr, ypred = graph_input_unprune(gr, gr_p, ypred, 0. if self.c['loss'] == 'binary_crossentropy' else float(-1e15))

            res.append(ev.eval_anssel(ypred, gr['si0']+gr['sj0'], gr['si1']+gr['sj1'], gr['score'], fname, MAP=True))
        return tuple(res)

    def res_columns(self, mres, pfx=' '):
        """ Produce README-format markdown table row piece summarizing
        important statistics """
        return('%s%.6f    |%s%.6f |%s%.6f |%s%.6f'
               % (pfx, mres[self.trainf]['MRR'],
                  pfx, mres[self.valf]['MRR'],
                  pfx, mres[self.testf].get('MAP', np.nan),
                  pfx, mres[self.testf].get('MRR', np.nan)))


def task():
    return AnsSelTask()
