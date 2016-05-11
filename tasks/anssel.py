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
from pysts.kerasts import graph_input_anssel, graph_nparray_anssel
from pysts.kerasts.callbacks import AnsSelCB
from pysts.kerasts.objectives import ranknet
import pysts.loader as loader
import pysts.nlp as nlp
from pysts.vocab import Vocabulary

from . import AbstractTask


def prescoring_model(model_module, c, weightsf):
    """ Setup and return a pre-scoring model """
    # We just make another instance of our task with the prescoring model
    # specific config, build the model and apply it
    prescore_task = task()
    prescore_task.set_conf(c)

    print('[Prescoring] Model')
    model = prescore_task.build_model(model_module.prep_model)

    print('[Prescoring] ' + weightsf)
    model.load_weights(weightsf)
    return model


def graph_input_prune(gr, ypred, N, skip_oneclass=False):
    """ Given a gr and a given scoring, keep only top N s1 for each s0,
    and stash the others away to _x-suffixed keys (for potential recovery). """
    def prune_filter(ypred, N):
        """ yield (index, passed) tuples """
        ys = sorted(enumerate(ypred), key=lambda yy: yy[1], reverse=True)
        i = 0
        for n, y in ys:
            yield n, (i < N)
            i += 1

    # Go through (s0, s1), keeping track of the beginning of the current
    # s0 block, and appending pruned versions
    i = 0
    grp = dict([(k, []) for k in gr.keys()] + [(k+'_x', []) for k in gr.keys()])
    for j in range(len(gr['si0']) + 1):
        if j < len(gr['si0']) and (j == 0 or np.all(gr['si0'][j]+gr['sj0'][j] == gr['si0'][j-1]+gr['sj0'][j-1])):
            # within same-s0 block, carry on
            continue
        # block boundary

        # possibly check if we have both classes picked (for training)
        if skip_oneclass:
            n_picked = 0
            for n, passed in prune_filter(ypred[i:j], N):
                if not passed:
                    break
                n_picked += gr['score'][i + n] > 0
            if n_picked == 0:
                # only false; tough luck, prune everything for this s0
                for k in gr.keys():
                    grp[k+'_x'] += list(gr[k][i:j])
                i = j
                continue

        # append pruned subset
        for n, passed in prune_filter(ypred[i:j], N):
            for k in gr.keys():
                if passed:
                    grp[k].append(gr[k][i + n])
                else:
                    grp[k+'_x'].append(gr[k][i + n])

        i = j

    return graph_nparray_anssel(grp)


def graph_input_unprune(gro, grp, ypred, xval):
    """ Reconstruct original graph gro from a pruned graph grp,
    with predictions set to always False for the filtered out samples.
    (xval denotes how the False is represented) """
    if 'score_x' not in grp:
        return grp, ypred  # not actually pruned

    gru = dict([(k, list(grp[k])) for k in gro.keys()])

    # XXX: this will generate non-continuous s0 blocks,
    # hopefully okay for all ev tools
    for k in gro.keys():
        gru[k] += grp[k+'_x']
    ypred = list(ypred)
    ypred += [xval for i in grp['score_x']]
    ypred = np.array(ypred)

    return graph_nparray_anssel(gru), ypred


class AnsSelTask(AbstractTask):
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

    def prescoring_apply(self, gr, skip_oneclass=False):
        """ Given a gr, prescore the pairs and do either pruning (for each s0,
        keep only top N s1 based on the prescoring) or add the prescore as
        an input. """
        if 'prescoring_model' not in self.c or (self.c['prescoring_prune'] is None and self.c['prescoring_input'] is None):
            return gr  # nothing to do

        if 'prescoring_model_inst' not in self.c:
            # cache the prescoring model instance
            self.c['prescoring_model_inst'] = prescoring_model(self.c['prescoring_model'], self.c['prescoring_c'], self.c['prescoring_weightsf'])
        print('[Prescoring] Predict')
        ypred = self.c['prescoring_model_inst'].predict(gr)['score'][:,0]

        if self.c['prescoring_input'] is not None:
            inp = self.c['prescoring_input']
            gr[inp] = np.reshape(ypred, (len(ypred), 1))  # 1D to 2D

        if self.c['prescoring_prune'] is not None:
            N = self.c['prescoring_prune']
            print('[Prescoring] Prune')
            gr = graph_input_prune(gr, ypred, N, skip_oneclass=skip_oneclass)

        return gr

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

        return super().fit_model(model, **kwargs)

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
