"""
KeraSTS interface for the Ubuntu dataset of the Answer Sentence Selection
(Next Utterance Ranking) task.

Training example:
    tools/train.py avg ubuntu data/anssel/ubuntu/v2-trainset.pickle data/anssel/ubuntu/v2-valset.pickle "vocabf='data/anssel/ubuntu/v2-vocab.pickle'"

If this crashes due to out-of-memory error, you'll need to lower the batch
size - pass e.g. batch_size=128.  To speed up training, you may want to
conversely bump the batch_size if you have a smaller model (e.g. cnn).

First, you must however run:
    tools/ubuntu_preprocess.py --revocab data/anssel/ubuntu/v2-trainset.csv data/anssel/ubuntu/v2-trainset.pickle data/anssel/ubuntu/v2-vocab.pickle
    tools/ubuntu_preprocess.py data/anssel/ubuntu/v2-valset.csv data/anssel/ubuntu/v2-valset.pickle data/anssel/ubuntu/v2-vocab.pickle
    tools/ubuntu_preprocess.py data/anssel/ubuntu/v2-testset.csv data/anssel/ubuntu/v2-testset.pickle data/anssel/ubuntu/v2-vocab.pickle
(N.B. this will include only the first 1M samples of the train set).

(TODO: Make these downloadable.)

Notes:
    * differs from https://github.com/npow/ubottu/blob/master/src/merge_data.py
      in that all unseen words outside of train set share a single
      common random vector rather than each having a different one
      (or deferring to stock GloVe vector)
    * reduced vocabulary only to words that appear at least twice,
      because the embedding matrix is too big for our GPUs otherwise
    * in case of too long sequences, the beginning is discarded rather
      than the end; this is different from KeraSTS default as well as
      probably the prior art

Ideas (future work):
    * rebuild the train set to train for a ranking task rather than
      just identification task?
"""

from __future__ import print_function
from __future__ import division

from keras.preprocessing.sequence import pad_sequences
import numpy as np
try:
    import cPickle
except ImportError:  # python3
    import pickle as cPickle
import pickle
import random
import traceback

import pysts.eval as ev
from pysts.kerasts import graph_input_anssel, graph_input_slice
import pysts.nlp as nlp

from . import AbstractTask
from .anssel import AnsSelTask


def pad_3d_sequence(seqs, maxlen, nd, dtype='int32'):
    pseqs = np.zeros((len(seqs), maxlen, nd)).astype(dtype)
    for i, seq in enumerate(seqs):
        trunc = np.array(seq[-maxlen:], dtype=dtype)  # trunacting='pre'
        pseqs[i, :trunc.shape[0]] = trunc  # padding='post'
    return pseqs


def pad_graph(gr, s0pad, s1pad):
    """ pad sequences in the graph """
    gr['si0'] = pad_sequences(gr['si0'], maxlen=s0pad, truncating='pre', padding='post')
    gr['si1'] = pad_sequences(gr['si1'], maxlen=s1pad, truncating='pre', padding='post')
    gr['sj0'] = pad_sequences(gr['sj0'], maxlen=s0pad, truncating='pre', padding='post')
    gr['sj1'] = pad_sequences(gr['sj1'], maxlen=s1pad, truncating='pre', padding='post')
    gr['f0'] = pad_3d_sequence(gr['f0'], maxlen=s0pad, nd=nlp.flagsdim)
    gr['f1'] = pad_3d_sequence(gr['f1'], maxlen=s1pad, nd=nlp.flagsdim)
    gr['score'] = np.array(gr['score'])


class UbuntuTask(AnsSelTask):
    def __init__(self):
        self.name = 'ubuntu'
        self.s0pad = 160
        self.s1pad = 160
        self.emb = None
        self.vocab = None

    def config(self, c):
        c['loss'] = 'binary_crossentropy'
        c['nb_epoch'] = 16
        c['batch_size'] = 192
        c['epoch_fract'] = 1/4

    def load_vocab(self, vocabf):
        # use plain pickle because unicode
        self.vocab = pickle.load(open(vocabf, "rb"))
        return self.vocab

    def load_set(self, fname, cache_dir=None):
        si0, si1, sj0, sj1, f0, f1, labels = cPickle.load(open(fname, "rb"))
        gr = graph_input_anssel(si0, si1, sj0, sj1, None, None, labels, f0, f1)
        return (gr, labels, self.vocab)

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

    def sample_pairs(self, gr, batch_size, shuffle=True, once=False):
        """ A generator that produces random pairs from the dataset """
        try:
            id_N = int((len(gr['si0']) + batch_size-1) / batch_size)
            ids = list(range(id_N))
            while True:
                if shuffle:
                    # XXX: We never swap samples between batches, does it matter?
                    random.shuffle(ids)
                for i in ids:
                    sl = slice(i * batch_size, (i+1) * batch_size)
                    ogr = graph_input_slice(gr, sl)
                    # TODO: Add support for discarding too long samples?
                    pad_graph(ogr, s0pad=self.s0pad, s1pad=self.s1pad)
                    ogr['se0'] = self.emb.map_jset(ogr['sj0'])
                    ogr['se1'] = self.emb.map_jset(ogr['sj1'])
                    # print(sl)
                    # print('<<0>>', ogr['sj0'], ogr['se0'])
                    # print('<<1>>', ogr['sj1'], ogr['se1'])
                    yield ogr
                if once:
                    break
        except Exception:
            traceback.print_exc()

    def fit_model(self, model, **kwargs):
        self.grv_p = self.grv  # no prescoring support (for now?)

        # Recompute epoch_fract based on the new train set size
        if self.c['epoch_fract'] != 1:
            kwargs['samples_per_epoch'] = int(len(self.gr['si0']) * self.c['epoch_fract'])

        return AbstractTask.fit_model(self, model, **kwargs)

    def eval(self, model):
        res = [None]
        for gr, fname in [(self.grv, self.valf), (self.grt, self.testf)]:
            if gr is None:
                res.append(None)
                continue
            ypred = self.predict(model, gr)
            res.append(ev.eval_ubuntu(ypred, np.array(gr['si0']) + np.array(gr['sj0']), gr['score'], fname))
        return tuple(res)

    def res_columns(self, mres, pfx=' '):
        """ Produce README-format markdown table row piece summarizing
        important statistics """
        return('%s%.6f |%s%.6f |%s%.6f  |%s%.6f |%s%.6f  |%s%.6f  '
               % (pfx, mres[self.valf]['MRR'],
                  pfx, mres[self.valf]['R2_1'],
                  pfx, mres[self.valf]['R10_2'],
                  pfx, mres[self.testf].get('MRR', np.nan),
                  pfx, mres[self.testf].get('R2_1', np.nan),
                  pfx, mres[self.testf].get('R10_2', np.nan)))


def task():
    return UbuntuTask()
