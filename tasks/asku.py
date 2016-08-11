"""
KeraSTS interface for the AskUbuntu dataset of the Paraphrasing
task.

FIXME below obsolete

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

from keras.callbacks import EarlyStopping, ModelCheckpoint
import numpy as np
import os.path
import random

import pysts.eval as ev
import pysts.loader as loader
from pysts.kerasts import graph_input_anssel, graph_input_slice
from pysts.kerasts.callbacks import AnsSelCB
from pysts.kerasts.objectives import ranknet
import pysts.nlp as nlp
from pysts.vocab import Vocabulary

from .para import ParaphrasingTask


class AskUTask(ParaphrasingTask):
    def __init__(self):
        self.name = 'asku'
        self.s0pad = 60
        self.s1pad = 60
        self.emb = None
        self.vocab = None

    def config(self, c):
        c['loss'] = ranknet
        c['nb_epoch'] = 16
        c['batch_size'] = 192
        c['epoch_fract'] = 1/4

    def load_vocab(self, vocabf):
        self.texts = loader.load_askubuntu_texts(vocabf)
        self.vocab = Vocabulary(self.texts.values(), prune_N=self.c['embprune'], icase=self.c['embicase'])
        return self.vocab

    def load_set(self, fname, cache_dir=None):
        links = loader.load_askubuntu_q(fname)
        return links

    def link_to_s(self, link):
        # convert link in the askubuntu_q format to a set of sentence pairs
        pid, qids, qlabels = link
        s0 = []
        s1 = []
        labels = []
        for qid, ql in zip(qids, qlabels):
            s0.append(self.texts[pid])
            s1.append(self.texts[qid])
            labels.append(ql)
        return s0, s1, labels

    def links_to_graph(self, links):
        s0 = []
        s1 = []
        labels = []
        for link in links:
            s0l, s1l, labelsl = self.link_to_s(link)
            s0 += s0l
            s1 += s1l
            labels += labelsl

        si0, sj0 = self.vocab.vectorize(s0, self.emb, spad=self.s0pad)
        si1, sj1 = self.vocab.vectorize(s1, self.emb, spad=self.s1pad)
        f0, f1 = nlp.sentence_flags(s0, s1, self.s0pad, self.s1pad)

        gr = graph_input_anssel(si0, si1, sj0, sj1, None, None, np.array(labels), f0, f1, s0, s1)
        return gr

    def load_data(self, trainf, valf, testf=None):
        self.trainf = trainf
        self.valf = valf
        self.testf = testf

        if self.vocab is None:
            # XXX: this vocab includes even val,test words!
            self.load_vocab(os.path.dirname(trainf) + '/text_tokenized.txt.gz')

        self.links = self.load_set(trainf)
        from itertools import chain
        self.gr = {'score': list(chain.from_iterable([l[2] for l in self.links]))}
        print('Training set: %d links, %d sentence pairs' % (len(self.links), len(self.gr['score'])))
        self.linksv = self.load_set(valf)
        self.grv = self.links_to_graph(self.linksv)
        if testf is not None:
            self.linkst = self.load_set(testf)
            self.grt = self.links_to_graph(self.linksv)
        else:
            self.linkst = None

    def sample_pairs(self, batch_size, once=False):
        """ A generator that produces random pairs from the dataset """
        ids = range(len(self.links))
        while True:
            random.shuffle(ids)
            links_to_yield = []
            n_yielded = 0
            for i in ids:
                link = self.links[i]
                links_to_yield.append(link)
                n_yielded += len(link[1])

                if n_yielded < batch_size:
                    continue

                # we have accumulated enough pairs, produce a graph
                ogr = self.links_to_graph(links_to_yield)
                links_to_yield = []
                n_yielded = 0
                yield ogr
            if once:
                break

    def fit_callbacks(self, weightsf):
        return [AnsSelCB(self.grv['si0'], self.grv),
                ModelCheckpoint(weightsf, save_best_only=True),
                EarlyStopping(patience=3)]

    def fit_model(self, model, **kwargs):
        batch_size = kwargs.pop('batch_size')
        kwargs['callbacks'] = self.fit_callbacks(kwargs.pop('weightsf'))
        return model.fit_generator(self.sample_pairs(batch_size), **kwargs)

    def eval(self, model):
        res = [None]
        for gr, fname in [(self.grv, self.valf), (self.grt, self.testf)]:
            if gr is None:
                res.append(None)
                continue
            ypred = model.predict(gr)['score'][:,0]
            res.append(ev.eval_ubuntu(ypred, gr['si0'], gr['score'], fname))
        return tuple(res)

    def res_columns(self, mres, pfx=' '):
        """ Produce README-format markdown table row piece summarizing
        important statistics """
        return('%s%.6f |%s%.6f |%s%.6f  |%s%.6f |%s%.6f  |%s%.6f  '
               % (pfx, mres[self.valf]['MRR'],
                  pfx, mres[self.valf]['R10_1'],
                  pfx, mres[self.valf]['R10_5'],
                  pfx, mres[self.testf].get('MRR', np.nan),
                  pfx, mres[self.testf].get('R10_1', np.nan),
                  pfx, mres[self.testf].get('R10_5', np.nan)))


def task():
    return AskUTask()
