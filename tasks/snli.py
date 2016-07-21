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

from .rte import RTETask


class SnliTask(RTETask):
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
        c['batch_size'] = 196
        c['epoch_fract'] = 1/4

    def load_vocab(self, vocabf):
        # use plain pickle because unicode
        self.vocab = pickle.load(open(vocabf, "rb"))
        return self.vocab

    def load_set(self, fname):
        si0, si1, sj0, sj1, f0_, f1_, labels = pickle.load(open(fname, "rb"))
        gr = graph_input_anssel(si0, si1, sj0, sj1, None, None, labels, f0_, f1_)
        return (gr, labels, self.vocab)


def task():
    return SnliTask()
