#!/usr/bin/python3
"""
Load a KeraSTS model on the Ubuntu Dialogue task and reuse it on an anssel task.

Usage: tools/ubuntu_anssel_transfer.py MODEL WEIGHTS VOCAB ATRAINDATA AVALDATA [PARAM=VALUE]...

Example: tools/ubuntu_anssel_transfer.py rnn ubu-weights-rnn--23fa2eff7cda310d-bestval.h5 data/anssel/ubuntu/v2-vocab.pickle data/anssel/yodaqa/curatedv2-training.csv data/anssel/yodaqa/curatedv2-val.csv pdim=1 ptscorer=B.dot_ptscorer
"""

from __future__ import print_function
from __future__ import division

import importlib
import numpy as np
try:
    import cPickle
except ImportError:  # python3
    import pickle as cPickle
import pickle
import random
import sys

from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.layers.core import Activation, Dropout
from keras.layers.recurrent import SimpleRNN, GRU, LSTM
from keras.models import Graph
from keras.preprocessing.sequence import pad_sequences

import pysts.embedding as emb
import pysts.eval as ev
import pysts.loader as loader
import pysts.nlp as nlp
from pysts.hyperparam import hash_params
from pysts.vocab import Vocabulary

from pysts.kerasts import graph_input_anssel, graph_input_slice
import pysts.kerasts.blocks as B
from pysts.kerasts.callbacks import AnsSelCB
from pysts.kerasts.objectives import ranknet, ranksvm, cicerons_1504

import anssel_train
import ubuntu_train
import models  # importlib python3 compatibility requirement


# XXX: Not the ubuntu_train default, obviously; but allows for fast training
# of big models.
s0pad = 80
s1pad = 80


def pad_3d_sequence(seqs, maxlen, nd, dtype='int32'):
    pseqs = np.zeros((len(seqs), maxlen, nd)).astype(dtype)
    for i, seq in enumerate(seqs):
        trunc = np.array(seq[-maxlen:], dtype=dtype)  # trunacting='pre'
        pseqs[i, :trunc.shape[0]] = trunc  # padding='post'
    return pseqs


def pad_graph(gr, s0pad=s0pad, s1pad=s1pad):
    """ pad sequences in the graph """
    gr['si0'] = pad_sequences(gr['si0'], maxlen=s0pad, truncating='pre', padding='post')
    gr['si1'] = pad_sequences(gr['si1'], maxlen=s1pad, truncating='pre', padding='post')
    gr['f0'] = pad_3d_sequence(gr['f0'], maxlen=s0pad, nd=nlp.flagsdim)
    gr['f1'] = pad_3d_sequence(gr['f1'], maxlen=s1pad, nd=nlp.flagsdim)
    gr['score'] = np.array(gr['score'])


def load_set(fname, vocab):
    si0, si1, f0, f1, labels = cPickle.load(open(fname, "rb"))
    gr = graph_input_anssel(si0, si1, labels, f0, f1)
    return gr


def config(module_config, params):
    c = dict()
    c['embdim'] = 300
    c['inp_e_dropout'] = 1/2
    c['inp_w_dropout'] = 0
    c['e_add_flags'] = True

    c['ptscorer'] = B.mlp_ptscorer
    c['mlpsum'] = 'sum'
    c['Ddim'] = 1

    c['loss'] = ranknet  # XXX: binary_crossentropy back?
    c['balance_class'] = True  # seems essential
    c['batch_size'] = 64
    c['nb_epoch'] = 16
    c['epoch_fract'] = 1/4  # XXX: or much smaller?

    c['fix_layers'] = []

    module_config(c)

    for p in params:
        k, v = p.split('=')
        c[k] = eval(v)

    ps, h = hash_params(c)
    return c, ps, h


def sample_pairs(gr, c, batch_size, once=False):
    """ A generator that produces random pairs from the dataset """
    # XXX: We drop the last few samples if (1e6 % batch_size != 0)
    # XXX: We never swap samples between batches, does it matter?
    ids = range(int(len(gr['si0']) / batch_size))
    while True:
        random.shuffle(ids)
        for i in ids:
            sl = slice(i * batch_size, (i+1) * batch_size)
            ogr = graph_input_slice(gr, sl)
            # TODO: Add support for discarding too long samples?
            pad_graph(ogr)
            yield ogr
        if once:
            break


def transfer_eval(runid, weightsf, module_prep_model, c, glove, vocab, gr, grv):
    print('Model')
    model = anssel_train.build_model(glove, vocab, module_prep_model, c, s0pad=s0pad, s1pad=s1pad)
    print('Model (weights)')
    model.load_weights(weightsf)

    for lname in c['fix_layers']:
        model.nodes[lname].trainable = False

    print('Training')
    if c.get('balance_class', False):
        one_ratio = np.sum(gr['score'] == 1) / len(gr['score'])
        class_weight = {'score': {0: one_ratio, 1: 0.5}}
    else:
        class_weight = {}
    model.fit(gr, validation_data=grv,
              callbacks=[AnsSelCB(s0v, grv),
                         ModelCheckpoint('weights-'+runid+'-bestval.h5', save_best_only=True, monitor='mrr', mode='max'),
                         EarlyStopping(monitor='mrr', mode='max', patience=4)],
              class_weight=class_weight,
              batch_size=conf['batch_size'], nb_epoch=conf['nb_epoch'], samples_per_epoch=int(len(gr['score'])*conf['epoch_fract']))
    model.save_weights('weights-'+runid+'-final.h5', overwrite=True)

    print('Predict&Eval (best epoch)')
    model.load_weights('weights-'+runid+'-bestval.h5')
    ev.eval_anssel(model.predict(grv)['score'][:,0], grv['si0'], grv['score'], 'anssel Val')


if __name__ == "__main__":
    modelname, weightsf, vocabf, trainf, valf = sys.argv[1:6]
    params = sys.argv[6:]

    module = importlib.import_module('.'+modelname, 'models')
    conf, ps, h = config(module.config, params)

    runid = '%s-%x' % (modelname, h)
    print('RunID: %s  (%s)' % (runid, ps))

    print('GloVe')
    glove = emb.GloVe(N=conf['embdim'])

    print('Dataset (vocab)')
    vocab = pickle.load(open(vocabf, "rb"))  # use plain pickle because unicode

    print('Dataset (anssel train)')
    s0, s1, y, _, gr_ = anssel_train.load_set(trainf, vocab, s0pad=s0pad, s1pad=s1pad)
    gr = loader.graph_adapt_ubuntu(gr_, vocab)
    print('Dataset (anssel val)')
    s0v, s1v, yv, _, grv_ = anssel_train.load_set(valf, vocab, s0pad=s0pad, s1pad=s1pad)
    grv = loader.graph_adapt_ubuntu(grv_, vocab)

    transfer_eval(runid, weightsf, module.prep_model, conf, glove, vocab, gr, grv)
