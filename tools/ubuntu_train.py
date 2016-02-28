#!/usr/bin/python3
"""
Train a KeraSTS model on the Ubuntu Dialogue task.

Usage: tools/ubuntu_train.py MODEL VOCAB TRAINDATA VALDATA [PARAM=VALUE]...

Example: tools/ubuntu_train.py cnn data/anssel/ubuntu/v2-vocab.pickle data/anssel/ubuntu/v2-trainset.pickle data/anssel/ubuntu/v2-valset.pickle

See coments in anssel_train.py.

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
    * move the fit_generator-related functions to KeraSTS
    * rebuild the train set to train for a ranking task rather than
      just identification task?
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
import models  # importlib python3 compatibility requirement


# XXX: We didn't verify this really covers all sentences.  It should cover
# a lot, though.
s0pad = 160
s1pad = 160


def pad_3d_sequence(seqs, maxlen, nd, dtype='int32'):
    pseqs = np.zeros((len(seqs), maxlen, nd)).astype(dtype)
    for i, seq in enumerate(seqs):
        trunc = np.array(seq[-maxlen:], dtype=dtype)  # trunacting='pre'
        pseqs[i, :trunc.shape[0]] = trunc  # padding='post'
    return pseqs


def pad_graph(gr):
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
    c['e_add_flags'] = True

    c['ptscorer'] = B.mlp_ptscorer
    c['mlpsum'] = 'sum'
    c['Ddim'] = 1

    c['batch_size'] = 192
    c['loss'] = 'binary_crossentropy'
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


def train_and_eval(runid, module_prep_model, c, glove, vocab, gr, grt):
    print('Model')
    model = anssel_train.build_model(glove, vocab, module_prep_model, c, s0pad=s0pad, s1pad=s1pad)

    print('Training')
    model.fit_generator(sample_pairs(gr, c, c['batch_size']),
                        callbacks=[#ValSampleCB(grt, c),  # loss function & ubuntu metrics
                                   AnsSelCB(grt['si0'], grt),  # MRR
                                   ModelCheckpoint('ubu-weights-'+runid+'-bestval.h5', save_best_only=True, monitor='mrr', mode='max'),
                                   EarlyStopping(monitor='mrr', mode='max', patience=6)],
                        nb_epoch=32, samples_per_epoch=200000)
    model.save_weights('ubu-weights-'+runid+'-final.h5', overwrite=True)

    print('Predict&Eval (best epoch)')
    model.load_weights('ubu-weights-'+runid+'-bestval.h5')
    ypredt = model.predict(grt)['score'][:,0]
    ev.eval_ubuntu(ypredt, grt['si0'], grt['score'], 'Val')


if __name__ == "__main__":
    modelname, vocabf, trainf, valf = sys.argv[1:5]
    params = sys.argv[5:]

    module = importlib.import_module('.'+modelname, 'models')
    conf, ps, h = config(module.config, params)

    runid = '%s-%x' % (modelname, h)
    print('RunID: %s  (%s)' % (runid, ps))

    print('GloVe')
    glove = emb.GloVe(N=conf['embdim'])

    print('Dataset (vocab)')
    vocab = pickle.load(open(vocabf, "rb"))  # use plain pickle because unicode
    print('Dataset (train)')
    gr = load_set(trainf, vocab)
    print('Dataset (val)')
    grt = load_set(valf, vocab)
    print('Padding (val)')
    pad_graph(grt)

    train_and_eval(runid, module.prep_model, conf, glove, vocab, gr, grt)
