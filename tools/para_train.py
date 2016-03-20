#!/usr/bin/python3
"""
Train a KeraSTS model on the Paraphrasing task.  Basically, it's a lot like
STS but with boolean output.

Usage: tools/para_train.py MODEL TRAINDATA VALDATA [PARAM=VALUE]...

Example: tools/para_train.py cnn data/para/msr/msr-para-train.tsv data/para/msr/msr-para-val.tsv inp_e_dropout=1/2

This applies the given text similarity model to the paraphrasing task.

Prerequisites:
    * Get glove.6B.300d.txt from http://nlp.stanford.edu/projects/glove/
"""

from __future__ import print_function
from __future__ import division

import importlib
import sys

from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.layers.core import Activation, Dropout
from keras.layers.recurrent import SimpleRNN, GRU, LSTM
from keras.models import Graph
import numpy as np

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


spad = 60


def load_set(fname, vocab=None, spad=spad):
    s0, s1, y = loader.load_msrpara(fname)

    if vocab is None:
        vocab = Vocabulary(s0 + s1)

    si0 = vocab.vectorize(s0, spad=spad)
    si1 = vocab.vectorize(s1, spad=spad)
    f0, f1 = nlp.sentence_flags(s0, s1, spad, spad)
    gr = graph_input_anssel(si0, si1, y, f0, f1, s0, s1)

    return (s0, s1, y, vocab, gr)


def config(module_config, params):
    c = dict()
    c['embdim'] = 300
    c['inp_e_dropout'] = 1/2
    c['inp_w_dropout'] = 0
    c['e_add_flags'] = True

    c['ptscorer'] = B.mlp_ptscorer
    c['mlpsum'] = 'sum'
    c['Ddim'] = 2

    c['loss'] = 'binary_crossentropy'
    c['balance_class'] = False
    c['batch_size'] = 160
    c['nb_epoch'] = 32
    module_config(c)

    for p in params:
        k, v = p.split('=')
        c[k] = eval(v)

    ps, h = hash_params(c)
    return c, ps, h


def prep_model(glove, vocab, module_prep_model, c, spad=spad):
    # Input embedding and encoding
    model = Graph()
    N = B.embedding(model, glove, vocab, spad, spad, c['inp_e_dropout'], c['inp_w_dropout'], add_flags=c['e_add_flags'])

    # Sentence-aggregate embeddings
    final_outputs = module_prep_model(model, N, spad, spad, c)

    # Measurement
    kwargs = dict()
    if c['ptscorer'] == B.mlp_ptscorer:
        kwargs['sum_mode'] = c['mlpsum']
    model.add_node(name='scoreS', input=c['ptscorer'](model, final_outputs, c['Ddim'], N, c['l2reg'], **kwargs),
                   layer=Activation('sigmoid'))
    model.add_output(name='score', input='scoreS')
    return model


def build_model(glove, vocab, module_prep_model, c, spad=spad, optimizer='adam', fix_layers=[]):
    if c['ptscorer'] is None:
        # non-neural model
        return module_prep_model(vocab, c, output='binary', spad=spad)

    model = prep_model(glove, vocab, module_prep_model, c, spad=spad)

    for lname in fix_layers:
        model.nodes[lname].trainable = False

    model.compile(loss={'score': c['loss']}, optimizer=optimizer)
    return model


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
    model.fit(gr, validation_data=grt,  # show_accuracy=True,
              callbacks=[ModelCheckpoint('para-weights-'+runid+'-bestval.h5', save_best_only=True),
                         EarlyStopping(patience=3)],
              class_weight=class_weight,
              batch_size=c['batch_size'], nb_epoch=c['nb_epoch'])
    model.save_weights('para-weights-'+runid+'-final.h5', overwrite=True)
    if c['ptscorer'] is None:
        model.save_weights('para-weights-'+runid+'-bestval.h5', overwrite=True)
    model.load_weights('para-weights-'+runid+'-bestval.h5')

    if do_eval:
        print('Predict&Eval (best val epoch)')
        ev.eval_para(model.predict(gr)['score'][:,0], gr['score'], 'Train')
        ev.eval_para(model.predict(grt)['score'][:,0], grt['score'], 'Val')
    return model


if __name__ == "__main__":
    modelname, trainf, valf = sys.argv[1:4]
    params = sys.argv[4:]

    module = importlib.import_module('.'+modelname, 'models')
    conf, ps, h = config(module.config, params)

    runid = '%s-%x' % (modelname, h)
    print('RunID: %s  (%s)' % (runid, ps))

    if conf['embdim'] is not None:
        print('GloVe')
        glove = emb.GloVe(N=conf['embdim'])
    else:
        glove = None

    print('Dataset')
    s0, s1, y, vocab, gr = load_set(trainf)
    s0t, s1t, yt, _, grt = load_set(valf, vocab)

    train_and_eval(runid, module.prep_model, conf, glove, vocab, gr, grt)
