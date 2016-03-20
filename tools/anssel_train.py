#!/usr/bin/python3
"""
Train a KeraSTS model on the Answer Sentence Selection task.

Usage: tools/anssel_train.py MODEL TRAINDATA VALDATA [PARAM=VALUE]...

Example: tools/anssel_train.py cnn data/anssel/wang/train-all.csv data/anssel/wang/dev.csv inp_e_dropout=1/2

This applies the given text similarity model to the anssel task.
Extra input pre-processing is done:
Rather than relying on the hack of using the word overlap counts as additional
features for final classification, individual tokens are annotated by overlap
features and that's passed to the model along with the embeddings.

Final comparison of summary embeddings is by default performed by
a multi-layered perceptron with elementwise products and sums as the input,
while the Ranknet loss function is used as an objective.  You may also try
e.g. dot-product (non-normalized cosine similarity) and binary crossentropy
or ranksvm as loss function, but it doesn't seem to make a lot of difference.

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


s0pad = 60
s1pad = 60


def load_set(fname, vocab=None, s0pad=s0pad, s1pad=s1pad):
    s0, s1, y, t = loader.load_anssel(fname)
    # TODO: Make use of the t-annotations

    if vocab is None:
        vocab = Vocabulary(s0 + s1)

    si0 = vocab.vectorize(s0, spad=s0pad)
    si1 = vocab.vectorize(s1, spad=s1pad)
    f0, f1 = nlp.sentence_flags(s0, s1, s0pad, s1pad)
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
    c['Ddim'] = 1

    c['loss'] = ranknet
    c['balance_class'] = False
    c['batch_size'] = 160
    c['nb_epoch'] = 16
    c['epoch_fract'] = 1/4
    module_config(c)

    for p in params:
        k, v = p.split('=')
        c[k] = eval(v)

    ps, h = hash_params(c)
    return c, ps, h


def prep_model(glove, vocab, module_prep_model, c, oact, s0pad, s1pad):
    # Input embedding and encoding
    model = Graph()
    N = B.embedding(model, glove, vocab, s0pad, s1pad, c['inp_e_dropout'], c['inp_w_dropout'], add_flags=c['e_add_flags'])

    # Sentence-aggregate embeddings
    final_outputs = module_prep_model(model, N, s0pad, s1pad, c)

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


def build_model(glove, vocab, module_prep_model, c, s0pad=s0pad, s1pad=s1pad, optimizer='adam', fix_layers=[], do_compile=True):
    if c['ptscorer'] is None:
        # non-neural model
        return module_prep_model(vocab, c)

    if c['loss'] == 'binary_crossentropy':
        oact = 'sigmoid'
    else:
        # ranking losses require wide output domain
        oact = 'linear'

    model = prep_model(glove, vocab, module_prep_model, c, oact, s0pad, s1pad)

    for lname in fix_layers:
        model.nodes[lname].trainable = False

    if do_compile:
        model.compile(loss={'score': c['loss']}, optimizer=optimizer)
    return model


def train_and_eval(runid, module_prep_model, c, glove, vocab, gr, s0, grt, s0t, s0pad=s0pad, s1pad=s1pad, do_eval=True):
    print('Model')
    model = build_model(glove, vocab, module_prep_model, c, s0pad=s0pad, s1pad=s1pad)

    print('Training')
    if c.get('balance_class', False):
        one_ratio = np.sum(gr['score'] == 1) / len(gr['score'])
        class_weight = {'score': {0: one_ratio, 1: 0.5}}
    else:
        class_weight = {}
    # XXX: samples_per_epoch is in brmson/keras fork, TODO fit_generator()?
    model.fit(gr, validation_data=grt,
              callbacks=[AnsSelCB(s0t, grt),
                         ModelCheckpoint('weights-'+runid+'-bestval.h5', save_best_only=True, monitor='mrr', mode='max'),
                         EarlyStopping(monitor='mrr', mode='max', patience=4)],
              class_weight=class_weight,
              batch_size=c['batch_size'], nb_epoch=c['nb_epoch'], samples_per_epoch=int(len(s0)*c['epoch_fract']))
    model.save_weights('weights-'+runid+'-final.h5', overwrite=True)
    if c['ptscorer'] is None:
        model.save_weights('weights-'+runid+'-bestval.h5', overwrite=True)
    model.load_weights('weights-'+runid+'-bestval.h5')

    if do_eval:
        print('Predict&Eval (best epoch)')
        ev.eval_anssel(model.predict(gr)['score'][:,0], s0, gr['score'], 'Train')
        ev.eval_anssel(model.predict(grt)['score'][:,0], s0t, grt['score'], 'Val')
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

    train_and_eval(runid, module.prep_model, conf, glove, vocab, gr, s0, grt, s0t)
