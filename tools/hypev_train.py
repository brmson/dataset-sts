#!/usr/bin/python3
"""
Script used for training on hypev datasets (principally, argus).

Usage: tools/hypev_train.py MODEL TRAINDATA VALDATA [PARAM=VALUE]...

Example: tools/hypev_train.py rnn data/hypev/argus/argus_train.csv data/hypev/argus/argus_test.csv dropout=0

Prerequisites:
    * Get glove.6B.50d.txt from http://nlp.stanford.edu/projects/glove/
"""

from __future__ import print_function
from __future__ import division

import importlib
import sys
import csv
import pickle

from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.layers.core import Activation
from keras.models import Graph
import numpy as np

import pysts.embedding as emb
import pysts.eval as ev
import pysts.loader as loader
import pysts.nlp as nlp
from pysts.hyperparam import hash_params
from pysts.vocab import Vocabulary

from pysts.kerasts import graph_input_anssel
from pysts.kerasts.callbacks import HypEvCB
import pysts.kerasts.blocks as B
import models  # importlib python3 compatibility requirement


s0pad = 60
s1pad = 60


def load_set(fname, vocab=None):
    s0, s1, y = loader.load_hypev(fname)
    # s0=questions, s1=answers

    if vocab is None:
        vocab = Vocabulary(s0 + s1)

    si0 = vocab.vectorize(s0, spad=s0pad)
    si1 = vocab.vectorize(s1, spad=s1pad)
    f0, f1 = nlp.sentence_flags(s0, s1, s0pad, s1pad)
    gr = graph_input_anssel(si0, si1, y, f0, f1, s0, s1)

    return s0, s1, y, vocab, gr


def config(module_config, params):
    c = dict()
    c['embdim'] = 50
    c['inp_e_dropout'] = 1/2
    c['inp_w_dropout'] = 0
    c['e_add_flags'] = True

    c['ptscorer'] = B.mlp_ptscorer
    c['mlpsum'] = 'sum'
    c['Ddim'] = 1

    c['loss'] = 'binary_crossentropy'
    c['balance_class'] = False
    c['batch_size'] = 160
    c['nb_epoch'] = 2
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


def build_model(glove, vocab, module_prep_model, c, s0pad=s0pad, s1pad=s1pad):
    if c['ptscorer'] is None:
        # non-neural model
        return module_prep_model(vocab, c)

    if c['loss'] == 'binary_crossentropy':
        oact = 'sigmoid'
    else:
        # ranking losses require wide output domain
        oact = 'linear'

    model = prep_model(glove, vocab, module_prep_model, c, oact, s0pad, s1pad)
    model.compile(loss={'score': c['loss']}, optimizer='adam')
    return model


def dump_questions(sq, sa, labels, results, text):
    question = ''
    label = 1
    avg = 0
    q_num = 0
    correct = 0
    n = 0
    f = open('printout_'+text+'.csv', 'wb')
    w = csv.writer(f, delimiter=',')
    for q, y, t, a in zip(sq, labels, results, sa):
        if q == question:
            n += 1
            avg = n/(n+1)*avg+t/(n+1)
            row = [q, y, t, '', a]
            w.writerow(row)
        else:
            row = [q, y, t, avg, a]
            w.writerow(row)
            if q_num != 0 and abs(label-avg) < 0.5:
                correct += 1
            question = q
            label = y
            avg = t
            q_num += 1
            n = 0
    if q_num != 0 and abs(label-avg) < 0.5:
        correct += 1

    # print('precision on separate questions ('+text+'):', correct/q_num)


def train_and_eval(runid, module_prep_model, c, glove, vocab, gr, s0, grt, s0t, do_eval=True):
    print('Model')
    model = build_model(glove, vocab, module_prep_model, c)

    print('Training')
    if c.get('balance_class', False):
        one_ratio = np.sum(gr['score'] == 1) / len(gr['score'])
        class_weight = {'score': {0: one_ratio, 1: 0.5}}
    else:
        class_weight = {}
    # XXX: samples_per_epoch is in brmson/keras fork, TODO fit_generator()?
    model.fit(gr, validation_data=grt,
              callbacks=[HypEvCB(s0t, grt),
                         ModelCheckpoint('hyp-weights-'+runid+'-bestval.h5', save_best_only=True, monitor='acc', mode='max'),
                         EarlyStopping(monitor='acc', mode='max', patience=4)],
              class_weight=class_weight,
              batch_size=c['batch_size'], nb_epoch=c['nb_epoch'])
    model.save_weights('hyp-weights-'+runid+'-final.h5', overwrite=True)
    if c['ptscorer'] is None:
        model.save_weights('hyp-weights-'+runid+'-bestval.h5', overwrite=True)
    model.load_weights('hyp-weights-'+runid+'-bestval.h5')

    if do_eval:
        print('Predict&Eval (best epoch)')
        prediction = model.predict(gr)['score'][:,0]
        prediction_t = model.predict(grt)['score'][:,0]
        ev.eval_hypev(prediction, s0, gr['score'], 'Train')
        ev.eval_hypev(prediction_t, s0t, grt['score'], 'Val')
        dump_questions(s0, s1, gr['score'], prediction, 'Train')
        dump_questions(s0t, s1t, grt['score'], prediction_t, 'Val')
    return model


if __name__ == "__main__":
    modelname, trainf, valf = sys.argv[1:4]
    # modelname, trainf, valf = 'avg', 'data/hypev/argus/argus_train.csv', 'data/hypev/argus/argus_test.csv'
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
    pickle.dump(vocab, open('vocab.txt', 'wb'))

    train_and_eval(runid, module.prep_model, conf, glove, vocab, gr, s0, grt, s0t)
