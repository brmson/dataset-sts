#!/usr/bin/python3
"""
Tool for evaluating the answer selection models
using the original trec_eval tool.

Usage: tools/anssel_treceval.py MODEL WEIGHTSFILE TRAINDATA VALDATA TREC_QRELS_FILE TREC_TOP_FILE [PARAM=VALUE]...
Example:
    tools/anssel_treceval.py cnn weights-bestval.h5 anssel-wang/train-all.csv anssel-wang/dev.csv /tmp/ground.txt /tmp/res.txt dropout=2/3 l2reg=1e-4
    trec_eval.8.1/trec_eval /tmp/ground.txt /tmp/res.txt
"""

from __future__ import print_function
from __future__ import division

import importlib
import subprocess
import sys

from keras.layers.convolutional import MaxPooling1D, AveragePooling1D
from keras.layers.recurrent import SimpleRNN, GRU, LSTM
import pysts.embedding as emb
import pysts.eval as ev
import pysts.kerasts.blocks as B
from pysts.kerasts.objectives import ranknet

import anssel_train


def save_trec_qrels(f, s0, s1, y):
    n = -1
    m = 0
    last_is0 = ''
    for is0, is1, iy in zip(s0, s1, y):
        if hash(tuple(is0)) != last_is0:
            last_is0 = hash(tuple(is0))
            m = 0
            n += 1
        print('%d 0 %d %d' % (n, m, iy), file=f)
        m += 1


def save_trec_top(f, s0, s1, y, code):
    n = -1
    m = 0
    last_is0 = ''
    for is0, is1, iy in zip(s0, s1, y):
        if hash(tuple(is0)) != last_is0:
            last_is0 = hash(tuple(is0))
            m = 0
            n += 1
        print('%d 0 %d 1 %f %s' % (n, m, iy, code), file=f)
        m += 1


def trec_eval_get(trec_qrels_file, trec_top_file, qty):
    p = subprocess.Popen('../trec_eval.8.1/trec_eval %s %s | grep %s | sed "s/.*\t//"' % (trec_qrels_file, trec_top_file, qty), stdout=subprocess.PIPE, shell=True)
    return float(p.communicate()[0])


if __name__ == "__main__":
    modelname, weightsfile, trainf, valf, trec_qrels_file, trec_top_file = sys.argv[1:7]
    params = sys.argv[7:]

    module = importlib.import_module('.'+modelname, 'models')
    conf, ps, h = anssel_train.config(module.config, params)

    print('GloVe')
    glove = emb.GloVe(N=conf['embdim'])

    print('Dataset')
    s0, s1, y, vocab, gr = anssel_train.load_set(trainf)
    s0t, s1t, yt, _, grt = anssel_train.load_set(valf, vocab)

    print('Model')
    model = anssel_train.build_model(glove, vocab, module.prep_model, conf)

    print('Weights')
    model.load_weights(weightsfile)

    print('Prediction')
    ypred = model.predict(gr)['score'][:,0]
    ypredt = model.predict(grt)['score'][:,0]

    ev.eval_anssel(ypred, s0, y, trainf)
    ev.eval_anssel(ypredt, s0t, yt, valf)

    with open(trec_qrels_file, 'wt') as f:
        save_trec_qrels(f, s0t, s1t, yt)
    with open(trec_top_file, 'wt') as f:
        save_trec_top(f, s0t, s1t, ypredt, modelname)
    mapt = trec_eval_get(trec_qrels_file, trec_top_file, 'map')
    print('%s MAP: %f' % (valf, mapt))
