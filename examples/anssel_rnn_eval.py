#!/usr/bin/python3
"""
Tool for evaluating the answer selection RNN model (from anssel_rnn)
using the original trec_eval tool.

Usage: anssel_rnn_eval.py INITPARAMS WEIGHTSFILE TRAINDATA TESTDATA TREC_QRELS_FILE TREC_TOP_FILE
Example:
    examples/anssel_rnn_eval.py "dropout=2/3, l2reg=1e-4" weights.h5 anssel-wang/train-all.csv anssel-wang/dev.csv /tmp/ground.txt /tmp/res.txt
    trec_eval.8.1/trec_eval /tmp/ground.txt /tmp/res.txt
"""

from __future__ import print_function
from __future__ import division

import sys

import anssel_rnn as E
import pysts.embedding as emb
import pysts.eval as ev
import pysts.kerasts.blocks as B
from pysts.kerasts.objectives import ranknet


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


def save_trec_top(f, s0, s1, y):
    n = -1
    m = 0
    last_is0 = ''
    for is0, is1, iy in zip(s0, s1, y):
        if hash(tuple(is0)) != last_is0:
            last_is0 = hash(tuple(is0))
            m = 0
            n += 1
        print('%d 0 %d 1 %f rnn' % (n, m, iy), file=f)
        m += 1


if __name__ == "__main__":
    initparams, weightsfile, traindata, testdata, trec_qrels_file, trec_top_file = sys.argv[1:]

    print('Datasets')
    s0, s1, y, vocab, gr = E.load_set(traindata)
    s0t, s1t, yt, _, grt = E.load_set(testdata, vocab)

    print('Glove')
    glove = emb.GloVe(300)  # XXX hardcoded N

    print('Model')
    kwargs = eval('dict(' + initparams + ')')
    # XXX: hardcoded loss function
    model = E.prep_model(glove, vocab, oact='linear', **kwargs)
    model.compile(loss={'score': ranknet}, optimizer='adam')

    print('Weights')
    model.load_weights(weightsfile)

    print('Prediction')
    ypred = model.predict(gr)['score'][:,0]
    ypredt = model.predict(grt)['score'][:,0]

    ev.eval_anssel(ypred, s0, y, 'Train')
    ev.eval_anssel(ypredt, s0t, yt, 'Test')

    with open(trec_qrels_file, 'wt') as f:
        save_trec_qrels(f, s0t, s1t, yt)
    with open(trec_top_file, 'wt') as f:
        save_trec_top(f, s0t, s1t, ypredt)
