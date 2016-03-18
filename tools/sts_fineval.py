#!/usr/bin/python3
# vim: set fileencoding=utf8:
"""
Evaluate a KeraSTS model on the Semantic Text Similarity task
for publication.

Usage: tools/sts_fineval.py N MODEL -- TRAINDATA... -- VALDATA... -- TESTDATA... [-- PARAM=VALUE]...

Example:
    tools/sts_fineval.py 16 avg data/sts/semeval-sts/all/201[-4].[^t]* -- data/sts/semeval-sts/all/2014.tweet-news.test.tsv -- data/sts/semeval-sts/all/2015.*

This runs sts_train N times on the given model, evaluating it on both
VALDATA and TESTDATA and producing mean metrics as well as 95% confidence
intervals and val/test correlations.

Do not run this in the course of regular development and parameter tuning;
it is meant only to produce the final evaluation results for publication.
"""

from __future__ import print_function
from __future__ import division

import importlib
import numpy as np
import pickle
import scipy.stats as ss
import sys

from keras.layers.convolutional import MaxPooling1D, AveragePooling1D
from keras.layers.recurrent import SimpleRNN, GRU, LSTM
import pysts.embedding as emb
import pysts.eval as ev
import pysts.kerasts.blocks as B
from pysts.kerasts.objectives import ranknet

import sts_train
import models  # importlib python3 compatibility requirement


def stat(niter, fname, qty, r, alpha=0.95, bonferroni=1.):
    if len(r) > 0:
        bar = ss.t.isf((1 - alpha) / bonferroni / 2, len(r) - 1) * np.std(r) / np.sqrt(len(r))
    else:
        bar = np.nan
    print('%s: Final %d×%s %f ±%f (%s)' % (fname, niter, qty, np.mean(r), bar, r))
    return bar


if __name__ == "__main__":
    niter, modelname = sys.argv[1:3]
    g = ([], [], [], [])
    g_i = 0
    for p in sys.argv[3:]:
        if p == '--':
            g_i += 1
            continue
        g[g_i].append(p)
    (trainf, valf, testf, params) = g

    niter = int(niter)
    module = importlib.import_module('.'+modelname, 'models')
    conf, ps, h = sts_train.config(module.config, params)

    if conf['embdim'] is not None:
        print('GloVe')
        glove = emb.GloVe(N=conf['embdim'])
    else:
        glove = None

    print('Dataset')
    s0, s1, y, vocab, gr = sts_train.load_set(trainf)
    s0v, s1v, yv, _, grv = sts_train.load_set(valf, vocab)
    ls0t, ls1t, lyt, lgrt = ([], [], [], [])
    for testf0 in testf:
        s0t, s1t, yt, _, grt = sts_train.load_set(testf0, vocab)
        ls0t.append(s0t)
        ls1t.append(s1t)
        lyt.append(yt)
        lgrt.append(grt)

    pr = []
    prv = []
    prt = []
    for i in range(niter):
        runid = '%s-%x-%02d' % (modelname, h, i)
        print('RunID: %s  (%s)' % (runid, ps))

        model = sts_train.train_and_eval(runid, module.prep_model, conf, glove, vocab, gr, grv, do_eval=False)

        print('Predict&Eval (best val epoch)')
        ypred = model.predict(gr)['classes']
        ypredv = model.predict(grv)['classes']
        ypredt = dict()
        for ti in range(len(testf)):
            ypredt[testf[ti]] = model.predict(lgrt[ti])['classes']

        pr.append(ev.eval_sts(ypred, gr['classes'], 'Train'))
        prv.append(ev.eval_sts(ypredv, grv['classes'], 'Val'))
        dprt = dict()
        for ti in range(len(testf)):
            dprt[testf[ti]] = ev.eval_sts(ypredt[testf[ti]], lgrt[ti]['classes'], testf[ti])
        prt.append(dprt)

        rdata = {'ps': ps, 'ypred': (ypred, ypredv, ypredt), 'pr': (pr[-1], prv[-1], prt[-1])}
        pickle.dump(rdata, open('%s-res.pickle' % (runid,), 'wb'), protocol=2)

    # prt view indexed by testf, not by iteration
    prtt = dict()
    for ti in range(len(testf)):
        prtt[testf[ti]] = [prt0[testf[ti]] for prt0 in prt]

    mprt = [np.mean(prtt[testf[ti]]) for ti in range(len(testf))]  # (n_testf,) mean across epochs
    prmt = [np.mean(list(prt0.values())) for prt0 in prt]  # (n_epochs,) mean across datasets

    brr = stat(niter, trainf, 'Pearson', pr)
    brrv = stat(niter, valf, 'Pearson', prv)
    brrt = dict()
    for ti in range(len(testf)):
        brrt[testf[ti]] = stat(niter, testf[ti], 'Pearson', prtt[testf[ti]])
    bmrrt = stat(niter, 'Mean Test', 'Pearson', mprt)

    # README table format:
    print(                  '| % -24s | %.6f | %.6f | %s | %.6f | %s' % (modelname, np.mean(pr), np.mean(prv),
                                                                         ' | '.join(['%.6f' % (m,) for m in mprt]), np.mean(prmt),
                                                                         '(defaults)' if not params else ' '.join(['``%s``' % (p,) for p in params])))
    print('|                          |±%.6f |±%.6f |%s |±%.6f | ' % (brr, brrv,
                                                                      ' |'.join(['±%.6f' % (np.mean(brrt[testf[ti]]),) for ti in range(len(testf))]),
                                                                      bmrrt))

    print('train-val Pearson Pearsonr: %f' % (ss.pearsonr(pr, prv)[0],))
    print('val-test Pearson Pearsonr: %f' % (ss.pearsonr(prv, prmt)[0],))
