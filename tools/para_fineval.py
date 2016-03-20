#!/usr/bin/python3
# vim: set fileencoding=utf8:
"""
Evaluate a KeraSTS model on the Paraphrasing task
for publication.

Usage: tools/para_train.py N MODEL TRAINDATA VALDATA TESTDATA [PARAM=VALUE]...

Example:
    tools/para_fineval.py 16 avg data/para/msr/msr-para-train.tsv data/para/msr/msr-para-val.tsv data/para/msr/msr-para-test.tsv

This runs para_train N times on the given model, evaluating it on both
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

from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.layers.recurrent import SimpleRNN, GRU, LSTM

import pysts.embedding as emb
import pysts.eval as ev
import pysts.kerasts.blocks as B

import para_train
import models  # importlib python3 compatibility requirement


def stat(niter, fname, qty, r, alpha=0.95, bonferroni=1.):
    if len(r) > 0:
        bar = ss.t.isf((1 - alpha) / bonferroni / 2, len(r) - 1) * np.std(r) / np.sqrt(len(r))
    else:
        bar = np.nan
    print('%s: Final %d×%s %f ±%f (%s)' % (fname, niter, qty, np.mean(r), bar, r))
    return bar


if __name__ == "__main__":
    niter, modelname, trainf, valf, testf = sys.argv[1:6]
    params = sys.argv[6:]

    niter = int(niter)
    module = importlib.import_module('.'+modelname, 'models')
    conf, ps, h = para_train.config(module.config, params)

    if conf['embdim'] is not None:
        print('GloVe')
        glove = emb.GloVe(N=conf['embdim'])
    else:
        glove = None

    print('Dataset')
    s0, s1, y, vocab, gr = para_train.load_set(trainf)
    s0v, s1v, yv, _, grv = para_train.load_set(valf, vocab)
    s0t, s1t, yt, _, grt = para_train.load_set(testf, vocab)

    acc, f1 = [], []
    accv, f1v = [], []
    acct, f1t = [], []
    for i in range(niter):
        runid = '%s-%x-%02d' % (modelname, h, i)
        print('RunID: %s  (%s)' % (runid, ps))

        model = para_train.train_and_eval(runid, module.prep_model, conf, glove, vocab, gr, grt, do_eval=False)

        print('Predict&Eval (best val epoch)')
        ypred = model.predict(gr)['score'][:,0]
        ypredv = model.predict(grv)['score'][:,0]
        ypredt = model.predict(grt)['score'][:,0]

        acc_, f1_ = ev.eval_para(ypred, y, trainf)
        acc.append(acc_)
        f1.append(f1_)
        acc_, f1_ = ev.eval_para(ypredv, yv, valf)
        accv.append(acc_)
        f1v.append(f1_)
        acc_, f1_ = ev.eval_para(ypredt, yt, testf)
        acct.append(acc_)
        f1t.append(f1_)

        rdata = {'ps': ps, 'ypred': (ypred, ypredv, ypredt), 'acc': (acc, accv, acct), 'f1': (f1, f1v, f1t)}
        pickle.dump(rdata, open('%s-res.pickle' % (runid,), 'wb'), protocol=2)

    bacc = stat(niter, trainf, 'Accuracy', acc)
    bf1 = stat(niter, trainf, 'F-score', f1)
    baccv = stat(niter, valf, 'Accuracy', accv)
    bf1v = stat(niter, valf, 'F-score', f1v)
    bacct = stat(niter, testf, 'Accuracy', acct)
    bf1t = stat(niter, testf, 'F-score', f1t)

    # README table format:
    print(                  '| % -24s | %.6f  | %.6f | %.6f | %.6f | %.6f | %.6f | %s' % (modelname,
          np.mean(acc), np.mean(f1), np.mean(accv), np.mean(f1v), np.mean(acct), np.mean(f1t),
          '(defaults)' if not params else ' '.join(['``%s``' % (p,) for p in params])))
    print('|                          |±%.6f  |±%.6f |±%.6f |±%.6f |±%.6f |±%.6f | ' % (bacc, bf1, baccv, bf1v, bacct, bf1t))

    print('train-val F1 Pearsonr: %f' % (ss.pearsonr(f1, f1v)[0],))
    print('val-test F1 Pearsonr: %f' % (ss.pearsonr(f1v, f1t)[0],))
