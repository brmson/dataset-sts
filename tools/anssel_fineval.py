#!/usr/bin/python3
# vim: set fileencoding=utf8:
"""
Evaluate a KeraSTS model on the Answer Sentence Selection task
for publication.

Usage: tools/anssel_fineval.py N MODEL TRAINDATA VALDATA TESTDATA [PARAM=VALUE]...

Example:
    tools/anssel_fineval.py 16 avg data/anssel/wang/train-all.csv data/anssel/wang/dev.csv data/anssel/wang/test.csv l2reg=1e-4

This runs anssel_train N times on the given model, evaluating it on both
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

import anssel_train
import anssel_treceval
import models  # importlib python3 compatibility requirement


def ev_map(s0, s1, y, ypred, fname):
    mapt = ev.trec_map(s0, s1, y, ypred)
    print('%s MAP: %f' % (fname, mapt))
    return mapt


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
    conf, ps, h = anssel_train.config(module.config, params)

    if conf['embdim'] is not None:
        print('GloVe')
        glove = emb.GloVe(N=conf['embdim'])
    else:
        glove = None

    print('Dataset')
    s0, s1, y, vocab, gr = anssel_train.load_set(trainf)
    s0v, s1v, yv, _, grv = anssel_train.load_set(valf, vocab)
    s0t, s1t, yt, _, grt = anssel_train.load_set(testf, vocab)

    mrr = []
    mrrv = []
    mrrt = []
    mapt = []
    for i in range(niter):
        runid = '%s-%x-%02d' % (modelname, h, i)
        print('RunID: %s  (%s)' % (runid, ps))

        model = anssel_train.train_and_eval(runid, module.prep_model, conf, glove, vocab, gr, s0, grv, s0v, do_eval=False)

        print('Predict&Eval (best val epoch)')
        ypred = model.predict(gr)['score'][:,0]
        ypredv = model.predict(grv)['score'][:,0]
        ypredt = model.predict(grt)['score'][:,0]

        mrr.append(ev.eval_anssel(ypred, s0, s1, y, trainf).MRR)
        mrrv.append(ev.eval_anssel(ypredv, s0v, s1v, yv, valf).MRR)
        rt = ev.eval_anssel(ypredt, s0t, s1t, yt, testf, MAP=True)
        mrrt.append(rt.MRR)
        mapt.append(rt.MAP)

        rdata = {'ps': ps, 'ypred': (ypred, ypredv, ypredt), 'mrr': (mrr, mrrv, mrrt), 'map': (None, None, mapt)}
        pickle.dump(rdata, open('%s-res.pickle' % (runid,), 'wb'), protocol=2)

    brr = stat(niter, trainf, 'MRR', mrr)
    brrv = stat(niter, valf, 'MRR', mrrv)
    bapt = stat(niter, testf, 'MAP', mapt)
    brrt = stat(niter, testf, 'MRR', mrrt)

    # README table format:
    print(                  '| % -24s | %.6f    | %.6f | %.6f | %.6f | %s' % (modelname, np.mean(mrr), np.mean(mrrv), np.mean(mapt), np.mean(mrrt),
                                                                              '(defaults)' if not params else ' '.join(['``%s``' % (p,) for p in params])))
    print('|                          |±%.6f    |±%.6f |±%.6f |±%.6f | ' % (brr, brrv, bapt, brrt))

    print('train-val MRR Pearsonr: %f' % (ss.pearsonr(mrr, mrrv)[0],))
    print('val-test MRR Pearsonr: %f' % (ss.pearsonr(mrrv, mrrt)[0],))
