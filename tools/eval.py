#!/usr/bin/python3
# vim: set fileencoding=utf8:
"""
Evaluate a set of pre-trained KeraSTS model instances (training runs)
to get publication-grade results.

Usage: tools/eval.py MODEL TASK TRAINDATA VALDATA TESTDATA WEIGHTFILES... [vocabf='VOCABF'] [PARAM=VALUE]...

Example:
    tools/eval.py cnn para \
            data/para/msr/msr-para-train.tsv data/para/msr/msr-para-val.tsv data/para/msr/msr-para-test.tsv \
            weights-para-avg--2e2c031f78c95c8c-00-bestval.h5 weights-para-avg--2e2c031f78c95c8c-01-bestval.h5 \
            inp_e_dropout=1/2

This works on whatever tools/train.py produced (weight files), loading
them and evaluating them on a given task + model + dataset.  Model parameters
must be the same as you used when training the model.

DO NOT RUN THIS on test data in the course of regular development and parameter
tuning; it is meant only to produce the final evaluation results for publication.
Before final evaluations, pass '-' instead of TESTDATA file and test split will
be ignored during evaluation.

VOCABDATA is typically the training set.  It is a separate argument
to allow for training on one dataset and running on another one, as
the vocabulary must always be the same for a given model instance
(so it'd be of the original dataset even if you evaluate on a new one).
Sometimes, you may want to use a different task to initialize the vocabulary
(e.g. for ubuntu-based transfer learning) - use "vocabt='ubuntu'" for that.

Pass as many weight files as you have available.  Means, 95% confidence
intervals and correlations will be calculated and reported.  Parameters
are distinguished from weight files by detecting the '=' sign.
"""

from __future__ import print_function
from __future__ import division

import importlib
import numpy as np
import scipy.stats as ss
import sys

import pysts.embedding as emb

from train import config
import models  # importlib python3 compatibility requirement
import tasks

# Unused imports for evaluating commandline params
from keras.layers.recurrent import SimpleRNN, GRU, LSTM
from pysts.kerasts.objectives import ranknet, ranksvm, cicerons_1504
import pysts.kerasts.blocks as B


def stat(niter, fname, qty, r, alpha=0.95, bonferroni=1.):
    if len(r) > 0:
        bar = ss.t.isf((1 - alpha) / bonferroni / 2, len(r) - 1) * np.std(r) / np.sqrt(len(r))
    else:
        bar = np.nan
    print('%s: Final %d×%s %f ±%f (%s)' % (fname, niter, qty, np.mean(r), bar, r))
    return bar


if __name__ == "__main__":
    modelname, taskname, trainf, valf, testf = sys.argv[1:6]
    g = ([], [])
    g_i = 0
    for p in sys.argv[6:]:
        if '=' in p:  # config param
            g_i = 1
        g[g_i].append(p)
    weightfs, params = g
    if testf == '-':
        testf = None

    model_module = importlib.import_module('.'+modelname, 'models')
    task_module = importlib.import_module('.'+taskname, 'tasks')
    task = task_module.task()
    conf, ps, h = config(model_module.config, task.config, params)
    task.set_conf(conf)

    # TODO we should be able to get away with actually *not* loading
    # this at all!
    if conf['embdim'] is not None:
        print('GloVe')
        task.emb = emb.GloVe(N=conf['embdim'])
    else:
        task.emb = None

    print('Dataset')
    if 'vocabf' in conf:
        if 'vocabt' in conf:
            taskv_module = importlib.import_module('.'+conf['vocabt'], 'tasks')
            taskv = taskv_module.task()
            taskv.load_vocab(conf['vocabf'])
            task.vocab = taskv.vocab
        else:
            task.load_vocab(conf['vocabf'])
    task.load_data(trainf, valf, testf)

    # Collect eval results
    res = {trainf: [], valf: [], testf: []}
    for weightf in weightfs:
        print('Model')
        model = task.build_model(model_module.prep_model)

        print(weightf)
        model.load_weights(weightf)

        print('Predict&Eval (best val epoch)')
        resT, resv, rest = task.eval(model)
        if resT is not None:
            res[trainf].append(resT)
        res[valf].append(resv)
        if rest is not None:
            res[testf].append(rest)
        print()

    # Report statistics and compute 95% bounds
    fres = {trainf: {}, valf: {}, testf: {}}  # res ransposed from split-run-field to split-field-run
    mres = {trainf: {}, valf: {}, testf: {}}  # mean
    bres = {trainf: {}, valf: {}, testf: {}}  # 95% bound
    for split in [trainf, valf, testf]:
        if split is None or not res[split]:
            continue
        for field in res[valf][0]._fields:
            values = [getattr(r, field) for r in res[split]]
            fres[split][field] = values
            mres[split][field] = np.mean(values)
            bres[split][field] = stat(len(weightfs), split, field, values)

    for field in res[valf][0]._fields:
        if fres[trainf]:
            print('train-val %s Pearsonr: %f' % (field, ss.pearsonr(fres[trainf][field], fres[valf][field])[0],))
        if fres[testf]:
            print('val-test %s Pearsonr: %f' % (field, ss.pearsonr(fres[valf][field], fres[testf][field])[0],))

    # data/ README table format
    print('| % -24s |%s | %s'
          % (modelname, task.res_columns(mres),
             '(defaults)' if not params
             else ' '.join(['``%s``' % (p,) for p in params if not p.startswith('vocabf=')])))
    print('| % -24s |%s |' % ('', task.res_columns(bres, '±')))
