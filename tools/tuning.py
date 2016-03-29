#!/usr/bin/python3
"""
Tune hyperparameters of a KeraSTS model on the given task, that is
train + evaluate the model with many different randomly samples config
settings.

Usage: tools/tuning.py MODEL TASK TRAINDATA VALDATA PARAM=VALUESET...

Example:
    tools/tuning.py cnn anssel data/anssel/wang/train-all.csv data/anssel/wang/dev.csv \
        "dropout=[1/2, 2/3, 3/4]" "inp_e_dropout=[1/2, 3/4, 4/5]" "l2reg=[1e-4, 1e-3, 1e-2]" \
        "project=[True, True, False]" "cnnact=['tanh', 'relu']" \
        "cdim={1: [0,0,1/2,1,2], 2: [0,0,1/2,1,2,0], 3: [0,0,1/2,1,2,0], 4: [0,0,1/2,1,2,0], 5: [0,0,1/2,1,2]},"

That is, the VALUESET is array of possible values for the given parameter;
in case the parameter takes a dict, it is a dict of key-valuesets.

TODO use spearmint or something for non-random sampling and estimation
of influence of different parameters on performance
"""

from __future__ import print_function
from __future__ import division

import importlib
import numpy as np
import sys
import time

import pysts.embedding as emb
from pysts.hyperparam import RandomSearch, hash_params

import models  # importlib python3 compatibility requirement
import tasks
from train import config, train_and_eval

# Unused imports for evaluating commandline params
from keras.layers.recurrent import SimpleRNN, GRU, LSTM
from pysts.kerasts.objectives import ranknet, ranksvm, cicerons_1504
import pysts.kerasts.blocks as B


if __name__ == "__main__":
    modelname, taskname, trainf, valf = sys.argv[1:5]
    params = sys.argv[5:]

    model_module = importlib.import_module('.'+modelname, 'models')
    task_module = importlib.import_module('.'+taskname, 'tasks')
    task = task_module.task()
    # Preliminary config:
    # (N.B. some conf values will be the sets, which is not something
    # we can use directly, but we just assume whatever we use below
    # directly wasn't specified as a tunable.)
    conf, ps, h = config(model_module.config, task.config, params)
    task.set_conf(conf)

    # TODO configurable embedding class
    if conf['embdim'] is not None:
        print('GloVe')
        task.emb = emb.GloVe(N=conf['embdim'])

    print('Dataset')
    if 'vocabf' in conf:
        task.load_vocab(conf['vocabf'])
    task.load_data(trainf, valf)

    tuneargs = dict()
    for p in params:
        k, v = p.split('=')
        v = eval(v)
        if isinstance(v, list) or isinstance(v, dict):
            tuneargs[k] = v

    rs = RandomSearch(modelname+'_'+taskname+'_log.txt', **tuneargs)

    for ps, h, pardict in rs():
        # final config for this run
        conf, ps, h = config(model_module.config, task.config, [])
        for k, v in pardict.items():
            conf[k] = v
        ps, h = hash_params(conf)
        task.set_conf(conf)

        runid = '%s-%s-%x' % (taskname, modelname, h)
        print()
        print(' ...... %s .................... %s' % (runid, ps))

        try:
            model, res = train_and_eval(runid, model_module.prep_model, task, conf)
            rs.report(ps, h, res[1])
        except Exception as e:
            print(e)
            time.sleep(1)
