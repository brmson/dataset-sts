#!/usr/bin/python3
"""
Train a KeraSTS model on the given task, save the trained model
to a weights file and evaluate it on a validation set.

Usage: tools/train.py MODEL TASK TRAINDATA VALDATA [PARAM=VALUE]...

Example: tools/train.py cnn para data/para/msr/msr-para-train.tsv data/para/msr/msr-para-val.tsv inp_e_dropout=1/2

Parameters are mostly task-specific and model-specific, see the
respective config() routines.  The training process itslef is
influenced by:

    * batch_size=N denotes number of samples per batch

    * nb_epoch=N denotes maximum number of epochs (tasks will
      typically include a val-dependent early stopping mechanism too)

    * nb_runs=N denotes number of re-trainings to attempt (1 by default);
      final weights are stored for each re-training, this is useful to
      control for randomness-induced evaluation instability (see also
      tools/eval.py); it's very much like just running this script
      N times, except faster (no embedding and dataset reloading)

Some other noteworthy task,model-generic parameters (even if
not train-specific) are:

    * adapt_ubuntu=True to add __eot__ __eos__ tokens to the dataset sentences
      like they are in the Ubuntu Dialogue dataset (useful for transfer
      learning but also potentially for the models as markers)

    * f_add=[INPUT...] to add extra graph inputs to the final ptscorer classifier
      as additional features.  For example, some of the anssel datasets accept
      ``f_add=['kw', 'akw']`` to include prescoring keyword matching.

    * prescoring=MODEL, prescoring_conf={CONFDICT}, prescoring_weightsf=FILE
      to apply a pre-scoring step on the dataset using the given model
      with the given config, loaded from given file; the precise usage
      (whether as a feature, rank-based pruning, etc.) of prescoring is
      task-specific; Ex.:

        "prescoring='termfreq'" "prescoring_conf={freq_mode: 'tf'}" \
                "prescoring_weightsf='weights-anssel-termfreq--120a2d2e6dcd0c16-bestval.h5'"
"""

from __future__ import print_function
from __future__ import division

import importlib
import numpy as np
import sys

import pysts.embedding as emb
from pysts.hyperparam import hash_params

import models  # importlib python3 compatibility requirement
import tasks

# Unused imports for evaluating commandline params
from keras.layers.recurrent import SimpleRNN, GRU, LSTM
from keras.optimizers import *
from pysts.kerasts.objectives import ranknet, ranksvm, cicerons_1504
import pysts.kerasts.blocks as B
from tasks import default_config


def config(model_config, task_config, params):
    c = default_config(model_config, task_config)

    for p in params:
        k, v = p.split('=')
        c[k] = eval(v)

    ps, h = hash_params(c)

    return c, ps, h


def train_model(runid, model, task, c):
    print('Training')
    fit_kwargs = dict()
    if c['balance_class']:
        one_ratio = np.sum(task.gr['score'] == 1) / len(task.gr['score'])
        fit_kwargs['class_weight'] = {'score': {0: one_ratio, 1: 0.5}}
    if 'score' in task.gr:
        n_samples = len(task.gr['score'])
    else:
        n_samples = len(task.gr['classes'])
    fit_kwargs['samples_per_epoch'] = int(n_samples * c['epoch_fract'])
    task.fit_model(model, weightsf='weights-'+runid+'-bestval.h5',
                   batch_size=c['batch_size'], nb_epoch=c['nb_epoch'],
                   **fit_kwargs)
    # model.save_weights('weights-'+runid+'-final.h5', overwrite=True)
    if c['ptscorer'] is None:
        model.save_weights('weights-'+runid+'-bestval.h5', overwrite=True)
    model.load_weights('weights-'+runid+'-bestval.h5')


def train_and_eval(runid, module_prep_model, task, c, do_eval=True):
    print('Model')
    model = task.build_model(module_prep_model)

    train_model(runid, model, task, c)

    if do_eval:
        print('Predict&Eval (best val epoch)')
        res = task.eval(model)
    else:
        res = None
    return model, res


if __name__ == "__main__":
    modelname, taskname, trainf, valf = sys.argv[1:5]
    params = sys.argv[5:]

    model_module = importlib.import_module('.'+modelname, 'models')
    task_module = importlib.import_module('.'+taskname, 'tasks')
    task = task_module.task()
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
    for i_run in range(conf['nb_runs']):
        if conf['nb_runs'] == 1:
            runid = '%s-%s-%x' % (taskname, modelname, h)
        else:
            runid = '%s-%s-%x-%02d' % (taskname, modelname, h, i_run)
        print('RunID: %s  (%s)' % (runid, ps))

        train_and_eval(runid, model_module.prep_model, task, conf)
