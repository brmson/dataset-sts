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
"""

from __future__ import print_function
from __future__ import division

import importlib
import sys

from keras.callbacks import ModelCheckpoint
from keras.layers.recurrent import SimpleRNN, GRU, LSTM
import numpy as np

import pysts.embedding as emb
import pysts.kerasts.blocks as B
from pysts.hyperparam import hash_params

from pysts.kerasts.objectives import ranknet, ranksvm, cicerons_1504
import models  # importlib python3 compatibility requirement
import tasks


def config(model_config, task_config, params):
    c = dict()
    c['embdim'] = 300
    c['inp_e_dropout'] = 1/2
    c['inp_w_dropout'] = 0
    c['e_add_flags'] = True

    c['ptscorer'] = B.mlp_ptscorer
    c['mlpsum'] = 'sum'
    c['Ddim'] = 2

    c['loss'] = 'mse'  # you really want to override this in each task's config()
    c['balance_class'] = False
    c['batch_size'] = 160
    c['nb_epoch'] = 16
    task_config(c)
    model_config(c)

    for p in params:
        k, v = p.split('=')
        c[k] = eval(v)

    ps, h = hash_params(c)
    return c, ps, h


def train_and_eval(runid, module_prep_model, task, c, do_eval=True):
    print('Model')
    model = task.build_model(module_prep_model, c)

    print('Training')
    if c['balance_class']:
        one_ratio = np.sum(task.gr['score'] == 1) / len(task.gr['score'])
        class_weight = {'score': {0: one_ratio, 1: 0.5}}
    else:
        class_weight = {}
    callbacks = task.fit_callbacks() + [ModelCheckpoint(task.name+'-weights-'+runid+'-bestval.h5', save_best_only=True)]
    # XXX: samples_per_epoch is in brmson/keras fork, TODO fit_generator()?
    model.fit(task.gr, validation_data=task.grv,  # show_accuracy=True,
              callbacks=callbacks, class_weight=class_weight,
              batch_size=c['batch_size'], nb_epoch=c['nb_epoch'])
    # model.save_weights(task.name+'-weights-'+runid+'-final.h5', overwrite=True)
    if c['ptscorer'] is None:
        model.save_weights(task.name+'-weights-'+runid+'-bestval.h5', overwrite=True)
    model.load_weights(task.name+'-weights-'+runid+'-bestval.h5')

    if do_eval:
        print('Predict&Eval (best val epoch)')
        task.eval(model)
    return model


if __name__ == "__main__":
    modelname, taskname, trainf, valf = sys.argv[1:5]
    params = sys.argv[5:]

    model_module = importlib.import_module('.'+modelname, 'models')
    task_module = importlib.import_module('.'+taskname, 'tasks')
    task = task_module.task()
    conf, ps, h = config(model_module.config, task.config, params)

    runid = '%s-%s-%x' % (taskname, modelname, h)
    print('RunID: %s  (%s)' % (runid, ps))

    # TODO configurable embedding class
    if conf['embdim'] is not None:
        print('GloVe')
        task.emb = emb.GloVe(N=conf['embdim'])

    print('Dataset')
    task.load_data(trainf, valf)

    train_and_eval(runid, model_module.prep_model, task, conf)
