#!/usr/bin/python3
"""
Load a KeraSTS model trained on TASK1 and reuse it on TASK2, fine-tuning
the model on that task first.

Usage: tools/transfer.py MODEL TASK1 VOCAB1 WEIGHTS TASK2 TRAIN2DATA VAL2DATA [PARAM=VALUE]...

Example: tools/transfer.py rnn ubuntu data/anssel/ubuntu/v2-vocab.pickle \
        ubu-weights-rnn--23fa2eff7cda310d-bestval.h5 \
        anssel data/anssel/yodaqa/curatedv2-training.csv data/anssel/yodaqa/curatedv2-val.csv \
        pdim=1 ptscorer=B.dot_ptscorer dropout=0 inp_e_dropout=0

VOCAB can be either a specialized vocabulary file if task1
uses one, or just the training set used to train the original
model.

When transfer learning from Ubuntu dataset, please note that all Ubuntu
sentences are terminated by special tokens __eou__ __eot__ - it makes
sense to decorate your target dataset with the same tokens, which can
be done by passing adapt_ubuntu=True config parameter.  (Do that for
both transfer and eval!)
"""

from __future__ import print_function
from __future__ import division

import importlib
import numpy as np
try:
    import cPickle
except ImportError:  # python3
    import pickle as cPickle
import pickle
import sys

from keras.callbacks import EarlyStopping, ModelCheckpoint

import pysts.embedding as emb
import pysts.eval as ev
import pysts.loader as loader
from pysts.hyperparam import hash_params

import pysts.kerasts.blocks as B
from pysts.kerasts.objectives import pearsonobj

from train import config, train_model
import models  # importlib python3 compatibility requirement
import tasks


def transfer_eval(runid, module_prep_model, task1, task2, weightsf, c):
    if 'model1' in c:
        # Support for transfer across models (e.g. from rnn to attn1511)
        # XXX: No separate model1_conf
        model1_module = importlib.import_module('.'+c['model1'], 'models')
        module_prep_model1 = model1_module.prep_model
    else:
        module_prep_model1 = module_prep_model

    # We construct both original and new model, then copy over
    # the weights from the original model
    print('Model')
    model1 = task1.build_model(module_prep_model1, do_compile=False)
    model = task2.build_model(module_prep_model)
    print('Model (weights)')
    model1.load_weights(weightsf)
    for n in model1.nodes.keys():
        if n in model.nodes:
            model.nodes[n].set_weights(model1.nodes[n].get_weights())
        else:
            print('- skipping (not in target) ' + n)
    print('Pre-training Transfer Evaluation')
    task2.eval(model)

    train_model(runid, model, task2, c)

    print('Predict&Eval (best val epoch)')
    res = task2.eval(model)
    return model, res


if __name__ == "__main__":
    modelname, task1name, vocab1f, weightsf, task2name, train2f, val2f = sys.argv[1:8]
    params = sys.argv[8:]

    model_module = importlib.import_module('.'+modelname, 'models')

    task1_module = importlib.import_module('.'+task1name, 'tasks')
    task1 = task1_module.task()
    task2_module = importlib.import_module('.'+task2name, 'tasks')
    task2 = task2_module.task()

    # setup conf with task2, because that's where we'll be doing
    # our training
    conf, ps, h = config(model_module.config, task2.config, params)
    task1c = dict(conf)
    if 'task1_conf' in conf:
        for k, v in conf.pop('task1_conf').items():
            task1c[k] = v
    task1.set_conf(task1c)
    task2.set_conf(conf)

    # TODO configurable embedding class
    if conf['embdim'] is not None:
        print('GloVe')
        task2.emb = emb.GloVe(N=conf['embdim'])
        task1.emb = task2.emb

    print('Dataset 1')
    task1.load_vocab(vocab1f)
    task2.vocab = task1.vocab
    print('Dataset 2')
    task2.load_data(train2f, val2f)

    for i_run in range(conf['nb_runs']):
        if conf['nb_runs'] == 1:
            runid = '%s-%s-%s-%x' % (task1name, task2name, modelname, h)
        else:
            runid = '%s-%s-%s-%x-%02d' % (task1name, task2name, modelname, h, i_run)
        print('RunID: %s  (%s)' % (runid, ps))

        transfer_eval(runid, model_module.prep_model, task1, task2, weightsf, conf)
