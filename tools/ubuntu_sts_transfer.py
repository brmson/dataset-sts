#!/usr/bin/python3
"""
Load a KeraSTS model on the Ubuntu Dialogue task and reuse it on an anssel task.

Usage: tools/ubuntu_anssel_transfer.py MODEL WEIGHTS VOCAB ATRAINDATA AVALDATA [PARAM=VALUE]...

Example: tools/ubuntu_anssel_transfer.py rnn ubu-weights-rnn--23fa2eff7cda310d-bestval.h5 data/anssel/ubuntu/v2-vocab.pickle data/anssel/yodaqa/curatedv2-training.csv data/anssel/yodaqa/curatedv2-val.csv pdim=1 ptscorer=B.dot_ptscorer
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
from pysts.kerasts.callbacks import AnsSelCB
from pysts.kerasts.objectives import pearsonobj

import anssel_train
import sts_train
import anssel_train
import models  # importlib python3 compatibility requirement


# XXX: Not the ubuntu_train default, obviously; but allows for fast training
# of big models.
spad = 80


def config(module_config, params):
    c = dict()
    c['embdim'] = 300
    c['inp_e_dropout'] = 1/2
    c['inp_w_dropout'] = 0
    c['e_add_flags'] = True

    c['ptscorer'] = B.mlp_ptscorer
    c['mlpsum'] = 'sum'
    c['Ddim'] = 1

    c['opt'] = 'adam'
    c['loss'] = pearsonobj
    c['batch_size'] = 64
    c['nb_epoch'] = 16

    c['fix_layers'] = []

    module_config(c)

    for p in params:
        k, v = p.split('=')
        c[k] = eval(v)

    ps, h = hash_params(c)
    return c, ps, h


def transfer_eval(runid, weightsf, module_prep_model, c, glove, vocab, gr, grv):
    # We construct both original and sts model, then copy over
    # the weights from the original model
    print('Model')
    umodel = anssel_train.build_model(glove, vocab, module_prep_model, c, s0pad=spad, s1pad=spad, do_compile=False)
    model = sts_train.build_model(glove, vocab, module_prep_model, c, spad=spad, optimizer=c['opt'], fix_layers=c['fix_layers'])
    print('Model (weights)')
    umodel.load_weights(weightsf)
    for n in umodel.nodes.keys():
        model.nodes[n].set_weights(umodel.nodes[n].get_weights())
    ev.eval_sts(model.predict(grv)['classes'][:,0], grv['classes'], 'sts Val (bef. train)')

    print('Training')
    model.fit(gr, validation_data=grv,
              callbacks=[STSPearsonCB(gr, grv),
                         ModelCheckpoint('sts-weights-'+runid+'-bestval.h5', save_best_only=True),
                         EarlyStopping(patience=4)],
              batch_size=conf['batch_size'], nb_epoch=conf['nb_epoch'])
    model.save_weights('sts-weights-'+runid+'-final.h5', overwrite=True)

    print('Predict&Eval (best epoch)')
    model.load_weights('sts-weights-'+runid+'-bestval.h5')
    ev.eval_sts(model.predict(grv)['classes'][:,0], grv['classes'], 'sts Val')


if __name__ == "__main__":
    modelname, weightsf, vocabf = sys.argv[1:4]
    g = ([], [], [])
    g_i = 0
    for p in sys.argv[4:]:
        if p == '--':
            g_i += 1
            continue
        g[g_i].append(p)
    (trainf, valf, params) = g

    module = importlib.import_module('.'+modelname, 'models')
    conf, ps, h = config(module.config, params)

    runid = '%s-%x' % (modelname, h)
    print('RunID: %s  (%s)' % (runid, ps))

    print('GloVe')
    glove = emb.GloVe(N=conf['embdim'])

    print('Dataset (vocab)')
    vocab = pickle.load(open(vocabf, "rb"))  # use plain pickle because unicode

    print('Dataset (sts train)')
    s0, s1, y, _, gr_ = sts_train.load_set(trainf, vocab, spad=spad)
    gr = loader.graph_adapt_ubuntu(gr_, vocab)
    print('Dataset (sts val)')
    s0v, s1v, yv, _, grv_ = sts_train.load_set(valf, vocab, spad=spad)
    grv = loader.graph_adapt_ubuntu(grv_, vocab)

    transfer_eval(runid, weightsf, module.prep_model, conf, glove, vocab, gr, grv)
