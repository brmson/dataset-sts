#!/usr/bin/python3
"""
Tool for evaluating the Ubuntu models based on the accuracy@N metrics.

Usage: tools/ubuntu_eval.py MODEL WEIGHTSFILE TRAINDATA VALDATA TREC_QRELS_FILE TREC_TOP_FILE [PARAM=VALUE]...
Example: tools/ubuntu_eval.py cnn ubu-weights-cnn-bestval.h5 data/anssel/ubuntu/v1-vocab.pickle data/anssel/ubuntu/v1-trainset.pickle data/anssel/ubuntu/v1-valset.pickle sdim=3
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

from keras.layers.convolutional import MaxPooling1D, AveragePooling1D
from keras.layers.recurrent import SimpleRNN, GRU, LSTM
import pysts.embedding as emb
import pysts.eval as ev
import pysts.kerasts.blocks as B
from pysts.kerasts.objectives import ranknet

import anssel_train
import ubuntu_train
import models  # importlib python3 compatibility requirement


if __name__ == "__main__":
    modelname, weightsfile, vocabf, trainf, valf = sys.argv[1:6]
    params = sys.argv[6:]

    module = importlib.import_module('.'+modelname, 'models')
    conf, ps, h = anssel_train.config(module.config, params)

    print('GloVe')
    glove = emb.GloVe(N=conf['embdim'])

    print('Dataset (vocab)')
    vocab = pickle.load(open(vocabf, "rb"))  # use plain pickle because unicode
    print('%d words loaded' % (len(vocab.word_idx),))

    print('Dataset (val)')
    grt = ubuntu_train.load_set(valf, vocab)
    print('Padding (val)')
    ubuntu_train.pad_graph(grt)

    print('Model')
    model = anssel_train.build_model(glove, vocab, module.prep_model, conf, s0pad=ubuntu_train.s0pad, s1pad=ubuntu_train.s1pad)
    print('%d parameters (except embedding matrix)' % (model.count_params() - model.nodes['emb'].count_params(),))

    print('Weights')
    model.load_weights(weightsfile)

    print('Prediction (val)')
    ypredt = model.predict(grt)['score'][:,0]

    ev.eval_ubuntu(ypredt, grt['si0'], grt['score'], valf)

    # print('Dataset (train)')
    # gr = ubuntu_train.load_set(trainf, vocab)
    # print('Prediction (train)')
    # ypred = model.predict(gr)['score'][:,0]
    # ev.eval_ubuntu(ypred, gr['si0'], gr['score'], trainf)
