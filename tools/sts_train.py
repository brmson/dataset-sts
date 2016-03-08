#!/usr/bin/python3
"""
Train a KeraSTS model on the Semantic Text Similarity task.

Usage: tools/sts_train.py MODEL TRAINDATA... -- VALDATA... [-- PARAM=VALUE...]

Example: tools/sts_train.py rnn data/sts/semeval-sts/all/201[0-4]* -- data/sts/semeval-sts/all/2015*

This applies the given text similarity model to the sts task.
Extra input pre-processing is done:
Rather than relying on the hack of using the word overlap counts as additional
features for final classification, individual tokens are annotated by overlap
features and that's passed to the model along with the embeddings.

Prerequisites:
    * Get glove.6B.300d.txt from http://nlp.stanford.edu/projects/glove/
"""

from __future__ import print_function
from __future__ import division

import importlib
import sys

from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.layers.core import Activation, Dense, Dropout
from keras.layers.recurrent import SimpleRNN, GRU, LSTM
from keras.models import Graph
from keras.regularizers import l2

import pysts.embedding as emb
import pysts.eval as ev
import pysts.loader as loader
import pysts.nlp as nlp
from pysts.hyperparam import hash_params
from pysts.vocab import Vocabulary

from pysts.kerasts import graph_input_sts
import pysts.kerasts.blocks as B
from pysts.kerasts.callbacks import STSPearsonCB
from pysts.kerasts.objectives import pearsonobj
import models  # importlib python3 compatibility requirement


spad = 60


def load_set(files, vocab=None, skip_unlabeled=True):
    def load_file(fname, skip_unlabeled=True):
        # XXX: ugly logic
        if 'sick2014' in fname:
            return loader.load_sick2014(fname)
        else:
            return loader.load_sts(fname, skip_unlabeled=skip_unlabeled)
    s0, s1, y = loader.concat_datasets([load_file(d, skip_unlabeled=skip_unlabeled) for d in files])

    if vocab is None:
        vocab = Vocabulary(s0 + s1)

    si0 = vocab.vectorize(s0, spad=spad)
    si1 = vocab.vectorize(s1, spad=spad)
    f0, f1 = nlp.sentence_flags(s0, s1, spad, spad)
    gr = graph_input_sts(si0, si1, y, f0, f1)

    return (s0, s1, y, vocab, gr)


def config(module_config, params):
    c = dict()
    c['embdim'] = 300
    c['inp_e_dropout'] = 3/4
    c['inp_w_dropout'] = 0
    c['e_add_flags'] = True

    c['ptscorer'] = B.dot_ptscorer
    c['mlpsum'] = 'sum'
    c['Ddim'] = 2

    c['loss'] = pearsonobj  # ...or 'categorical_crossentropy'
    c['batch_size'] = 160
    c['nb_epoch'] = 32
    module_config(c)

    for p in params:
        k, v = p.split('=')
        c[k] = eval(v)

    ps, h = hash_params(c)
    return c, ps, h


def prep_model(glove, vocab, module_prep_model, c):
    # Input embedding and encoding
    model = Graph()
    N = B.embedding(model, glove, vocab, spad, spad, c['inp_e_dropout'], c['inp_w_dropout'], add_flags=c['e_add_flags'])

    # Sentence-aggregate embeddings
    final_outputs = module_prep_model(model, N, spad, spad, c)

    # Measurement
    kwargs = dict()
    if c['ptscorer'] == B.mlp_ptscorer:
        kwargs['sum_mode'] = c['mlpsum']
    model.add_node(name='scoreS', input=c['ptscorer'](model, final_outputs, c['Ddim'], N, c['l2reg'], **kwargs),
                   layer=Activation('linear'))
    model.add_node(name='out', input='scoreS',
                   layer=Dense(6, W_regularizer=l2(c['l2reg'])))
    model.add_node(name='outS', input='out',
                   layer=Activation('softmax'))

    model.add_output(name='classes', input='outS')
    return model


def build_model(glove, vocab, module_prep_model, c):
    model = prep_model(glove, vocab, module_prep_model, c)
    model.compile(loss={'classes': c['loss']}, optimizer='adam')
    return model


def train_and_eval(runid, module_prep_model, c, glove, vocab, gr, grt):
    print('Model')
    model = build_model(glove, vocab, module_prep_model, c)

    print('Training')
    # XXX: samples_per_epoch is in brmson/keras fork, TODO fit_generator()?
    model.fit(gr, validation_data=grt,
              callbacks=[STSPearsonCB(gr, grt),
                         ModelCheckpoint('sts-weights-'+runid+'-bestval.h5', save_best_only=True, monitor='pearson', mode='max'),
                         EarlyStopping(monitor='pearson', mode='max', patience=3)],
              batch_size=c['batch_size'], nb_epoch=c['nb_epoch'])
    model.save_weights('sts-weights-'+runid+'-final.h5', overwrite=True)

    print('Predict&Eval')
    model.load_weights('sts-weights-'+runid+'-bestval.h5')
    ev.eval_sts(model.predict(gr)['classes'], gr['classes'], 'Train')
    ev.eval_sts(model.predict(grt)['classes'], grt['classes'], 'Val')


if __name__ == "__main__":
    modelname = sys.argv[1]
    g = ([], [], [])
    g_i = 0
    for p in sys.argv[2:]:
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

    print('Dataset')
    s0, s1, y, vocab, gr = load_set(trainf)
    s0t, s1t, yt, _, grt = load_set(valf, vocab)

    train_and_eval(runid, module.prep_model, conf, glove, vocab, gr, grt)
