#!/usr/bin/python3
"""
An example anssel classifier using the (Tai, 2015) approach
http://arxiv.org/abs/1503.00075 with mean vectors and model as in 4.2,
using the awesome Keras deep learning library.

Play with it to see effect of GloVe dimensionality, hidden layer
(and its size), various regularization etc.!

TODO: KL cost function

Prerequisites:
    * Get glove.6B.300d.txt from http://nlp.stanford.edu/projects/glove/

Performance (2000 iters):
    * YodaQA
        38026/38026 [==============================] - 5s - loss: 0.4670 - val_loss: 0.4741
        Train Accuracy: raw 0.851575 (y=0 0.810761, y=1 0.892389), bal 0.851575
        Train MRR: 0.744192  (on training set, y=0 is subsampled!)
        Test Accuracy: raw 0.777469 (y=0 0.795328, y=1 0.474425), bal 0.634876
        Test MRR: 0.276420
    * wang
        25372/25372 [==============================] - 3s - loss: 0.5956 - val_loss: 0.5407
        Train Accuracy: raw 0.752601 (y=0 0.705423, y=1 0.799779), bal 0.752601
        Train MRR: 0.806276  (on training set, y=0 is subsampled!)
        Test Accuracy: raw 0.755201 (y=0 0.809883, y=1 0.491935), bal 0.650909
        Test MRR: 0.620635
"""

from __future__ import print_function

import argparse

from keras.models import Graph
from keras.layers.core import Activation, Dense, Dropout
from keras.regularizers import l2

import pysts.embedding as emb
import pysts.loader as loader
import pysts.eval as ev


def load_set(glove, fname, balance=False, subsample0=3):
    s0, s1, labels, _, _, _ = loader.load_anssel(fname, subsample0=subsample0)
    print('(%s) Loaded dataset: %d' % (fname, len(s0)))
    e0, e1, s0, s1, labels = loader.load_embedded(glove, s0, s1, labels, balance=balance)
    return ([e0, e1], labels)


def prep_model(glove, dropout=1/2, l2reg=1e-4):
    model = Graph()

    # Process sentence embeddings
    model.add_input(name='e0', input_shape=(glove.N,))
    model.add_input(name='e1', input_shape=(glove.N,))
    model.add_node(name='e0_', input='e0',
                   layer=Dropout(dropout))
    model.add_node(name='e1_', input='e1',
                   layer=Dropout(dropout))

    # Generate element-wise features from the pair
    # (the Activation is a nop, merge_mode is the important part)
    model.add_node(name='sum', inputs=['e0_', 'e1_'], layer=Activation('linear'), merge_mode='sum')
    model.add_node(name='mul', inputs=['e0_', 'e1_'], layer=Activation('linear'), merge_mode='mul')

    # Use MLP to generate classes
    model.add_node(name='hidden', inputs=['sum', 'mul'], merge_mode='concat',
                   layer=Dense(50, W_regularizer=l2(l2reg)))
    model.add_node(name='hiddenS', input='hidden',
                   layer=Activation('sigmoid'))
    model.add_node(name='out', input='hiddenS',
                   layer=Dense(1, W_regularizer=l2(l2reg)))
    model.add_node(name='outS', input='out',
                   layer=Activation('sigmoid'))

    model.add_output(name='score', input='outS')
    return model


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Benchmark kst1503 on binary classification / point ranking task (anssel-yodaqa)")
    parser.add_argument("-N", help="GloVe dim", type=int, default=300)
    parser.add_argument("--balance", help="whether to manually balance the dataset", type=int, default=1)
    parser.add_argument("--wang", help="whether to run on Wang inst. of YodaQA dataset", type=int, default=0)
    args = parser.parse_args()

    glove = emb.GloVe(N=args.N)
    if args.wang == 1:
        Xtrain, ytrain = load_set(glove, 'data/anssel/wang/train-all.csv', balance=(args.balance == 1))
        Xtest, ytest = load_set(glove, 'data/anssel/wang/test.csv', subsample0=1)
    else:
        Xtrain, ytrain = load_set(glove, 'data/anssel/yodaqa/curatedv1-training.csv', balance=(args.balance == 1))
        Xtest, ytest = load_set(glove, 'data/anssel/yodaqa/curatedv1-val.csv', subsample0=1)

    model = prep_model(glove)
    model.compile(loss={'score': 'binary_crossentropy'}, optimizer='adam')
    model.fit({'e0': Xtrain[0], 'e1': Xtrain[1], 'score': ytrain},
              batch_size=20, nb_epoch=2000,
              validation_data={'e0': Xtest[0], 'e1': Xtest[1], 'score': ytest})
    ev.eval_anssel(model.predict({'e0': Xtrain[0], 'e1': Xtrain[1]})['score'][:, 0], Xtrain[0], Xtrain[1], ytrain, 'Train')
    ev.eval_anssel(model.predict({'e0': Xtest[0], 'e1': Xtest[1]})['score'][:, 0], Xtest[0], Xtest[1], ytest, 'Test')
