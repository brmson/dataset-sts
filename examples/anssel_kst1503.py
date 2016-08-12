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

from keras.models import Model
from keras.layers import Dense, Dropout, Input, merge
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
    # Process sentence embeddings
    e0 = Input(shape=(glove.N,), name='e0')
    e1 = Input(shape=(glove.N,), name='e1')
    # dropout here triggers keras error, wtf?

    # Generate element-wise features from the pair
    # (the Activation is a nop, merge_mode is the important part)
    ew_sum = merge([e0, e1], mode='sum')
    ew_mul = merge([e0, e1], mode='mul')
    ew = merge([ew_sum, ew_mul], mode='concat', concat_axis=-1)

    # Use MLP to generate classes
    hidden = Dense(glove.N*2, activation='sigmoid', W_regularizer=l2(l2reg), init='identity')(Dropout(dropout)(ew))
    score = Dense(1, activation='sigmoid', W_regularizer=l2(l2reg), name='score')(hidden)

    return Model(input=[e0, e1], output=score)


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
    model.compile(loss={'score': 'binary_crossentropy'}, optimizer='adam', metrics=['accuracy'])
    model.fit({'e0': Xtrain[0], 'e1': Xtrain[1]}, {'score': ytrain},
              batch_size=20, nb_epoch=2000,
              validation_data=({'e0': Xtest[0], 'e1': Xtest[1]}, {'score': ytest}))
    ev.eval_anssel(model.predict({'e0': Xtrain[0], 'e1': Xtrain[1]})[:, 0], Xtrain[0], Xtrain[1], ytrain, 'Train')
    ev.eval_anssel(model.predict({'e0': Xtest[0], 'e1': Xtest[1]})[:, 0], Xtest[0], Xtest[1], ytest, 'Test')
