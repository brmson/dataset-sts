#!/usr/bin/python3
"""
An example sts2015 classifier using the (Yu, 2014) approach
http://arxiv.org/abs/1412.1632 of training (M, b) such that:

    f(q, a) = sigmoid(q * M * a.T + b)

We use a trivial keras model to train an embedding projection M
while using bag-of-words average for the two sentenes.

Prerequisites:
    * Get glove.6B.50d.txt from http://nlp.stanford.edu/projects/glove/

Performance (3000 iters):
    9092/9092 [==============================] - 0s - loss: 1.5190 - acc: 0.3543 - val_loss: 1.9104 - val_acc: 0.1753
    Train Pearson: 0.655718
    Train Spearman: 0.586274
    Train MSE: 1.233807
    Test Pearson: 0.236014
    Test Spearman: 0.228541
    Test MSE: 2.705417
"""

from __future__ import print_function

import glob
import numpy as np
from sklearn.metrics import mean_squared_error as mse
from scipy.stats import pearsonr
from scipy.stats import spearmanr

from keras.models import Sequential
from keras.layers.core import Activation, Dense, Dropout, Flatten, Merge, RepeatVector
from keras.regularizers import l2

import pysts.embedding as emb
import pysts.loader as loader


def load_set(glove, globmask):
    s0, s1, labels = loader.concat_datasets([loader.load_sts(d) for d in glob.glob(globmask)])
    print('(%s) Loaded dataset: %d' % (globmask, len(s0)))

    # for padding and sequences (e.g. keras RNNs):
    # print('(%s) s0[-1000]: %d tokens' % (globmask, np.sort([np.shape(s) for s in s0], axis=0)[-1000]))
    # print('(%s) s1[-1000]: %d tokens' % (globmask, np.sort([np.shape(s) for s in s1], axis=0)[-1000]))
    # s0 = glove.pad_set(s0, 40)
    # s1 = glove.pad_set(s1, 40)

    # for averaging:
    s0 = glove.map_set(s0, ndim=1)
    s1 = glove.map_set(s1, ndim=1)
    return ([np.array(s0), np.array(s1)], labels)


def prep_model(glove, dropout=0.1, l2reg=1e-3):
    # XXX: this is a bit hacky to combine dot-product with six-wise output
    model0 = Sequential()  # s0
    model0.add(Dropout(dropout, input_shape=(glove.N,)))
    model0.add(Dense(input_dim=glove.N, output_dim=glove.N, W_regularizer=l2(l2reg)))  # M matrix
    model0.add(RepeatVector(6))  # [nclass]
    model1 = Sequential()  # s1
    model1.add(Dropout(dropout, input_shape=(glove.N,)))
    model1.add(RepeatVector(6))  # [nclass]

    model = Sequential()
    model.add(Merge([model0, model1], mode='dot', dot_axes=([2], [2])))
    model.add(Flatten())  # 6x6 matrix with cross-activations -> 36 vector
    model.add(Dense(6, W_regularizer=l2(l2reg)))  # 36 vector -> 6 vector, ugh
    model.add(Activation('softmax'))
    return model


def eval_set(model, X, y, name):
    ycat = model.predict_proba(X)
    ypred = loader.sts_categorical2labels(ycat)
    print('%s Pearson: %f' % (name, pearsonr(ypred, y)[0],))
    print('%s Spearman: %f' % (name, spearmanr(ypred, y)[0],))
    print('%s MSE: %f' % (name, mse(ypred, y),))


if __name__ == "__main__":
    glove = emb.GloVe(N=50)  # for our naive method, 300**2 would be too much
    Xtrain, ytrain = load_set(glove, 'sts/all/201[0-4]*')
    Xtest, ytest = load_set(glove, 'sts/all/2015*')

    model = prep_model(glove)
    model.compile(loss='categorical_crossentropy', optimizer='adam')
    model.fit(Xtrain, loader.sts_labels2categorical(ytrain), batch_size=80, nb_epoch=30, show_accuracy=True,
              validation_data=(Xtest, loader.sts_labels2categorical(ytest)))
    eval_set(model, Xtrain, ytrain, 'Train')
    eval_set(model, Xtest, ytest, 'Test')
