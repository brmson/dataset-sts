#!/usr/bin/python3
"""
An example anssel classifier using the (Yu, 2014) approach
http://arxiv.org/abs/1412.1632 of training (M, b) such that:

    f(q, a) = sigmoid(q * M * a.T + b)

We use sklearn's logistic regression to train (flattened) M and b,
while using bag-of-words q and a.  This should be roughly equivalent
to the paper's unigram method, but we use GloVe vectors which is better.

Prerequisites:
    * Get glove.6B.50d.txt from http://nlp.stanford.edu/projects/glove/

Performance on anssel-yodaqa:
    * Completely unbalanced
        Train Accuracy: 0.899176 (y=0 0.983992, y=1 0.334139)
        Train MRR: 0.626233  (on training set, y=0 is subsampled!)
        Test Accuracy: 0.926688 (y=0 0.965770, y=1 0.095908)
        Test MRR: 0.218704
    * sklearn balancing (class_weight='auto')
    * Manual balancing (balance_dataset())
        Train Accuracy: 0.841345 (y=0 0.820824, y=1 0.861866)
        Train MRR: 0.622557  (on training set, y=0 is subsampled!)
        Test Accuracy: 0.573783 (y=0 0.729471, y=1 0.418095)
        Test MRR: 0.245059
"""

from __future__ import print_function

import argparse
import numpy as np
from sklearn import linear_model

import pysts.embedding as emb
import pysts.loader as loader
import pysts.eval as ev


def load_set(glove, fname, balance=True, subsample0=3):
    s0, s1, labels = loader.load_anssel_yodaqa(fname, subsample0=subsample0)
    print('(%s) Loaded dataset: %d' % (fname, len(s0)))

    if balance:
        s0, s1, labels = loader.balance_dataset((s0, s1, labels))

    # for padding and sequences (e.g. keras RNNs):
    # print('(%s) s0[-1000]: %d tokens' % (globmask, np.sort([np.shape(s) for s in s0], axis=0)[-1000]))
    # print('(%s) s1[-1000]: %d tokens' % (globmask, np.sort([np.shape(s) for s in s1], axis=0)[-1000]))
    # s0 = glove.pad_set(s0, 25)
    # s1 = glove.pad_set(s1, 60)
    # return (np.hstack((s0, s1)), labels)

    # for averaging:
    s0 = glove.map_set(s0, ndim=1)
    s1 = glove.map_set(s1, ndim=1)
    # To train the projection matrix M, we expand X to pairwise element multiplications instead of just concatenating s0, s1
    X = np.array([np.ravel(np.outer(s0[i], s1[i])) for i in range(len(s0))])
    return (s0, X, labels)


def eval_set(logreg, s0, X, y, name):
    ypred = logreg.predict_proba(X)[:, 1]
    rawacc, y0acc, y1acc, balacc = ev.binclass_accuracy(y, ypred)
    print('%s Accuracy: raw %f (y=0 %f, y=1 %f), bal %f' % (name, rawacc, y0acc, y1acc, balacc))
    print('%s MRR: %f  %s' % (name, ev.mrr(s0, y, ypred), '(on training set, y=0 is subsampled!)' if name == 'Train' else ''))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Benchmark yu1412 on binary classification / point ranking task (anssel-yodaqa)")
    parser.add_argument("-N", help="GloVe dim", type=int, default=50)  # for our naive method, 300**2 would be too much
    parser.add_argument("--balance", help="whether to manually balance the dataset", type=int, default=1)
    args = parser.parse_args()

    glove = emb.GloVe(N=args.N)
    s0train, Xtrain, ytrain = load_set(glove, 'anssel-yodaqa/curatedv1-training.csv', balance=(args.balance == 1))
    s0test, Xtest, ytest = load_set(glove, 'anssel-yodaqa/curatedv1-val.csv', subsample0=1)

    logreg = linear_model.LogisticRegression(verbose=1, n_jobs=7)
    logreg.fit(Xtrain, ytrain)
    eval_set(logreg, s0train, Xtrain, ytrain, 'Train')
    eval_set(logreg, s0test, Xtest, ytest, 'Test')
