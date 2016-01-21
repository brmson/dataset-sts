#!/usr/bin/python3
"""
An example sts2015 classifier using the (Yu, 2014) approach
http://arxiv.org/abs/1412.1632 of training (M, b) such that:

    f(q, a) = sigmoid(q * M * a.T + b)

We use sklearn's logistic regression to train (flattened) M and b,
while using q.

Prerequisites:
    * Get glove.6B.50d.txt from http://nlp.stanford.edu/projects/glove/

FIXME:
    * Instead of treating y-score as continuous, try categorical approach
      with label encoding from Tree LSTM paper (Tai, Socher, Manning)
      (as in eval_sick)
"""

from __future__ import print_function

import glob
import numpy as np
from sklearn.metrics import mean_squared_error as mse
from scipy.stats import pearsonr
from scipy.stats import spearmanr
from sklearn import linear_model

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
    # return (np.hstack((s0, s1)), labels)

    # for averaging:
    s0 = glove.map_set(s0, ndim=1)
    s1 = glove.map_set(s1, ndim=1)
    # To train the projection matrix M, we expand X to pairwise element multiplications instead of just concatenating s0, s1
    X = np.array([np.ravel(np.outer(s0[i], s1[i])) for i in range(len(s0))])
    return (X, labels)


def eval_set(logreg, X, y, name):
    ypred = logreg.predict_proba(X)[:, 1]
    print('%s Pearson: %f' % (name, pearsonr(ypred, y)[0],))
    print('%s Spearman: %f' % (name, spearmanr(ypred, y)[0],))
    print('%s MSE: %f' % (name, mse(ypred, y),))


if __name__ == "__main__":
    glove = emb.GloVe(N=50)  # for our naive method, 300**2 would be too much
    Xtrain, ytrain = load_set(glove, 'sts/all/201[0-4]*')
    Xtest, ytest = load_set(glove, 'sts/all/2015*')

    logreg = linear_model.LogisticRegression(solver='sag', verbose=1)
    # XXX: we do a horrible cheat here as sklearn logreg is categorical
    # logreg.fit(Xtrain, ytrain)
    logreg.fit(Xtrain, ytrain > 0.5, sample_weight=np.abs(0.5-ytrain))
    eval_set(logreg, Xtrain, ytrain, 'Train')
    eval_set(logreg, Xtest, ytest, 'Test')
