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

Performance:
    * anssel/wang train-all (c.f. http://aclweb.org/aclwiki/index.php?title=Question_Answering_(State_of_the_art)):
        Train Accuracy: raw 0.707709 (y=0 0.684140, y=1 0.731279), bal 0.707709
        Train MRR: 0.715685  (on training set, y=0 is subsampled!)
        Test Accuracy: raw 0.681692 (y=0 0.689280, y=1 0.645161), bal 0.667221
        Test MRR: 0.671137
    * anssel/yodaqa:
        Train Accuracy: raw 0.736470 (y=0 0.721243, y=1 0.751696), bal 0.736470
        Train MRR: 0.490585  (on training set, y=0 is subsampled!)
        Test Accuracy: raw 0.634714 (y=0 0.640241, y=1 0.540921), bal 0.590581
        Test MRR: 0.330349

(N.B. (Yu, 2014) also trains a second-level classifier that combines
this similarity with |common_keywords|.  We don't do that here - exercise
for the reader, yields state-of-art-2014 MRR on anssel/wang!)

A more complicated implementation of this script lives in
https://github.com/brmson/Sentence-selection
"""

from __future__ import print_function

import argparse
import numpy as np
from sklearn import linear_model

import pysts.embedding as emb
import pysts.loader as loader
import pysts.eval as ev


def load_set(glove, fname, balance=False, subsample0=3):
    s0, s1, labels, _, _, _ = loader.load_anssel(fname, subsample0=subsample0)
    print('(%s) Loaded dataset: %d' % (fname, len(s0)))
    e0, e1, s0, s1, labels = loader.load_embedded(glove, s0, s1, labels, balance=balance)
    return ([e0, e1], labels)


def logreg_M(e0, e1):
    # To train the projection matrix M, we expand X to pairwise element multiplications instead of just concatenating e0, e1
    return np.array([np.ravel(np.outer(e0[i], e1[i])) for i in range(len(e0))])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Benchmark yu1412 on binary classification / point ranking task (anssel-yodaqa)")
    parser.add_argument("-N", help="GloVe dim", type=int, default=50)  # for our naive method, 300**2 would be too much
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

    logreg = linear_model.LogisticRegression(C=0.01, verbose=1, n_jobs=7)
    logreg.fit(logreg_M(*Xtrain), ytrain)
    ev.eval_anssel(logreg.predict_proba(logreg_M(*Xtrain))[:, 1], Xtrain[0], Xtrain[1], ytrain, 'Train')
    ev.eval_anssel(logreg.predict_proba(logreg_M(*Xtest))[:, 1], Xtest[0], Xtest[1], ytest, 'Test')


"""
Performance tuning on anssel-yodaqa:
    * Completely unbalanced, C=1
        Train Accuracy: 0.899176 (y=0 0.983992, y=1 0.334139)
        Train MRR: 0.626233  (on training set, y=0 is subsampled!)
        Test Accuracy: 0.926688 (y=0 0.965770, y=1 0.095908)
        Test MRR: 0.218704
    * sklearn balancing (class_weight='auto'), C=1
        Train Accuracy: 0.816569 (y=0 0.812480, y=1 0.843807)
        Train MRR: 0.620643  (on training set, y=0 is subsampled!)
        Test Accuracy: 0.714450 (y=0 0.727787, y=1 0.430946)
        Test MRR: 0.235821
    * Our balancing (balance_dataset()), C=1
        Train Accuracy: raw 0.837037 (y=0 0.821142, y=1 0.852932), bal 0.837037
        Train MRR: 0.627617  (on training set, y=0 is subsampled!)
        Test Accuracy: raw 0.709681 (y=0 0.723455, y=1 0.416880), bal 0.570168
        Test MRR: 0.245249
    * Our balancing, C=0.1 (stronger regularization):
        Train Accuracy: raw 0.800009 (y=0 0.789987, y=1 0.810031), bal 0.800009
        Train MRR: 0.578199  (on training set, y=0 is subsampled!)
        Test Accuracy: raw 0.709796 (y=0 0.721891, y=1 0.452685), bal 0.587288
        Test MRR: 0.257044
    * Our balancing, C=0.05
        Train Accuracy: raw 0.787538 (y=0 0.778060, y=1 0.797016), bal 0.787538
        Train MRR: 0.555112  (on training set, y=0 is subsampled!)
        Test Accuracy: raw 0.709107 (y=0 0.720869, y=1 0.459079), bal 0.589974
        Test MRR: 0.281093
    * Our balancing, C=0.01 ****** default settings
        Train Accuracy: raw 0.746315 (y=0 0.740148, y=1 0.752483), bal 0.746315
        Train MRR: 0.516240  (on training set, y=0 is subsampled!)
        Test Accuracy: raw 0.699454 (y=0 0.709318, y=1 0.489770), bal 0.599544
        Test MRR: 0.301773
    * Our balancing, C=0.005
        Train Accuracy: raw 0.733028 (y=0 0.726090, y=1 0.739966), bal 0.733028
        Train MRR: 0.493889  (on training set, y=0 is subsampled!)
        Test Accuracy: raw 0.691755 (y=0 0.701197, y=1 0.491049), bal 0.596123
        Test MRR: 0.296532
    * Our balancing, C=0.001
        Train Accuracy: raw 0.709129 (y=0 0.708176, y=1 0.710081), bal 0.709129
        Train MRR: 0.468256  (on training set, y=0 is subsampled!)
        Test Accuracy: raw 0.674059 (y=0 0.683090, y=1 0.482097), bal 0.582593
        Test MRR: 0.269031

"""
