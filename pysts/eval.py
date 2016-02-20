"""
Evaluation tools, mainly non-straightforward methods.
"""

from __future__ import print_function
from __future__ import division

import numpy as np
from scipy.stats import pearsonr
from scipy.stats import spearmanr
from sklearn.metrics import mean_squared_error as mse

from . import loader


def binclass_accuracy(y, ypred):
    """
    Compute accuracy for binary classification tasks, taking into account
    grossly unbalanced datasets.

    Returns (rawacc, y0acc, y1acc, balacc) where balacc is average of y0acc
    and y1acc, regardless of their true balance in the dataset.

    (The idea is that even if the unfortunate reality is that we have much
    less y1 samples, their detection is equally important.)
    """
    rawacc = np.sum((ypred > 0.5) == (y > 0.5)) / ypred.shape[0]
    y0acc = np.sum(np.logical_and(ypred < 0.5, y < 0.5)) / np.sum(y < 0.5)
    y1acc = np.sum(np.logical_and(ypred > 0.5, y > 0.5)) / np.sum(y > 0.5)
    balacc = (y0acc + y1acc) / 2
    return (rawacc, y0acc, y1acc, balacc)


def aggregate_s0(s0, y, ypred, k=None):
    """
    Generate tuples (s0, [(y, ypred), ...]) where the list is sorted
    by the ypred score.  This is useful for a variety of list-based
    measures in the "anssel"-type tasks.
    """
    ybys0 = dict()
    for i in range(len(s0)):
        try:
            s0is = s0[i].tostring()
        except AttributeError:
            s0is = str(s0[i])
        if s0is in ybys0:
            ybys0[s0is].append((y[i], ypred[i]))
        else:
            ybys0[s0is] = [(y[i], ypred[i])]

    for s, yl in ybys0.items():
        if k is not None:
            yl = yl[:k]
        ys = sorted(yl, key=lambda yy: yy[1], reverse=True)
        yield (s, ys)


def recall_at(s0, y, ypred, N, k=None):
    """
    Compute Recall@N, that is, the expected probability of whether
    y==1 is within the top N samples sorted by ypred, considering first
    k samples in dataset (per each s0).
    """
    acc = []
    for s, ys in aggregate_s0(s0, y, ypred, k):
        acc.append(np.sum([yy[0] for yy in ys[:N]]) > 0)
    return np.mean(acc)


def mrr(s0, y, ypred):
    """
    Compute MRR (mean reciprocial rank) of y-predictions, by grouping
    y-predictions for the same s0 together.  This metric is relevant
    e.g. for the "answer sentence selection" task where we want to
    identify and take top N most relevant sentences.
    """
    rr = []
    for s, ys in aggregate_s0(s0, y, ypred):
        if np.sum([yy[0] for yy in ys]) == 0:
            continue  # do not include s0 with no right answers in MRR
        # to get rank, if we are in a larger cluster of same-scored sentences,
        # we must get |cluster|/2-ranked, not 1-ranked!
        # python3 -c 'import pysts.eval; import numpy as np; print(pysts.eval.mrr([np.array([0]),np.array([0]),np.array([0]),np.array([1]),np.array([1])], [1,0,0,1,1], [0.4,0.3,0.4,0.5,0.3]))'
        ysd = dict()
        for yy in ys:
            if yy[1] in ysd:
                ysd[yy[1]].append(yy[0])
            else:
                ysd[yy[1]] = [yy[0]]
        rank = 0
        for yp in sorted(ysd.keys(), reverse=True):
            if np.sum(ysd[yp]) > 0:
                rankofs = 1 - np.sum(ysd[yp]) / len(ysd[yp])
                rank += len(ysd[yp]) * rankofs
                break
            rank += len(ysd[yp])
        rr.append(1 / float(1+rank))

    return np.mean(rr)


def eval_sts(ycat, y, name, quiet=False):
    """ Evaluate given STS regression-classification predictions and print results. """
    if ycat.ndim == 1:
        ypred = ycat
    else:
        ypred = loader.sts_categorical2labels(ycat)
    if y.ndim == 1:
        ygold = y
    else:
        ygold = loader.sts_categorical2labels(y)
    pr = pearsonr(ypred, ygold)[0]
    if not quiet:
        print('%s Pearson: %f' % (name, pr,))
        print('%s Spearman: %f' % (name, spearmanr(ypred, ygold)[0],))
        print('%s MSE: %f' % (name, mse(ypred, ygold),))
    return pr


def eval_anssel(ypred, s0, y, name):
    rawacc, y0acc, y1acc, balacc = binclass_accuracy(y, ypred)
    mrr_ = mrr(s0, y, ypred)
    print('%s Accuracy: raw %f (y=0 %f, y=1 %f), bal %f' % (name, rawacc, y0acc, y1acc, balacc))
    print('%s MRR: %f  %s' % (name, mrr_, '(on training set, y=0 may be subsampled!)' if name == 'Train' else ''))
    return mrr_


def eval_ubuntu(ypred, s0, y, name):
    mrr_ = mrr(s0, y, ypred)
    r1_2 = recall_at(s0, y, ypred, N=1, k=2)
    r1_10 = recall_at(s0, y, ypred, N=1)
    r2_10 = recall_at(s0, y, ypred, N=2)
    r5_10 = recall_at(s0, y, ypred, N=5)
    print('%s MRR: %f' % (name, mrr_))
    print('%s 2-R@1: %f' % (name, r1_2))
    print('%s 10-R@1: %f  10-R@2: %f  10-R@5: %f' % (name, r1_10, r2_10, r5_10))
    return (mrr_, r1_2, r1_10, r2_10, r5_10)
