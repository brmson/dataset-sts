"""
Evaluation tools, mainly non-straightforward methods.
"""

from __future__ import print_function
from __future__ import division

import numpy as np


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


def mrr(s0, y, ypred):
    """
    Compute MRR (mean reciprocial rank) of y-predictions, by grouping
    y-predictions for the same s0 together.  This metric is relevant
    e.g. for the "answer sentence selection" task where we want to
    identify and take top N most relevant sentences.
    """
    ybys0 = dict()
    for i in range(len(s0)):
        if s0[i].tostring() in ybys0:
            ybys0[s0[i].tostring()].append((y[i], ypred[i]))
        else:
            ybys0[s0[i].tostring()] = [(y[i], ypred[i])]

    rr = []
    for s in ybys0.keys():
        ys = sorted(ybys0[s], key=lambda yy: yy[1], reverse=True)
        rank = -1
        for i in range(len(ys)):
            if ys[i][0] == 1:
                rank = i
                break
        if rank == -1:
            continue  # do not include s0 with no right answers in MRR
        rr.append(1 / float(1+rank))

    return np.mean(rr)
