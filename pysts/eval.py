"""
Evaluation tools, mainly non-straightforward methods.
"""

from __future__ import print_function
from __future__ import division

from collections import namedtuple
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

    # XXX could probably simplify and merge with above
    n_tp = np.sum(np.logical_and(ypred > 0.5, y > 0.5))
    n_fp = np.sum(np.logical_and(ypred > 0.5, y < 0.5))
    n_fn = np.sum(np.logical_and(ypred < 0.5, y > 0.5))
    prec = n_tp / (n_tp + n_fp)  # how many reported positives are ok
    recall = n_tp / (n_tp + n_fn)  # how many real positives we catch
    f_score = 2 * (prec * recall) / (prec + recall)

    return (rawacc, y0acc, y1acc, balacc, f_score)

def multiclass_accuracy(y, ypred):
    """
    Compute accuracy for multiclass classification tasks
    Returns (rawacc, class_acc) where rawacc is the accuracy on the whole set
    and class_acc contains accuracies on all classes respectively
    """
    result = np.zeros(ypred.shape)
    clss=y.shape[1]
    class_correct=np.zeros(clss)
    ok=0
    for row in range(ypred.shape[0]):
        result[row,np.argmax(ypred[row])]=1
        for cls in range(clss):
            if y[row,cls]==result[row,cls]:
                class_correct[cls]+=1
                if y[row,cls] == 1:
                    ok += 1
    class_acc=np.zeros(clss)
    for cls in range(clss):
        class_acc[cls]=(1.0*class_correct[cls])/y.shape[0]
    rawacc = (ok*1.0)/y.shape[0]
    return rawacc, class_acc

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


def trec_map(s0, s1, y, ypred):
    """ Use the official trec_eval tool to compute the mean average precision
    (MAP), a ranking measure that differs from MRR by taking into account also
    ranking of other than the top-ranked correct samples. """
    import subprocess
    import tempfile

    def save_trec_qrels(f, s0, s1, y):
        n = -1
        m = 0
        last_is0 = ''
        for is0, is1, iy in zip(s0, s1, y):
            if hash(tuple(is0)) != last_is0:
                last_is0 = hash(tuple(is0))
                m = 0
                n += 1
            print('%d 0 %d %d' % (n, m, iy), file=f)
            m += 1

    def save_trec_top(f, s0, s1, y, code):
        n = -1
        m = 0
        last_is0 = ''
        for is0, is1, iy in zip(s0, s1, y):
            if hash(tuple(is0)) != last_is0:
                last_is0 = hash(tuple(is0))
                m = 0
                n += 1
            print('%d 0 %d 1 %f %s' % (n, m, iy, code), file=f)
            m += 1

    def trec_eval_get(trec_qrels_file, trec_top_file, qty):
        p = subprocess.Popen('../trec_eval.8.1/trec_eval %s %s | grep %s | sed "s/.*\t//"' % (trec_qrels_file, trec_top_file, qty), stdout=subprocess.PIPE, shell=True)
        return float(p.communicate()[0])

    with tempfile.NamedTemporaryFile(mode="wt") as qrf:
        save_trec_qrels(qrf, s0, s1, y)
        qrf.flush()
        with tempfile.NamedTemporaryFile(mode="wt") as topf:
            save_trec_top(topf, s0, s1, ypred, '.')
            topf.flush()
            mapt = trec_eval_get(qrf.name, topf.name, 'map')
    return mapt


STSRes = namedtuple('STSRes', ['Pearson', 'Spearman', 'MSE'])


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
    sr = spearmanr(ypred, ygold)[0]
    e = mse(ypred, ygold)
    if not quiet:
        print('%s Pearson: %f' % (name, pr,))
        print('%s Spearman: %f' % (name, sr,))
        print('%s MSE: %f' % (name, e,))
    return STSRes(pr, sr, e)


AnsSelRes = namedtuple('AnsSelRes', ['MRR', 'MAP'])


def eval_anssel(ypred, s0, s1, y, name, MAP=False):
    rawacc, y0acc, y1acc, balacc, f_score = binclass_accuracy(y, ypred)
    mrr_ = mrr(s0, y, ypred)
    print('%s Accuracy: raw %f (y=0 %f, y=1 %f), bal %f' % (name, rawacc, y0acc, y1acc, balacc))
    print('%s MRR: %f  %s' % (name, mrr_, '(on training set, y=0 may be subsampled!)' if name == 'Train' else ''))
    if MAP:
        map_ = trec_map(s0, s1, y, ypred)
        print('%s MAP: %f' % (name, map_))
    else:
        map_ = None
    return AnsSelRes(mrr_, map_)


ParaRes = namedtuple('ParaRes', ['Accuracy', 'F1'])


def eval_para(ypred, y, name):
    rawacc, y0acc, y1acc, balacc, f_score = binclass_accuracy(y, ypred)
    print('%s Accuracy: raw %f (y=0 %f, y=1 %f), bal %f;  F-Score: %f' % (name, rawacc, y0acc, y1acc, balacc, f_score))
    return ParaRes(rawacc, f_score)


HypEvRes = namedtuple('HypEvRes', ['QAccuracy', 'QF1'])
AbcdRes = namedtuple('ABCDRes', ['AbcdAccuracy', 'AbcdMRR'])


def eval_hypev(qids, ypred, y, name):
    if qids is None:
        rawacc, y0acc, y1acc, balacc, f_score = binclass_accuracy(y, ypred)
        print('%s QAccuracy: real %f  (y=0 %f, y=1 %f, bal %f);  F-Score: %f' % (name, rawacc, y0acc, y1acc, balacc, f_score))
        return HypEvRes(rawacc, f_score)
    else:
        rawacc = recall_at(qids, y, ypred, N=1)
        mrr_ = mrr(qids, y, ypred)
        print('%s AbcdAccuracy: %f;  MRR: %f' % (name, rawacc, mrr_))
        return AbcdRes(rawacc, mrr_)


UbuntuRes = namedtuple('UbuntuRes', ['MRR', 'R2_1', 'R10_1', 'R10_2', 'R10_5'])


def eval_ubuntu(ypred, s0, y, name):
    mrr_ = mrr(s0, y, ypred)
    r1_2 = recall_at(s0, y, ypred, N=1, k=2)
    r1_10 = recall_at(s0, y, ypred, N=1)
    r2_10 = recall_at(s0, y, ypred, N=2)
    r5_10 = recall_at(s0, y, ypred, N=5)
    print('%s MRR: %f' % (name, mrr_))
    print('%s 2-R@1: %f' % (name, r1_2))
    print('%s 10-R@1: %f  10-R@2: %f  10-R@5: %f' % (name, r1_10, r2_10, r5_10))
    return UbuntuRes(mrr_, r1_2, r1_10, r2_10, r5_10)


RTERes = namedtuple('RTERes', ['Accuracy'])


def eval_rte(ypred, y, name):
    cls_names = ['contradiction', 'neutral', 'entailment']
    rawacc, cls_acc = multiclass_accuracy(y, ypred)
    print('%s Accuracy: %.3f, %s accuracy %.3f, %s accuracy %.3f, %s accuracy %.3f' % (name, rawacc,
                                                                                        cls_names[0], cls_acc[0],
                                                                                        cls_names[1], cls_acc[1],
                                                                                        cls_names[2], cls_acc[2]))
    return RTERes(rawacc)
