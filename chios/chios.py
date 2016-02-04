from __future__ import print_function

import csv
from nltk.tokenize import word_tokenize
import numpy as np

import pysts.eval as ev


def load_chios(dsfile):
    """ load a dataset in the anssel csv format """
    s0 = []
    s1 = []
    labels = []
    # halabels,hqnlabels,hanlabels,mtext,mqnlabels,manlabels
    hlabels = []
    mlabels = []
    i = 0
    with open(dsfile) as f:
        c = csv.DictReader(f)
        for l in c:
            label = int(l['label'])
            labels.append(label)
            try:
                qtext = l['htext'].decode('utf8')
                atext = l['mtext'].decode('utf8')
            except AttributeError:  # python3 has no .decode()
                qtext = l['htext']
                atext = l['mtext']
            s0.append(word_tokenize(qtext))
            s1.append(word_tokenize(atext))
            hlab = [[int(tl) for tl in l['halabels'].split(' ')],
                    [int(tl) for tl in l['hqnlabels'].split(' ')],
                    [int(tl) for tl in l['hanlabels'].split(' ')]]
            mlab = [[0       for tl in l['mqnlabels'].split(' ')],
                    [int(tl) for tl in l['mqnlabels'].split(' ')],
                    [int(tl) for tl in l['manlabels'].split(' ')]]
            hlabels.append(np.array(hlab).T)
            mlabels.append(np.array(mlab).T)
            i += 1
    return (s0, s1, np.array(labels), np.array(hlabels), np.array(mlabels))


def eval_chios(ypred, s0, y, name):
    rawacc, y0acc, y1acc, balacc = ev.binclass_accuracy(y, ypred)
    mrr_ = ev.mrr(s0, y, ypred)
    print('%s Accuracy: raw %f (y=0 %f, y=1 %f), bal %f' % (name, rawacc, y0acc, y1acc, balacc))
    print('%s MRR: %f  %s' % (name, mrr_, '(on training set, y=0 is subsampled!)' if name == 'Train' else ''))
    return mrr_
