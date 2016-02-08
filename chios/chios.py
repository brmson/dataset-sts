from __future__ import print_function, division

import csv
from collections import defaultdict, namedtuple
from nltk.tokenize import word_tokenize
from keras.callbacks import Callback
import numpy as np
import random

import pysts.eval as ev
import pysts.loader as loader


Pair = namedtuple('Pair', ['qid', 's0', 's1', 'l', 'l0', 'l1'])


def load_chios(dsfile, origbit):
    """ load a dataset in the anssel csv format """
    pairs = []
    i = 0
    with open(dsfile) as f:
        c = csv.DictReader(f)
        for l in c:
            label = int(l['label'])
            try:
                qtext = l['htext'].decode('utf8')
                atext = l['mtext'].decode('utf8')
            except AttributeError:  # python3 has no .decode()
                qtext = l['htext']
                atext = l['mtext']
            s0 = word_tokenize(qtext)
            s1 = word_tokenize(atext)
            hlab = [[int(tl) for tl in l['halabels'].split(' ')],
                    [int(tl) for tl in l['hqnlabels'].split(' ')],
                    [int(tl) for tl in l['hanlabels'].split(' ')],
                    [origbit for tl in l['hanlabels'].split(' ')]]
            mlab = [[0       for tl in l['mqnlabels'].split(' ')],
                    [int(tl) for tl in l['mqnlabels'].split(' ')],
                    [int(tl) for tl in l['manlabels'].split(' ')],
                    [origbit for tl in l['manlabels'].split(' ')]]
            l0 = np.array(hlab).T
            l1 = np.array(mlab).T
            pairs.append(Pair(l['qid'], s0, s1, label, l0, l1))
            i += 1
    return pairs


def sample_questions(glove, pairs, embpar=None, B=False, skip_long=True, once=False, gen_classes=False, ret_qpair=False):
    questions = defaultdict(list)
    for p in pairs:
        questions[p.qid].append(p)
    qids = list(questions.keys())

    if embpar is None:
        embpar = dict()

    while True:
        random.shuffle(qids)
        for qid in qids:
            qpairs = questions[qid]
            if skip_long and embpar.get('ndim', 1) == 2:
                qpairs = [p for p in qpairs if len(p.s0) <= embpar['s0pad'] and len(p.s1) <= embpar['s1pad']]
                if not qpairs:
                    continue
            s0 = [p.s0 for p in qpairs]
            s1 = [p.s1 for p in qpairs]
            labels = np.array([p.l for p in qpairs])
            if np.all(labels < 1) or np.all(labels > 0):
                # no or all right answers, skip!
                continue
            e0, e1, s0, s1, labels = loader.load_embedded(glove, s0, s1, labels, **embpar)
            if embpar.get('ndim', 1) == 2:
                l0 = glove.pad_set([p.l0 for p in qpairs], embpar['s0pad'], N=4)
                l1 = glove.pad_set([p.l1 for p in qpairs], embpar['s1pad'], N=4)
                e0 = np.dstack((e0, l0))
                e1 = np.dstack((e1, l1))
            data = {'e0': e0, 'e1': e1, 'score': labels}
            if B:
                data['e0B'] = e0
                data['e1B'] = e1
            if gen_classes:
                data['s0h'] = [hash(tuple(s)) for s in s0]
            if ret_qpair:
                data['p'] = qpairs
            yield data
        if once:
            return


global last_val_stats
last_val_stats = (0,0,0,0,0)

class SampleValCB(Callback):
    def __init__(self, glove, ptest, embpar=None, B=False):
        self.glove = glove
        self.ptest = ptest
        self.embpar = embpar
	self.B = B

    def on_epoch_end(self, epoch, logs={}):
        mtloss = np.mean([self.model.test_on_batch(data) for data in sample_questions(self.glove, self.ptest, embpar=self.embpar, B=self.B, skip_long=False, once=True)])
        n = 0
        top_mrr = 0
        top_acc1 = 0
        sums_mrr = 0
        sums_acc1 = 0
        for data in sample_questions(self.glove, self.ptest, embpar=self.embpar, B=self.B, skip_long=False, once=True, gen_classes=True):
            pdata = dict(data)
            pdata.pop('s0h')
            pred = self.model.predict_on_batch(pdata)['score'][:, 0]

            sums = defaultdict(list)
            sums_trueh = None
            rank = 1
            for i, k in sorted(enumerate(pred), key=lambda e: e[1], reverse=True):
                sums[data['s0h'][i]].append(k)
                if sums_trueh is None and data['score'][i] > 0.5:
                    sums_trueh = data['s0h'][i]
                    if rank == 1:
                        top_acc1 += 1
                    top_mrr += 1/rank
                rank += 1

            rank = 1
            for h, k in sorted(sums.items(), key=lambda e: np.mean(e[1]), reverse=True):
                if h == sums_trueh:
                    if rank == 1:
                        sums_acc1 += 1
                    sums_mrr += 1/rank
                rank += 1

            n += 1
        top_mrr /= n
        top_acc1 /= n
        sums_mrr /= n
        sums_acc1 /= n
        global last_val_stats
        last_val_stats = (mtloss, top_mrr, top_acc1, sums_mrr, sums_acc1)
        print('       valloss: %.4f valtop(mrr: %.4f acc@1: %.4f) valsum(mrr: %.4f acc@1: %.4f)' % (mtloss, top_mrr, top_acc1, sums_mrr, sums_acc1))


def eval_chios(ypred, s0, y, name):
    rawacc, y0acc, y1acc, balacc = ev.binclass_accuracy(y, ypred)
    mrr_ = ev.mrr(s0, y, ypred)
    print('%s Accuracy: raw %f (y=0 %f, y=1 %f), bal %f' % (name, rawacc, y0acc, y1acc, balacc))
    print('%s MRR: %f  %s' % (name, mrr_, '(on training set, y=0 is subsampled!)' if name == 'Train' else ''))
    return mrr_
