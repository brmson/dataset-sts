"""
This module provides a variety of loaders for our datasets.

A longer term goal is to unify the dataset formats to reduce the variety
of datasets needed, but we will always have some inherent variety, e.g. you
may want to load the sick2014 dataset from sts (relevancy) or entailment
(or contradiction) perspective.
"""


from __future__ import print_function

import codecs
import csv
from nltk.tokenize import word_tokenize
import numpy as np


def load_anssel(dsfile, subsample0=1, skip_oneclass=True):
    """ load a dataset in the anssel csv format;

    subsample0=N denotes that only every N-th 0-labelled sample
    should be loaded; so e.g. N=3 reduces 80k negatives to 28k
    negatives in the training set (vs. 4k positives); N=10k
    gets you just 8k negatives, etc.

    skip_oneclass=True denotes that samples with s0 that is associated
    only with 0-labelled or 1-labelled samples are skipped (i.e. questions
    with only positives/negatives are skipped) """
    s0 = []
    s1 = []
    labels = []
    toklabels = []
    i = 0

    s0blacklist = set()
    if skip_oneclass:
        # A separate pass that builds a blacklist of s0 to skip
        with open(dsfile) as f:
            c = csv.DictReader(f)
            s0l, npos, nneg = ('', 0, 0)
            for l in c:
                try:
                    qtext = l['qtext'].decode('utf8')
                except AttributeError:  # python3 has no .decode()
                    qtext = l['qtext']
                if s0l == '':
                    s0l = qtext
                elif s0l != qtext:
                    if npos == 0 or nneg == 0:
                        s0blacklist.add(s0l)
                    s0l, npos, nneg = (qtext, 0, 0)
                label = int(l['label'])
                if label == 1:
                    npos += 1
                else:
                    nneg += 1
            if npos == 0 or nneg == 0:
                s0blacklist.add(s0l)

    with open(dsfile) as f:
        c = csv.DictReader(f)
        for l in c:
            label = int(l['label'])
            if label == 0 and (i % subsample0) != 0:
                i += 1
                continue
            try:
                qtext = l['qtext'].decode('utf8')
                atext = l['atext'].decode('utf8')
            except AttributeError:  # python3 has no .decode()
                qtext = l['qtext']
                atext = l['atext']
            if qtext in s0blacklist:
                continue
            labels.append(label)
            s0.append(qtext.split(' '))
            s1.append(atext.split(' '))
            if 'toklabels' in l:
                toklabels.append([int(tl) for tl in l['toklabels'].split(' ')])
            i += 1
    return (s0, s1, np.array(labels), toklabels if toklabels else None)


def load_sick2014(dsfile, mode='relatedness'):
    """ load a dataset in the sick2014 tsv .txt format;

    mode='relatedness': use the sts relatedness score as label
    mode='entailment': use -1 (contr.), 0 (neutral), 1 (ent.) as label """
    s0 = []
    s1 = []
    labels = []
    with open(dsfile) as f:
        first = True
        for line in f:
            if first:
                # skip first line with header
                first = False
                continue
            line = line.rstrip()
            pair_ID, sentence_A, sentence_B, relatedness_score, entailment_judgement = line.split('\t')
            if mode == 'relatedness':
                label = float(relatedness_score)
            elif mode == 'entailment':
                if entailment_judgement == 'CONTRADICTION':
                    label = -1
                elif entailment_judgement == 'NEUTRAL':
                    label = 0
                elif entailment_judgement == 'ENTAILMENT':
                    label = +1
                else:
                    raise ValueError('invalid label on line: %s' % (line,))
            else:
                raise ValueError('invalid mode: %s' % (mode,))
            labels.append(label)
            s0.append(word_tokenize(sentence_A))
            s1.append(word_tokenize(sentence_B))
    return (s0, s1, np.array(labels))


def load_sts(dsfile, skip_unlabeled=True):
    """ load a dataset in the sts tsv format """
    s0 = []
    s1 = []
    labels = []
    with codecs.open(dsfile, encoding='utf8') as f:
        for line in f:
            line = line.rstrip()
            label, s0x, s1x = line.split('\t')
            if label == '':
                if skip_unlabeled:
                    continue
                else:
                    labels.append(-1.)
            else:
                labels.append(float(label))
            s0.append(word_tokenize(s0x))
            s1.append(word_tokenize(s1x))
    return (s0, s1, np.array(labels))


def concat_datasets(datasets):
    """ Concatenate multiple loaded datasets into a single large one.

    Example: s0, s1, lab = concat_datasets([load_sts(d) for glob.glob('data/sts/semeval-sts/all/201[0-4]*')]) """
    s0 = []
    s1 = []
    labels = []
    for s0x, s1x, labelsx in datasets:
        s0 += s0x
        s1 += s1x
        labels += list(labelsx)
    return (s0, s1, np.array(labels))


def balance_dataset(ds):
    """ Supersample 1-labelled items to achieve a balanced dataset
    with random classifier giving 50%.

    This makes sense only for datasets with crisp 0/1 labels! """
    # FIXME: we assume 1-labelled < 0-labelled
    y = ds[2]
    class1 = np.where(y == 1)[0]
    n_imbal = np.sum(y == 0) - np.sum(y == 1)

    s0 = list(ds[0])
    s1 = list(ds[1])
    labels = list(ds[2])
    has_toklabels = len(ds) > 3
    if has_toklabels:
        toklabels = list(ds[3]) if ds[3] is not None else None
    for i in np.random.choice(class1, size=n_imbal):
        s0.append(ds[0][i])
        s1.append(ds[1][i])
        labels.append(ds[2][i])
        if has_toklabels and toklabels is not None:
            toklabels.append(ds[3][i])
    if has_toklabels:
        return (s0, s1, np.array(labels), toklabels)
    else:
        return (s0, s1, np.array(labels))


def load_embedded(glove, s0, s1, labels, balance=False, ndim=1, s0pad=25, s1pad=60):
    """ Post-process loaded (s0, s1, labels) by mapping it to embeddings,
    plus optionally balancing (if labels are binary) and optionally not
    averaging but padding and returning full-sequence matrices.

    Note that this is now deprecated, especially if you use Keras - use the
    vocab.Vocabulary class. """

    if balance:
        s0, s1, labels = balance_dataset((s0, s1, labels))

    if ndim == 1:
        # for averaging:
        e0 = np.array(glove.map_set(s0, ndim=1))
        e1 = np.array(glove.map_set(s1, ndim=1))
    else:
        # for padding and sequences (e.g. keras RNNs):
        # print('(%s) s0[-1000]: %d tokens' % (globmask, np.sort([np.shape(s) for s in s0], axis=0)[-1000]))
        # print('(%s) s1[-1000]: %d tokens' % (globmask, np.sort([np.shape(s) for s in s1], axis=0)[-1000]))
        e0 = glove.pad_set(glove.map_set(s0), s0pad)
        e1 = glove.pad_set(glove.map_set(s1), s1pad)
    return (e0, e1, s0, s1, labels)


def sts_labels2categorical(labels, nclass=6):
    """
    From continuous labels in [0,5], generate 5D binary-ish vectors.
    This enables us to do classification instead of regression.
    (e.g. sigmoid output would be troublesome with the original labeling)

    Label encoding from Tree LSTM paper (Tai, Socher, Manning)

    (Based on https://github.com/ryankiros/skip-thoughts/blob/master/eval_sick.py)
    """
    Y = np.zeros((len(labels), nclass))
    for j, y in enumerate(labels):
        if np.floor(y) + 1 < nclass:
            Y[j, int(np.floor(y)) + 1] = y - np.floor(y)
        Y[j, int(np.floor(y))] = np.floor(y) - y + 1
    return Y


def sts_categorical2labels(Y, nclass=6):
    """
    From categorical score encoding, reconstruct continuous labels.
    This is useful to convert classifier output to something we can
    correlate with gold standard again.
    """
    r = np.arange(nclass)
    return np.dot(Y, r)
