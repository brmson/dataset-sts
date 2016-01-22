"""
This module provides a variety of loaders for our datasets.

A longer term goal is to unify the dataset formats to reduce the variety
of datasets needed, but we will always have some inherent variety, e.g. you
may want to load the sick2014 dataset from sts (relevancy) or entailment
(or contradiction) perspective.
"""


from __future__ import print_function

import csv
from nltk.tokenize import word_tokenize
import numpy as np


def load_anssel(dsfile, subsample0=3):
    """ load a dataset in the anssel csv format;

    subsample0=N denotes that only every N-th 0-labelled sample
    should be loaded; so e.g. N=3 reduces 80k negatives to 28k
    negatives in the training set (vs. 4k positives); N=10k
    gets you just 8k negatives, etc. """
    s0 = []
    s1 = []
    labels = []
    i = 0
    with open(dsfile) as f:
        c = csv.DictReader(f)
        for l in c:
            label = int(l['label'])
            if label == 0 and (i % subsample0) != 0:
                i += 1
                continue
            labels.append(label)
            try:
                qtext = l['qtext'].decode('utf8')
                atext = l['atext'].decode('utf8')
            except AttributeError:  # python3 has no .decode()
                qtext = l['qtext']
                atext = l['atext']
            s0.append(word_tokenize(qtext))
            s1.append(word_tokenize(atext))
            i += 1
    return (s0, s1, np.array(labels))


def load_sick2014(dsfile, mode='relatedness'):
    """ load a dataset in the sick2014 tsv .txt format, with labels normalized to [0,1];

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
                label = float(relatedness_score) / 5
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


def load_sts(dsfile):
    """ load a dataset in the sts tsv format, with labels normalized to [0,1] """
    s0 = []
    s1 = []
    labels = []
    with open(dsfile) as f:
        for line in f:
            line = line.rstrip()
            label, s0x, s1x = line.split('\t')
            if label == '':
                continue  # some pairs are unlabeled, skip
            labels.append(float(label) / 5)
            s0.append(word_tokenize(s0x))
            s1.append(word_tokenize(s1x))
    return (s0, s1, np.array(labels))


def concat_datasets(datasets):
    """ Concatenate multiple loaded datasets into a single large one.

    Example: s0, s1, lab = concat_datasets([load_sts(d) for glob.glob('sts/all/201[0-4]*')]) """
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
    for i in np.random.choice(class1, size=n_imbal):
        s0.append(ds[0][i])
        s1.append(ds[1][i])
        labels.append(ds[2][i])
    return (s0, s1, np.array(labels))
