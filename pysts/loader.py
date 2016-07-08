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
import gzip
from nltk.tokenize import word_tokenize
import numpy as np
import json


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
    kwweights = []
    aboutkwweights = []
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
            if 'kwweight' in l:
                kwweights.append([float(l['kwweight'])])
            if 'aboutkwweight' in l:
                aboutkwweights.append([float(l['aboutkwweight'])])
            if 'toklabels' in l:
                toklabels.append([int(tl) for tl in l['toklabels'].split(' ')])
            i += 1
    return (s0, s1, np.array(labels),
            np.array(kwweights) if kwweights else None,
            np.array(aboutkwweights) if aboutkwweights else None,
            toklabels if toklabels else None)


def load_hypev(dsfile):
    """ load a dataset in the hypev csv format;

    TODO: the (optional) semantic token labels (linked entities etc.)
    are not loaded. """
    s0 = []
    s1 = []
    labels = []
    qids = []

    with open(dsfile) as f:
        c = csv.DictReader(f)
        for l in c:
            label = int(l['label'])
            try:
                htext = l['htext'].decode('utf8')
                mtext = l['mtext'].decode('utf8')
                if 'qid' in l:
                    qids.append(l['qid'].decode('utf8'))
            except AttributeError:  # python3 has no .decode()
                htext = l['htext']
                mtext = l['mtext']
                if 'qid' in l:
                    qids.append(l['qid'])
            labels.append(label)
            s0.append(htext.split(' '))
            s1.append(mtext.split(' '))
    return (s0, s1, np.array(labels), qids if qids else None)


hypev_xtra_c = [
    '#Question sentiment',
    '#Sentence sentiment',
    '#@Subject match',
    '#@Object match',
    '#@Verb similarity (spacy)',
    '#@Verb similarity (WordNet)',
    '#Match score',
    '#@Antonyms',
    '#@Verb similarity (WordNetBinary)',
]
hypev_xtra_r = [
    '#@Subject match',
    '#@Object match',
    '#@Verb similarity (spacy)',
    '#@Verb similarity (WordNet)',
    '@Date relevance',
    '@Elastic score',
    '#@Antonyms',
    '#@Verb similarity (WordNetBinary)',
]

def load_hypev_xtra(rows):
    """ load an auxiliary feature dataset in the argus format.
    This dataset contains a vector of extra features per each
    hypothesis pair, which can then be appended for training.

    Normally:
    dsfile = re.sub('\.([^.]*)$', '_aux.\1', basename)  # train.tsv -> train_aux.tsv
    with open(dsfile) as f:
        c = csv.DictReader(f, delimiter='\t')
        xtra = load_hypev_xtra(c)
    """
    xtra = {'#': [], '@': []}
    for l in rows:
        if l.get('Class_GS', None) == 'Class_GS':
            continue
        # TODO: ==0 features
        xtra1 = {'#': np.zeros(len(hypev_xtra_c)), '@': np.zeros(len(hypev_xtra_r))}
        for k, v in l.items():
            if '#' in k:
                xtra1['#'][hypev_xtra_c.index(k)] = v
            elif '@' in k:
                xtra1['@'][hypev_xtra_r.index(k)] = v
        xtra['#'].append(xtra1['#'])
        xtra['@'].append(xtra1['@'])
    xtra['#'] = np.array(xtra['#'])
    xtra['@'] = np.array(xtra['@'])
    return xtra


def load_mctest(basename):
    """ load a dataset in the MCTest format - pair of .statements.tsv and .ans
    files with the given basename stem. """
    s0 = []
    s1 = []
    labels = []
    qids = []
    types = []

    tsvf = open(basename + '.statements.tsv')
    ansf = open(basename + '.ans')

    tsvcol = ['qid', 'comment', 'story',
              'q0', 'htext0A', 'htext0B', 'htext0C', 'htext0D',
              'q1', 'htext1A', 'htext1B', 'htext1C', 'htext1D',
              'q2', 'htext2A', 'htext2B', 'htext2C', 'htext2D',
              'q3', 'htext3A', 'htext3B', 'htext3C', 'htext3D']

    n_stories = 0
    n_hyp = 0
    for tsvl, ansl in zip(tsvf, ansf):
        data = dict(zip(tsvcol, tsvl.split('\t')))
        ansdata = ansl.rstrip().split('\t')

        storytext = data['story'].replace('\\newline', '\n')
        story = [(word_tokenize(s) + ['.']) for s in storytext.split('.')]
        n_stories += 1

        for i, ans in enumerate(ansdata):
            qtype = data['q%d' % (i,)].split(':')[0]
            for letter in ['A', 'B', 'C', 'D']:
                n_hyp += 1
                htext = word_tokenize(data['htext%d%s' % (i, letter)])
                for mtext in story:
                    s0.append(htext)
                    s1.append(mtext)
                    qids.append('%s_%d' % (data['qid'], i))
                    labels.append(1. if ans == letter else 0.)
                    types.append(qtype)

    print('Loaded %d stories, %d hypotheses' % (n_stories, n_hyp))
    return (s0, s1, np.array(labels), qids, types)


rte_lmappings = {'contradiction': np.array([1,0,0]), 'neutral': np.array([0,1,0]), 'entailment': np.array([0,0,1])}

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
                if entailment_judgement.lower() in rte_lmappings:
                    label = rte_lmappings[entailment_judgement.lower()]
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

def load_snli(dsfile, vocab):
    '''
    Loads the dataset in json format.
    Note that a sentence pair is not loaded if its label is not in the known labels, this happens
    in case when the label is "-", which implies that less then 3 of 5 annotators agreed on one label for the pair
    during dataset validation.
    '''
    s0i = []
    s1i = []
    labels = []
    i = 0
    skips=0
    with open(dsfile) as f:
        for l in f:
            d=json.loads(l)
            if i % 5000 == 0:
                print('%d samples read, %d no label skips' % (i,skips))
            label = d['gold_label']
            if label in rte_lmappings:
                s0 = word_tokenize(d['sentence1'])
                s1 = word_tokenize(d['sentence2'])
                s0i.append(s0)
                s1i.append(s1)
                labels.append(rte_lmappings[label])
            else:
                skips+=1
            i += 1
    print('%s dataset file loaded. %d samples read, %d no label skips' % (dsfile,i,skips))
    return (s0i, s1i, np.array(labels))


def load_msrpara(dsfile):
    """ load a dataset in the msrpara tsv format """
    s0 = []
    s1 = []
    labels = []
    with codecs.open(dsfile, encoding='utf8') as f:
        firstline = True
        for line in f:
            if firstline:
                firstline = False
                continue
            line = line.rstrip()
            label, s0id, s1id, s0x, s1x = line.split('\t')
            labels.append(float(label))
            s0.append(word_tokenize(s0x))
            s1.append(word_tokenize(s1x))
    return (s0, s1, np.array(labels))


def load_askubuntu_texts(tfile):
    # https://github.com/taolei87/rcnn/blob/master/code/qa/myio.py
    empty_cnt = 0
    texts = {}
    fopen = gzip.open if tfile.endswith(".gz") else open
    with fopen(tfile) as fin:
        for line in fin:
            id, title, body = line.split("\t")
            if len(title) == 0:
                continue
            title = title.strip().split()
            body = body.strip().split()
            texts[id] = title  #(title, body)
    return texts


def load_askubuntu_q(qfile):
    # https://github.com/taolei87/rcnn/blob/master/code/qa/myio.py
    links = []
    with open(qfile) as fin:
        for line in fin:
            parts = line.split("\t")
            pid, pos, neg = parts[:3]
            pos = pos.split()
            neg = neg.split()
            if len(pos) == 0: continue
            s = set()
            qids = [ ]
            qlabels = [ ]
            for q in neg:
                if q not in s:
                    qids.append(q)
                    qlabels.append(0 if q not in pos else 1)
                    s.add(q)
            for q in pos:
                if q not in s:
                    qids.append(q)
                    qlabels.append(1)
                    s.add(q)
            links.append((pid, qids, qlabels))
    return links



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


def graph_adapt_ubuntu(gr, vocab):
    """ modify an anssel graph dataset to look like ubuntu's; XXX move elsewhere? """
    gr2 = dict(gr)
    for k in ['si0', 'si1']:
        gr2[k] = []
        for s in gr[k]:
            s2 = list(s)
            try:
                s2[s2.index(0)] = vocab.word_idx['__eou__']
                if k == 'si0':
                    s2[s2.index(0)] = vocab.word_idx['__eot__']
            except ValueError:
                pass
            gr2[k].append(s2)
        gr2[k] = np.array(gr2[k])
    return gr2
