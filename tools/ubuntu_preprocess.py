#!/usr/bin/python3
"""
The Ubuntu dataset is pretty large.  To conserve memory, this script
goes through it (...several times) to preprocess it and dump it in
a way that can be loaded by training scripts efficiently.

Usage: tools/ubuntu_preprocess.py [--revocab] DATAFILE DUMPFILE VOCABFILE

Example: tools/ubuntu_preprocess.py data/anssel/ubuntu/v1-trainset.csv data/anssel/ubuntu/v1-trainset.pickle data/anssel/ubuntu/v1-vocab.pickle
"""

from __future__ import print_function
from __future__ import division

import csv
import pickle
import sys

import pysts.embedding as emb
import pysts.eval as ev
import pysts.loader as loader
import pysts.nlp as nlp
from pysts.vocab import Vocabulary


MAX_SAMPLES = 1000000


def sentence_gen(dsfile):
    """ yield sentences from data file """
    i = 0
    with open(dsfile) as f:
        c = csv.reader(f, delimiter=',')
        for qtext, atext, label in c:
            if i % 10000 == 0:
                print('%d samples' % (i,))
            try:
                qtext = qtext.decode('utf8')
                atext = atext.decode('utf8')
            except AttributeError:  # python3 has no .decode()
                qtext = qtext
                atext = atext
            yield qtext.replace('</s>', '__EOS__').split(' ')
            yield atext.replace('</s>', '__EOS__').split(' ')
            i += 1
            if i > MAX_SAMPLES:
                break


def load_set(dsfile, vocab, emb):
    s0i = []
    s1i = []
    s0j = []
    s1j = []
    f0 = []
    f1 = []
    labels = []

    i = 0
    with open(dsfile) as f:
        c = csv.reader(f, delimiter=',')
        for qtext, atext, label in c:
            if i % 10000 == 0:
                print('%d samples' % (i,))
            try:
                qtext = qtext.decode('utf8')
                atext = atext.decode('utf8')
            except AttributeError:  # python3 has no .decode()
                qtext = qtext
                atext = atext
            s0 = qtext.replace('</s>', '__EOS__').split(' ')
            s1 = atext.replace('</s>', '__EOS__').split(' ')
            si0, sj0 = vocab.vectorize([s0], emb, spad=None)
            si1, sj1 = vocab.vectorize([s1], emb, spad=None)
            f0_, f1_ = nlp.sentence_flags([s0], [s1], len(s0), len(s1))

            s0i.append(si0[0])
            s1i.append(si1[0])
            s0j.append(sj0[0])
            s1j.append(sj1[0])
            f0.append(f0_[0])
            f1.append(f1_[0])
            labels.append(int(label))
            i += 1
            if i > MAX_SAMPLES:
                break

    return (s0i, s1i, s0j, s1j, f0, f1, labels)


if __name__ == "__main__":
    args = sys.argv[1:]
    if args[0] == '--revocab':
        revocab = True
        args = args[1:]
    else:
        revocab = False

    dataf, dumpf, vocabf = args

    glove = emb.GloVe(N=300)  # XXX: hardcoded

    if revocab:
        vocab = Vocabulary(sentence_gen(dataf), count_thres=2, prune_N=100)
        print('%d words' % (len(vocab.word_idx)))
        pickle.dump(vocab, open(vocabf, "wb"))
    else:
        vocab = pickle.load(open(vocabf, "rb"))
        print('%d words' % (len(vocab.word_idx)))

    s0i, s1i, s0j, s1j, f0, f1, labels = load_set(dataf, vocab, glove)
    pickle.dump((s0i, s1i, s0j, s1j, f0, f1, labels), open(dumpf, "wb"))

    # glove = emb.GloVe(N=300)
