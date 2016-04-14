#!/usr/bin/python3
"""
The preprocessing of the SNLI dataset needs over 10GB and also takes some time.  To save memory and time, the script
goes through it (...several times) to preprocess it and dump it in
a way that can be loaded by training scripts efficiently.

Usage: tools/snli_preprocess.py [--revocab] TRAINFILE VALIDATIONFILE DUMPTRAINFILE DUMPVALIDATIONFILE VOCABFILE

Example: tools/snli_preprocess.py data/snli/snli_1.0_train.jsonl data/snli/snli_1.0_test.jsonl data/snli/snli_1.0_train.pickle data/snli/snli_1.0_test.pickle data/snli/v1-vocab.pickle
"""

import json
import pickle
import sys
import numpy as np
from pysts.vocab import Vocabulary
import pysts.loader as loader
from nltk.tokenize import word_tokenize
import pysts.nlp as nlp



TRAIN_LINES=550152
TEST_LINES=10000
spad=60

def sentence_gen(dsfiles):
    """ yield sentences from data files (train, validation) """
    i = 0
    for fname in dsfiles:
        with open(fname) as f:
            for l in f:
                d=json.loads(l)
                yield word_tokenize(d['sentence1'])
                yield word_tokenize(d['sentence2'])
                i += 1


def load_set(fname,vocab):
    s0, s1, labels = loader.load_snli(fname, vocab)
    si0 = vocab.vectorize(s0, spad)
    si1 = vocab.vectorize(s1, spad)
    f0, f1 = nlp.sentence_flags(s0, s1, spad, spad)
    return (si0, si1, f0, f1, labels)

def build_subset(input,output,sample_inds):
    linenum=0
    outputf=open(output,'w')
    for line in open(input):
        if linenum in sample_inds:
            outputf.write(line)
        if linenum%5000==0:
            print("%d lines processed" %(linenum))
        linenum+=1
    outputf.close()

def extract_subset(trainf, testf,fraction,train_out,test_out):
    # prepare train data subset
    fraction=float(fraction)
    print('Preparing train subset')
    samplenum=np.ceil(fraction*TRAIN_LINES)
    sample_inds=set(np.random.randint(TRAIN_LINES, size=samplenum))
    build_subset(trainf, train_out, sample_inds)
    # prepare test dataset
    print('Preparing test subset')
    samplenum=np.ceil(fraction*TEST_LINES)
    sample_inds=set(np.random.randint(TEST_LINES, size=samplenum))
    build_subset(testf, test_out, sample_inds)

if __name__ == "__main__":
    args = sys.argv[1:]
    if args[0] == '--revocab':
        revocab = True
        args = args[1:]
    else:
        revocab = False

    trainf, valf, dumptrainf, dumpvalf, vocabf = args

    if revocab:
        vocab = Vocabulary(sentence_gen([trainf]), count_thres=2)
        print('%d words' % (len(vocab.word_idx)))
        pickle.dump(vocab, open(vocabf, "wb"))
    else:
        vocab = pickle.load(open(vocabf, "rb"))
        print('%d words' % (len(vocab.word_idx)))

    print('Preprocessing train file')
    s0i, s1i, f0, f1, labels = load_set(trainf, vocab)
    pickle.dump((s0i, s1i, f0, f1, labels), open(dumptrainf, "wb"))

    print('Preprocessing validation file')
    s0i, s1i, f0, f1, labels = load_set(valf, vocab)
    pickle.dump((s0i, s1i, f0, f1, labels), open(dumpvalf, "wb"))

