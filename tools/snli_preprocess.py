#!/usr/bin/python3
"""
The preprocessing of the SNLI dataset needs over 10GB and also takes some time.  To save memory and time, the script
goes through it (...several times) to preprocess it and dump it in
a way that can be loaded by training scripts efficiently.
Usage: tools/snli_preprocess.py [--revocab] TRAINFILE VALIDATIONFILE TESTFILE DUMPTRAINFILE DUMPVALIDATIONFILE DUMPTESTFILE VOCABFILE
Example: tools/snli_preprocess.py data/snli/snli_1.0_train.jsonl data/snli/snli_1.0_dev.jsonl data/snli/snli_1.0_test.jsonl data/snli/snli_1.0_train.pickle data/snli/snli_1.0_dev.pickle data/snli/snli_1.0_test.pickle data/snli/v1-vocab.pickle
"""

import json
import pickle
import sys
import pysts.embedding as emb
from pysts.vocab import Vocabulary
import pysts.loader as loader
from nltk.tokenize import word_tokenize
import pysts.nlp as nlp


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


def load_set(fname,vocab,glove):
    s0, s1, labels = loader.load_snli(fname, vocab)
    si0,sj0 = vocab.vectorize(s0, glove, spad)
    si1,sj1 = vocab.vectorize(s1, glove, spad)
    f0_, f1_ = nlp.sentence_flags(s0, s1, spad, spad)
    return (si0, si1, sj0, sj1, f0_, f1_, labels)



if __name__ == "__main__":
    args = sys.argv[1:]
    if args[0] == '--revocab':
        revocab = True
        args = args[1:]
    else:
        revocab = False

    trainf, valf, testf, dumptrainf, dumpvalf, dumptestf, vocabf = args

    if revocab:
        vocab = Vocabulary(sentence_gen([trainf]), count_thres=2)
        print('%d words' % (len(vocab.word_idx)))
        pickle.dump(vocab, open(vocabf, "wb"))
    else:
        vocab = pickle.load(open(vocabf, "rb"))
        print('%d words' % (len(vocab.word_idx)))

    glove = emb.GloVe(N=300)  # XXX: hardcoded

    print('Preprocessing train file')
    si0, si1, sj0, sj1, f0_, f1_, labels = load_set(trainf, vocab, glove)
    pickle.dump((si0, si1, sj0, sj1, f0_, f1_, labels), open(dumptrainf, "wb"))

    print('Preprocessing validation file')
    si0, si1, sj0, sj1, f0_, f1_, labels = load_set(valf, vocab, glove)
    pickle.dump((si0, si1, sj0, sj1, f0_, f1_, labels), open(dumpvalf, "wb"))

    print('Preprocessing test file')
    si0, si1, sj0, sj1, f0_, f1_, labels = load_set(testf, vocab, glove)
    pickle.dump((si0, si1, sj0, sj1, f0_, f1_, labels), open(dumptestf, "wb"))

