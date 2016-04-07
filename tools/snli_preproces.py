import json
import pickle
import sys
import numpy as np
from pysts.vocab import Vocabulary
from nltk.tokenize import word_tokenize


TRAIN_LINES=550152
TEST_LINES=10000

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
    testset, valset, vocabf = sys.argv[1:4]
    vocab = Vocabulary(sentence_gen([testset,valset]), count_thres=2)
    pickle.dump(vocab, open(vocabf, "wb"))
