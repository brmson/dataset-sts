#!/usr/bin/python3
# Universal dataset preprocessing by performing segmentation and rewriting
# the dataset to v1 format.
# Example: preprocess.py ubuntu-ranking-dataset-creator/src/train.csv v2-trainset.csv train
# (the last argument can be either "train" or "test", denotes the CSV style)
#
# TODO: We may want to add further substitutions of entities and paths, or even
# better token decorations for such tokens in the pysts.nlp and anssel/yodaqa
# style.

import unicodecsv
import re
import sys

from tweetmotif.twokenize import tokenize


def str2tok(string):
    return tokenize(string)


def produce(outcsv, ctx, followup, label):
    return outcsv.writerow([' '.join(str2tok(ctx)), ' '.join(str2tok(followup)), label])


def transcript(incsv, outcsv, mode):
    for row in incsv:
        if mode == 'train':
            # Context,Utterance,Label
            produce(outcsv, row['Context'], row['Utterance'], int(float(row['Label'])))
        else:
            # Context,Ground Truth Utterance,Distractor_0,Distractor_1,Distractor_2,Distractor_3,Distractor_4,Distractor_5,Distractor_6,Distractor_7,Distractor_8
            produce(outcsv, row['Context'], row['Ground Truth Utterance'], 1)
            for d in range(9):
                produce(outcsv, row['Context'], row['Distractor_%d'%(d,)], 0)


if __name__ == "__main__":
    rawfile = sys.argv[1]
    postfile = sys.argv[2]
    mode = sys.argv[3]

    inf = open(rawfile, 'r')
    outf = open(postfile, 'w')

    incsv = unicodecsv.DictReader(inf, encoding='utf-8')
    outcsv = unicodecsv.writer(outf, encoding='utf-8')
    transcript(incsv, outcsv, mode)
