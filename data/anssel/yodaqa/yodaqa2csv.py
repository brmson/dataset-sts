#!/usr/bin/python3
# Generate CSV file from the YodaQA embsel output format.
# Example: yodaqa2csv.py ../dataset-factoid-curated/curated-train.tsv ../yodaqa/data/ml/embsel/sentsel-curated-train/ sentsel.csv

import csv
import os
import re
import sys

from nltk.tokenize import word_tokenize


def load_regexen(fname):
    regexen = dict()
    with open(fname, 'rt') as inp:
        for line in inp:
            line = line.strip()
            line = line.split('\t')
            regexen[line[2].strip()] = line[3].strip()
    return regexen


def regex_overlap(text, regex):
    """ for a list of tokens in text and a regex, match it in the
    tokens and return a list of booleans denoting which tokens are
    a part of the match """
    overlap = [False for t in text]
    for m in re.finditer(regex, ' '.join(text)):
        begin, end = m.span()
        i = 0
        for j, t in enumerate(text):
            if begin <= i < end or begin <= i+len(t) < end:
                overlap[j] = True
            i += len(t) + 1
    return overlap


def load_jacana(fname, regexen):
    samples = []
    with open(fname, 'rt') as inp:
        for line in inp:
            line = line.strip()
            if line.startswith('<Q> '):
                qorig = line[len('<Q> '):]
                q = word_tokenize(qorig)
            else:
                l = line.split(' ')
                label = int(l[0])
                kwweight = float(l[1])
                aboutkwweight = float(l[2])
                text = word_tokenize(' '.join(l[3:]))
                toklabels = regex_overlap(text, regexen[qorig])
                samples.append({'qtext': ' '.join(q), 'label': label,
                                'atext': ' '.join(text),
                                'kwweight': kwweight, 'aboutkwweight': aboutkwweight,
                                'toklabels': ' '.join([str(0+tl) for tl in toklabels])})
    return samples


def write_csv(fname, samples):
    with open(fname, 'w') as outp:
        outcsv = csv.DictWriter(outp, fieldnames=['qtext', 'label', 'atext', 'kwweight', 'aboutkwweight', 'toklabels'])
        outcsv.writeheader()
        for s in samples:
            outcsv.writerow(s)


if __name__ == "__main__":
    gsfile = sys.argv[1]
    regexen = load_regexen(gsfile)

    indir = sys.argv[2]
    samples = []
    for fname in os.listdir(indir):
        samples += load_jacana(indir+'/'+fname, regexen)

    outf = sys.argv[3]
    write_csv(outf, samples)
