#!/usr/bin/python3
# Generate CSV file from the YodaQA embsel output format.
# Example: jacana2csv.py ../yodaqa/data/ml/embsel/sentsel-curated-train/ sentsel.csv

import csv
import os
import sys


def load_jacana(fname):
    samples = []
    with open(fname, 'rt') as inp:
        for line in inp:
            line = line.strip()
            if line.startswith('<Q> '):
                q = line[len('<Q> '):]
            else:
                l = line.split(' ')
                label = int(l[0])
                text = ' '.join(l[3:])
                samples.append({'qtext': q, 'label': label, 'atext': text})
    return samples


def write_csv(fname, samples):
    with open(fname, 'w') as outp:
        outcsv = csv.DictWriter(outp, fieldnames=['qtext', 'label', 'atext'])
        outcsv.writeheader()
        for s in samples:
            outcsv.writerow(s)


if __name__ == "__main__":
    indir = sys.argv[1]
    samples = []
    for fname in os.listdir(indir):
        samples += load_jacana(indir+'/'+fname)

    outf = sys.argv[2]
    write_csv(outf, samples)
