#!/usr/bin/python3
# Convert the Jacana/Heilman(?) style pseudo-XML files to simple CSVs.
# Example: data/anssel/wang/pseudoxml2csv.py ../jacana-data/dev-less-than-40.xml data/anssel/wang/dev.csv

import csv
import sys


def load_pseudoxml(fname):
    samples = []
    state = None
    qtext = None
    with open(fname, 'rt') as inp:
        for line in inp:
            line = line.strip()
            if line.startswith('<') and not (line.startswith('<num>') or line.startswith('<\t')):
                state = line
                continue
            elif state is None:
                continue  # non-first-line in a block
            elif state == '<question>':
                qtext = line.split('\t')
            elif state == '<positive>':
                samples.append({'qtext': ' '.join(qtext), 'label': 1, 'atext': ' '.join(line.split('\t'))})
            elif state == '<negative>':
                samples.append({'qtext': ' '.join(qtext), 'label': 0, 'atext': ' '.join(line.split('\t'))})
            else:
                raise ValueError((state, line))
            state = None
    return samples


def write_csv(fname, samples):
    with open(fname, 'w') as outp:
        outcsv = csv.DictWriter(outp, fieldnames=['qtext', 'label', 'atext'])
        outcsv.writeheader()
        for s in samples:
            outcsv.writerow(s)


if __name__ == "__main__":
    inf = sys.argv[1]
    samples = load_pseudoxml(inf)

    outf = sys.argv[2]
    write_csv(outf, samples)
