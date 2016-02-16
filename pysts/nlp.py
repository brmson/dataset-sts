"""
NLP preprocessing tools for sentences.

Currently, this just tags the token sequences by some trivial boolean flags
that denote some token characteristics and sentence-sentence overlaps.

In principle, this module could however include a lot more sophisticated
NLP tagging pipelines, or loading precomputed such data.
"""

import numpy as np
import re

from nltk.corpus import stopwords
stop = stopwords.words('english')


flagsdim = 4

def sentence_flags(s0, s1, s0pad, s1pad):
    """ For sentence lists s0, s1, generate numpy tensor
    (#sents, spad, flagsdim) that contains a sparse indicator vector of
    various token properties.  It is meant to be concatenated to the token
    embedding. """

    def gen_iflags(s, spad):
        iflags = []
        for i in range(len(s)):
            iiflags = [[False, False] for j in range(spad)]
            for j, t in enumerate(s[i]):
                if j >= spad:
                    break
                number = False
                capital = False
                if re.match('^[0-9\W]*[0-9]+[0-9\W]*$', t):
                    number = True
                if j > 0 and re.match('^[A-Z]', t):
                    capital = True
                iiflags[j] = [number, capital]
            iflags.append(iiflags)
        return iflags

    def gen_mflags(s0, s1, s0pad):
        """ generate flags for s0 that represent overlaps with s1 """
        mflags = []
        for i in range(len(s0)):
            mmflags = [[False, False] for j in range(s0pad)]
            for j in range(min(s0pad, len(s0[i]))):
                unigram = False
                bigram = False
                for k in range(len(s1[i])):
                    if s0[i][j].lower() != s1[i][k].lower():
                        continue
                    # do not generate trivial overlap flags, but accept them as part of bigrams                    
                    if s0[i][j].lower() not in stop and not re.match('^\W+$', s0[i][j]):
                        unigram = True
                    try:
                        if s0[i][j+1].lower() == s1[i][k+1].lower():
                            bigram = True
                    except IndexError:
                        pass
                mmflags[j] = [unigram, bigram]
            mflags.append(mmflags)
        return mflags

    # individual flags (for understanding)
    iflags0 = gen_iflags(s0, s0pad)
    iflags1 = gen_iflags(s1, s1pad)

    # s1-s0 match flags (for attention)
    mflags0 = gen_mflags(s0, s1, s0pad)
    mflags1 = gen_mflags(s1, s0, s1pad)

    return [np.dstack((iflags0, mflags0)),
            np.dstack((iflags1, mflags1))]
