"""
The "STS for Keras" toolkit.  Contains various Keras blocks that make
putting together and comfortably running neural STS models a breeze.
"""

import numpy as np


def graph_input_anssel(si0, si1, sj0, sj1, se0, se1, y, f0=None, f1=None, s0=None, s1=None, kw=None, akw=None):
    """ Produce Keras task specification from vocab-vectorized sentences.
    The resulting 'gr' structure is our master dataset container, as well
    as something that Keras can recognize as Graph model input.

    * si0, si1:  Words as indices in vocab; 0 == not in vocab
    * sj0, sj1:  Words as indices in GloVe; 0 == not in glove
                 (or in vocab too, which is preferred; never si0>0 and si1>0 at once)
    * se0, se1:  Words as embeddings (based on sj; 0 for nonzero-si)
    * y:  Labels
    * f0, f1:  NLP flags (word class, overlaps, ...)
    * s0, s1:  Words as strings
    * kw, akw:  Scalars for auxiliary pair scoring (overlap scores in yodaqa dataset)

    To get unique word indices, sum si0+sj1.

    """
    gr = {'si0': si0, 'si1': si1,
          'sj0': sj0, 'sj1': sj1,
          'score': y}
    if se0 is not None:
        gr['se0'] = se0
        gr['se1'] = se1
    if f0 is not None:
        gr['f0'] = f0
        gr['f1'] = f1
    if s0 is not None:
        # This is useful for non-neural baselines
        gr['s0'] = s0
        gr['s1'] = s1
    if kw is not None:
        # yodaqa-specific keyword weight counters
        gr['kw'] = kw
        gr['akw'] = akw
    return gr


def graph_nparray_anssel(gr):
    """ Make sure that what should be nparray is nparray. """
    for k in ['si0', 'si1', 'sj0', 'sj1', 'se0', 'se1', 'f0', 'f1', 'score', 'kw', 'akw', 'bm25']:
        if k in gr:
            gr[k] = np.array(gr[k])
    return gr


def graph_input_sts(si0, si1, sj0, sj1, y, f0=None, f1=None, s0=None, s1=None):
    """ Produce Keras task specification from vocab-vectorized sentences. """
    import pysts.loader as loader
    gr = {'si0': si0, 'si1': si1,
          'sj0': sj0, 'sj1': sj1,
          'classes': loader.sts_labels2categorical(y)}
    if f0 is not None:
        gr['f0'] = f0
        gr['f1'] = f1
    if s0 is not None:
        # This is useful for non-neural baselines
        gr['s0'] = s0
        gr['s1'] = s1
    return gr


def graph_input_slice(gr, sl):
    """ Produce a slice of the original graph dataset.

    Example: grs = graph_input_slice(gr, slice(500, 1000)) """
    grs = dict()
    for k, v in gr.items():
        grs[k] = v[sl]
    return grs
