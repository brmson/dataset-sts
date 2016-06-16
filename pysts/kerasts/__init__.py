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


def graph_input_prune(gr, ypred, N, skip_oneclass=False):
    """ Given a gr and a given scoring, keep only top N s1 for each s0,
    and stash the others away to _x-suffixed keys (for potential recovery). """
    def prune_filter(ypred, N):
        """ yield (index, passed) tuples """
        ys = sorted(enumerate(ypred), key=lambda yy: yy[1], reverse=True)
        i = 0
        for n, y in ys:
            yield n, (i < N)
            i += 1

    # Go through (s0, s1), keeping track of the beginning of the current
    # s0 block, and appending pruned versions
    i = 0
    grp = dict([(k, []) for k in gr.keys()] + [(k+'_x', []) for k in gr.keys()])
    for j in range(len(gr['si0']) + 1):
        if j < len(gr['si0']) and (j == 0 or np.all(gr['si0'][j]+gr['sj0'][j] == gr['si0'][j-1]+gr['sj0'][j-1])):
            # within same-s0 block, carry on
            continue
        # block boundary

        # possibly check if we have both classes picked (for training)
        if skip_oneclass:
            n_picked = 0
            for n, passed in prune_filter(ypred[i:j], N):
                if not passed:
                    break
                n_picked += gr['score'][i + n] > 0
            if n_picked == 0:
                # only false; tough luck, prune everything for this s0
                for k in gr.keys():
                    grp[k+'_x'] += list(gr[k][i:j])
                i = j
                continue

        # append pruned subset
        for n, passed in prune_filter(ypred[i:j], N):
            for k in gr.keys():
                if passed:
                    grp[k].append(gr[k][i + n])
                else:
                    grp[k+'_x'].append(gr[k][i + n])

        i = j

    return graph_nparray_anssel(grp)


def graph_input_unprune(gro, grp, ypred, xval):
    """ Reconstruct original graph gro from a pruned graph grp,
    with predictions set to always False for the filtered out samples.
    (xval denotes how the False is represented) """
    if 'score_x' not in grp:
        return grp, ypred  # not actually pruned

    gru = dict([(k, list(grp[k])) for k in gro.keys()])

    # XXX: this will generate non-continuous s0 blocks,
    # hopefully okay for all ev tools
    for k in gro.keys():
        gru[k] += grp[k+'_x']
    ypred = list(ypred)
    ypred += [xval for i in grp['score_x']]
    ypred = np.array(ypred)

    return graph_nparray_anssel(gru), ypred
