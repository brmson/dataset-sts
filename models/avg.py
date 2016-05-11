"""
A simple averaging model.

In its default settings, this is the baseline unigram (Yu, 2014) approach
http://arxiv.org/abs/1412.1632 of training (M, b) such that:

    f(q, a) = sigmoid(q * M * a.T + b)

However, rather than a dot-product, the MLP comparison is used as it works
dramatically better.

This model can also represent the Deep Averaging Networks
(http://cs.umd.edu/~miyyer/pubs/2015_acl_dan.pdf) with this configuration:

    inp_e_dropout=0 inp_w_dropout=1/3 deep=2 "pact='relu'"

The model also supports preprojection of embeddings (not done by default;
wproj=True), though it doesn't do a lot of good it seems - the idea was to
allow mixin of NLP flags.


Performance:
    * anssel-yodaqa:
      valMRR=0.334864 (dot)
"""

from __future__ import print_function
from __future__ import division

from keras.layers.core import Activation, Dense, Dropout, TimeDistributedDense, TimeDistributedMerge
from keras.regularizers import l2

import pysts.kerasts.blocks as B


def config(c):
    c['l2reg'] = 1e-5

    # word-level projection before averaging
    c['wproject'] = False
    c['wdim'] = 1
    c['wact'] = 'linear'

    c['deep'] = 0
    c['nnact'] = 'relu'
    c['nninit'] = 'glorot_uniform'

    c['project'] = True
    c['pdim'] = 1
    c['pact'] = 'tanh'

    # model-external:
    c['inp_e_dropout'] = 1/3
    c['inp_w_dropout'] = 0
    # anssel-specific:
    c['ptscorer'] = B.mlp_ptscorer
    c['mlpsum'] = 'sum'
    c['Ddim'] = 1


def prep_model(model, N, s0pad, s1pad, c):
    winputs = ['e0', 'e1']
    if c['wproject']:
        model.add_shared_node(name='wproj', inputs=winputs, outputs=['e0w', 'e1w'],
                              layer=TimeDistributedDense(output_dim=int(N*c['wdim']),
                                                         activation=c['wact']))
        winputs = ['e0w', 'e1w']

    model.add_shared_node(name='bow', inputs=winputs, outputs=['e0b', 'e1b'],
                          layer=TimeDistributedMerge(mode='ave'))
    bow_last = ('e0b', 'e1b')

    for i in range(c['deep']):
        bow_next = ('e0b[%d]'%(i,), 'e1b[%d]'%(i,))
        model.add_shared_node(name='deep[%d]'%(i,), inputs=bow_last, outputs=bow_next,
                              layer=Dense(output_dim=N, init=c['nninit'],
                                          activation=c['nnact'],
                                          W_regularizer=l2(c['l2reg'])))
        bow_last = bow_next

    # Projection
    if c['project']:
        model.add_shared_node(name='proj', inputs=bow_last, outputs=['e0p', 'e1p'],
                              layer=Dense(input_dim=N, output_dim=int(N*c['pdim']),
                                          activation=c['pact'],
                                          W_regularizer=l2(c['l2reg'])))
        return ('e0p', 'e1p')
    else:
        return bow_last
