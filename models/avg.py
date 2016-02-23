"""
A simple averaging model.

TODO: Support pre-averaging non-linear projection.  These should help with
anssel-wang as they could give more weight to per-token overlap features.


Performance:
    * anssel-wang:
      devMRR=0.738617, testMRR=0.630773, testMAP=0.556700

    * anssel-yodaqa:
      valMRR=0.334864
"""

from __future__ import print_function
from __future__ import division

from keras.layers.core import Activation, Dense, Dropout, TimeDistributedMerge
from keras.regularizers import l2

import pysts.kerasts.blocks as B


def config(c):
    c['l2reg'] = 1e-5

    c['deep'] = 0
    c['nnact'] = 'relu'
    c['nninit'] = 'glorot_uniform'

    c['project'] = True
    c['pdim'] = 1
    c['pact'] = 'tanh'

    # model-external:
    c['inp_e_dropout'] = 1/3
    # anssel-specific:
    c['ptscorer'] = B.dot_ptscorer
    c['mlpsum'] = 'sum'
    c['Ddim'] = 1


def prep_model(model, N, s0pad, s1pad, c):
    model.add_shared_node(name='bow', inputs=['e0_', 'e1_'], outputs=['e0b', 'e1b'],
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
