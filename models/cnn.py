"""
A model with a very simple architecture that never-the-less achieves
2015-state-of-art results on the anssel-wang task (with token flags).
You can also see this model as a standalone fully contained script
in ``examples/anssel-cnn.py``.

The architecture uses multi-width CNN and max-pooling to produce sentence embeddings,
adaptable word embedding matrix preinitialized with 300D GloVe and a projection
matrix (MemNN-like) applied to both sentences to project them to a common
external similarity space.

This will be a part of our upcoming paper; meanwhile, if you need to cite this,
refer to the dataset-sts GitHub repo, please.


Performance:
    * anssel-wang:  (the model parameters were tuned to maximize devMRR on wang)
      devMRR=0.876154, testMRR=0.820956, testMAP=0.7321
    * anssel-yodaqa:  (using the wang-tuned parameters)
      valMRR=0.377590
"""

from __future__ import print_function
from __future__ import division

from keras.layers.core import Activation, Dense, Dropout
from keras.regularizers import l2

import pysts.kerasts.blocks as B


def config(c):
    c['dropout'] = 4/5
    c['l2reg'] = 1e-4

    c['cnnact'] = 'tanh'
    c['cnninit'] = 'glorot_uniform'
    c['cdim'] = {1: 1, 2: 1/2, 3: 1/2, 4: 1/2, 5: 1/2}

    c['project'] = True
    c['pdim'] = 2.5
    c['pact'] = 'linear'

    # model-external:
    c['inp_e_dropout'] = 1/2
    # anssel-specific:
    c['ptscorer'] = B.mlp_ptscorer
    c['mlpsum'] = 'sum'
    c['Ddim'] = 1


def prep_model(model, N, s0pad, s1pad, c):
    Nc = B.cnnsum_input(model, N, s0pad,
                        dropout=c['dropout'], l2reg=c['l2reg'],
                        cnninit=c['cnninit'], cnnact=c['cnnact'], cdim=c['cdim'])

    # Projection
    if c['project']:
        model.add_shared_node(name='proj', inputs=['e0s_', 'e1s_'], outputs=['e0p', 'e1p'],
                              layer=Dense(input_dim=Nc, output_dim=int(N*c['pdim']),
                                          W_regularizer=l2(c['l2reg']), activation=c['pact']))
        # This dropout is controversial; it might be harmful to apply,
        # or at least isn't a clear win.
        # model.add_shared_node(name='projdrop', inputs=['e0p', 'e1p'], outputs=['e0p_', 'e1p_'],
        #                       layer=Dropout(c['dropout'], input_shape=(N,)))
        # return ('e0p_', 'e1p_')
        return ('e0p', 'e1p')
    else:
        return ('e0s_', 'e1s_')
