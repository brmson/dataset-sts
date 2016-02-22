"""
A model with a very simple architecture that never-the-less closely
approaches 2015-state-of-art results on the anssel-wang task (with
token flags).

The architecture uses shared bidirectional GRU to produce sentence embeddings,
adaptable word embedding matrix preinitialized with 300D GloVe, projection
matrix (MemNN-like) applied to both sentences to project them to a common
external similarity space.

This will be a part of our upcoming paper; meanwhile, if you need to cite this,
refer to the dataset-sts GitHub repo, please.


Performance:
    * anssel-wang:  (the model parameters were tuned to maximize devMRR on wang)
      * project=False, dot_ptscorer
                rnnbidi=False - devMRR=0.773352, testMRR=0.745151
                rnnbidi=True - devMRR=0.796154, testMRR=0.774527

      * project=True, dot_ptscorer
                rnnbidi=False - devMRR=0.818654, testMRR=0.709342
                rnnbidi=True - devMRR=0.840403, testMRR=0.762395

      * project=False, mlp_ptscorer
                rnnbidi=False - devMRR=0.829890, testMRR=0.756679
                rnnbidi=True - devMRR=0.869744, testMRR=0.787051

      * project=True, mlp_ptscorer (dropout after project)
                rnnbidi=False - devMRR=0.830641, testMRR=0.797059, testMAP=0.7097
                rnnbidi=True - devMRR=0.833887, testMRR=0.808252, testMAP=0.7262

      * project=True, mlp_ptscorer (CURRENT)
                rnnbidi=False - devMRR=0.844359, testMRR=0.813130, testMAP=0.7249
                rnnbidi=True - devMRR=0.857949, testMRR=0.797496, testMAP=0.7275

    * anssel-yodaqa:  (using the wang-tuned parameters)
      valMRR=0.312343

"""

from __future__ import print_function
from __future__ import division

from keras.layers.core import Activation, Dense, Dropout
from keras.layers.recurrent import SimpleRNN, GRU, LSTM
from keras.regularizers import l2

import pysts.kerasts.blocks as B


def config(c):
    c['dropout'] = 3/4
    c['l2reg'] = 1e-4

    c['rnnbidi'] = True
    c['rnn'] = GRU
    c['rnnbidi_mode'] = 'sum'
    c['rnnact'] = 'tanh'
    c['rnninit'] = 'glorot_uniform'
    c['sdim'] = 2
    c['rnnlevels'] = 1

    c['project'] = True
    c['pdim'] = 2.5
    c['pact'] = 'linear'

    # model-external:
    c['inp_e_dropout'] = 3/4
    # anssel-specific:
    c['ptscorer'] = B.mlp_ptscorer
    c['mlpsum'] = 'sum'
    c['Ddim'] = 2


def prep_model(model, N, s0pad, s1pad, c):
    B.rnn_input(model, N, s0pad,
                dropout=c['dropout'], sdim=c['sdim'],
                rnnbidi=c['rnnbidi'], rnn=c['rnn'], rnnact=c['rnnact'], rnninit=c['rnninit'],
                rnnbidi_mode=c['rnnbidi_mode'], rnnlevels=c['rnnlevels'])

    # Projection
    if c['project']:
        model.add_shared_node(name='proj', inputs=['e0s_', 'e1s_'], outputs=['e0p', 'e1p'],
                              layer=Dense(input_dim=int(N*c['sdim']), output_dim=int(N*c['pdim']),
                                          W_regularizer=l2(c['l2reg']), activation=c['pact']))
        # This dropout is controversial; it might be harmful to apply,
        # or at least isn't a clear win.
        # model.add_shared_node(name='projdrop', inputs=['e0p', 'e1p'], outputs=['e0p_', 'e1p_'],
        #                       layer=Dropout(c['dropout'], input_shape=(N,)))
        # return ('e0p_', 'e1p_')
        return ('e0p', 'e1p')
    else:
        return ('e0s_', 'e1s_')
