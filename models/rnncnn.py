"""
A simple model that combines RNN + CNN.  It's CNN with input
pre-smeared by RNN, or attn1511 without the attentoin.
"""

from __future__ import print_function
from __future__ import division

from keras.layers.core import Activation, Dense, Dropout
from keras.layers.recurrent import SimpleRNN, GRU, LSTM
from keras.regularizers import l2

import pysts.kerasts.blocks as B


def config(c):
    c['dropout'] = 0
    c['dropoutfix_inp'] = 0
    c['dropoutfix_rec'] = 0
    c['l2reg'] = 1e-4

    c['rnnbidi'] = True
    c['rnn'] = GRU
    c['rnnbidi_mode'] = 'sum'
    c['rnnact'] = 'tanh'
    c['rnninit'] = 'glorot_uniform'
    c['sdim'] = 1
    c['rnnlevels'] = 1

    c['cnnsiamese'] = True
    c['cnnact'] = 'relu'
    c['cnninit'] = 'glorot_uniform'
    c['cdim'] = {1: 1, 2: 1/2, 3: 1/2, 4: 1/2, 5: 1/2}

    c['project'] = True
    c['pdim'] = 2
    c['pact'] = 'tanh'

    # model-external:
    c['inp_e_dropout'] = 0
    c['inp_w_dropout'] = 0
    # anssel-specific:
    c['ptscorer'] = B.mlp_ptscorer
    c['mlpsum'] = 'sum'
    c['Ddim'] = 2


def prep_model(model, N, s0pad, s1pad, c):
    B.rnn_input(model, N, s0pad, return_sequences=True,
                dropout=c['dropout'], dropoutfix_inp=c['dropoutfix_inp'], dropoutfix_rec=c['dropoutfix_rec'],
                sdim=c['sdim'],
                rnnbidi=c['rnnbidi'], rnn=c['rnn'], rnnact=c['rnnact'], rnninit=c['rnninit'],
                rnnbidi_mode=c['rnnbidi_mode'], rnnlevels=c['rnnlevels'])

    Nc = B.cnnsum_input(model, N, s0pad, inputs=['e0s_', 'e1s_'], pfx='cnn',
                        dropout=c['dropout'], l2reg=c['l2reg'], siamese=c['cnnsiamese'],
                        cnninit=c['cnninit'], cnnact=c['cnnact'], cdim=c['cdim'])

    # Projection
    if c['project']:
        model.add_shared_node(name='proj', inputs=['cnne0s_', 'cnne1s_'], outputs=['e0p', 'e1p'],
                              layer=Dense(input_dim=Nc, output_dim=int(N*c['pdim']),
                                          W_regularizer=l2(c['l2reg']), activation=c['pact']))
        # This dropout is controversial; it might be harmful to apply,
        # or at least isn't a clear win.
        # model.add_shared_node(name='projdrop', inputs=['e0p', 'e1p'], outputs=['e0p_', 'e1p_'],
        #                       layer=Dropout(c['dropout'], input_shape=(N,)))
        # return ('e0p_', 'e1p_')
        return ('e0p', 'e1p')
    else:
        return ('cnne0s_', 'cnne1s_')
