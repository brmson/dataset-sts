"""
A model that follows Fariz Rahman's approach
popular in Keras examples of combining CNN + RNN for text classification.

Performance:
    * anssel-wang:  (the model parameters were tuned to maximize devMRR on wang,
      but only briefly)
      devMRR=0.853077, testMRR=0.770822, testMAP=0.7047
    * anssel-yodaqa:  (same parameters as on wang)
      valMRR=0.380500

"""

from __future__ import print_function
from __future__ import division

from keras.layers.convolutional import Convolution1D, MaxPooling1D
from keras.layers.core import Activation, Dense, Dropout
from keras.layers.recurrent import SimpleRNN, GRU, LSTM
from keras.regularizers import l2

import pysts.kerasts.blocks as B


def config(c):
    c['dropout'] = 4/5
    c['dropoutfix_inp'] = 0
    c['dropoutfix_rec'] = 0
    c['l2reg'] = 1e-4

    c['cnnact'] = 'tanh'
    c['cnninit'] = 'glorot_uniform'
    c['cdim'] = 1
    c['cfiltlen'] = 3
    c['maxpool_len'] = 2

    c['rnnbidi'] = True
    c['rnn'] = GRU
    c['rnnbidi_mode'] = 'sum'
    c['rnnact'] = 'tanh'
    c['rnninit'] = 'glorot_uniform'
    c['sdim'] = 2
    c['rnnlevels'] = 1

    c['project'] = True
    c['pdim'] = 2.5

    # model-external:
    c['inp_e_dropout'] = 1/2
    c['inp_w_dropout'] = 0
    # anssel-specific:
    c['ptscorer'] = B.mlp_ptscorer
    c['mlpsum'] = 'sum'
    c['Ddim'] = 2


def cnn_input(model, N, spad, l2reg=1e-4,
              cnninit='glorot_uniform', cnnact='tanh',
              cdim=2, cfiltlen=3):
    """ A CNN pooling layer that takes sequence of embeddings e0, e1 and
    processes them using a CNN to produce a new sequence of fixed-length-context
    sensitive embeddings.  Returns output tensor shape.

    The output layers are e0c, e1c.
    """
    nb_filter = int(N*cdim)
    model.add_shared_node(name='aconv',
                          inputs=['e0', 'e1'], outputs=['e0c', 'e1c'],
                          layer=Convolution1D(input_shape=(spad, N),
                                              nb_filter=nb_filter, filter_length=cfiltlen,
                                              activation=cnnact, W_regularizer=l2(l2reg),
                                              init=cnninit))

    return (spad - cfiltlen + 1, nb_filter)


def prep_model(model, N, s0pad, s1pad, c):
    (sc, Nc) = cnn_input(model, N, s0pad, l2reg=c['l2reg'],
                         cnninit=c['cnninit'], cnnact=c['cnnact'],
                         cdim=c['cdim'], cfiltlen=c['cfiltlen'])

    if c['maxpool_len'] > 1:
        model.add_shared_node(name='pool', inputs=['e0c', 'e1c'], outputs=['e0g', 'e1g'],
                              layer=MaxPooling1D(pool_length=c['maxpool_len']))
        sc /= c['maxpool_len']
        cnn_outputs = ['e0g', 'e1g']
    else:
        cnn_outputs = ['e0c', 'e1c']
    model.add_node(name='e0c_', input=cnn_outputs[0], layer=Dropout(c['dropout']))
    model.add_node(name='e1c_', input=cnn_outputs[1], layer=Dropout(c['dropout']))

    B.rnn_input(model, Nc, sc, inputs=['e0c_', 'e1c_'],
                dropout=c['dropout'], dropoutfix_inp=c['dropoutfix_inp'], dropoutfix_rec=c['dropoutfix_rec'],
                sdim=c['sdim'],
                rnnbidi=c['rnnbidi'], rnn=c['rnn'], rnnact=c['rnnact'], rnninit=c['rnninit'],
                rnnbidi_mode=c['rnnbidi_mode'], rnnlevels=c['rnnlevels'])

    # Projection
    if c['project']:
        model.add_shared_node(name='proj', inputs=['e0s_', 'e1s_'], outputs=['e0p', 'e1p'],
                              layer=Dense(input_dim=int(N*c['sdim']), output_dim=int(N*c['pdim']),
                                          W_regularizer=l2(c['l2reg'])))
        # This dropout is controversial; it might be harmful to apply,
        # or at least isn't a clear win.
        # model.add_shared_node(name='projdrop', inputs=['e0p', 'e1p'], outputs=['e0p_', 'e1p_'],
        #                       layer=Dropout(c['dropout'], input_shape=(N,)))
        # return ('e0p_', 'e1p_')
        return ('e0p', 'e1p')
    else:
        return ('e0s_', 'e1s_')
