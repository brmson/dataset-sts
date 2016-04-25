"""
Predefined Keras Graph blocks that represent common model components.
"""

from __future__ import division
from __future__ import print_function

from keras.layers.convolutional import Convolution1D, MaxPooling1D
from keras.layers.core import Activation, Dense, Dropout, Flatten, LambdaMerge
from keras.layers.embeddings import Embedding
from keras.layers.recurrent import GRU
from keras.regularizers import l2

import pysts.nlp as nlp


def embedding(model, glove, vocab, s0pad, s1pad, dropout, dropout_w,
              trainable=True, add_flags=True, create_inputs=True):
    """ The universal sequence input layer.

    Declare inputs si0, si1, f0, f1 (vectorized sentences and NLP flags)
    and generate outputs e0, e1 representing vector sequences, and e0_, e1_
    with dropout applied.  Returns the vector dimensionality.

    With trainable=True, allows adaptation of the embedding matrix during
    training.  With add_flags=True, append the NLP flags to the embeddings. """

    if create_inputs:
        for m, p in [(0, s0pad), (1, s1pad)]:
            model.add_input('si%d'%(m,), input_shape=(p,), dtype='int')
            if add_flags:
                model.add_input('f%d'%(m,), input_shape=(p, nlp.flagsdim))

    if add_flags:
        outputs = ['e0[0]', 'e1[0]']
    else:
        outputs = ['e0', 'e1']

    model.add_shared_node(name='emb', inputs=['si0', 'si1'], outputs=outputs,
                          layer=Embedding(input_dim=vocab.size(), input_length=s1pad,
                                          output_dim=glove.N, mask_zero=True,
                                          weights=[vocab.embmatrix(glove)], trainable=trainable,
                                          dropout=dropout_w))
    if add_flags:
        for m in [0, 1]:
            model.add_node(name='e%d'%(m,), inputs=['e%d[0]'%(m,), 'f%d'%(m,)], merge_mode='concat', layer=Activation('linear'))
        N = glove.N + nlp.flagsdim
    else:
        N = glove.N

    model.add_shared_node(name='embdrop', inputs=['e0', 'e1'], outputs=['e0_', 'e1_'],
                          layer=Dropout(dropout, input_shape=(N,)))

    return N


def rnn_input(model, N, spad, dropout=3/4, dropoutfix_inp=0, dropoutfix_rec=0,
              sdim=2, rnnbidi=True, return_sequences=False,
              rnn=GRU, rnnact='tanh', rnninit='glorot_uniform', rnnbidi_mode='sum',
              rnnlevels=1,
              inputs=['e0_', 'e1_'], pfx=''):
    """ An RNN layer that takes sequence of embeddings e0_, e1_ and
    processes them using an RNN + dropout.

    If return_sequences=False, it returns just the final hidden state of the RNN;
    otherwise, it return a sequence of contextual token embeddings instead.
    At any rate, the output layers are e0s_, e1s_.

    If rnnlevels>1, a multi-level stacked RNN architecture like in Wang&Nyberg
    http://www.aclweb.org/anthology/P15-2116 is applied.
    """
    for i in range(1, rnnlevels):
        rnn_input(model, N, spad, dropout=0, sdim=sdim, rnnbidi=rnnbidi, return_sequences=True,
                  rnn=rnn, rnnact=rnnact, rnninit=rnninit, rnnbidi_mode=rnnbidi_mode,
                  rnnlevels=1, inputs=inputs, pfx='L%d'%(i,))
        inputs = ['L%de0s_'%(i,), 'L%de1s_'%(i,)]

    if rnnbidi:
        if rnnbidi_mode == 'concat':
            sdim /= 2
        model.add_shared_node(name=pfx+'rnnf', inputs=inputs, outputs=[pfx+'e0sf', pfx+'e1sf'],
                              layer=rnn(input_dim=N, output_dim=int(N*sdim), input_length=spad,
                                        init=rnninit, activation=rnnact,
                                        return_sequences=return_sequences,
                                        dropout_W=dropoutfix_inp, dropout_U=dropoutfix_rec))
        model.add_shared_node(name=pfx+'rnnb', inputs=inputs, outputs=[pfx+'e0sb', pfx+'e1sb'],
                              layer=rnn(input_dim=N, output_dim=int(N*sdim), input_length=spad,
                                        init=rnninit, activation=rnnact,
                                        return_sequences=return_sequences, go_backwards=True,
                                        dropout_W=dropoutfix_inp, dropout_U=dropoutfix_rec))
        model.add_node(name=pfx+'e0s', inputs=[pfx+'e0sf', pfx+'e0sb'], merge_mode=rnnbidi_mode, layer=Activation('linear'))
        model.add_node(name=pfx+'e1s', inputs=[pfx+'e1sf', pfx+'e1sb'], merge_mode=rnnbidi_mode, layer=Activation('linear'))

    else:
        model.add_shared_node(name=pfx+'rnn', inputs=inputs, outputs=[pfx+'e0s', pfx+'e1s'],
                              layer=rnn(input_dim=N, output_dim=int(N*sdim), input_length=spad,
                                        init=rnninit, activation=rnnact,
                                        return_sequences=return_sequences,
                                        dropout_W=dropoutfix_inp, dropout_U=dropoutfix_rec))

    model.add_shared_node(name=pfx+'rnndrop', inputs=[pfx+'e0s', pfx+'e1s'], outputs=[pfx+'e0s_', pfx+'e1s_'],
                          layer=Dropout(dropout, input_shape=(spad, int(N*sdim)) if return_sequences else (int(N*sdim),)))


def cnnsum_input(model, N, spad, dropout=3/4, l2reg=1e-4,
                 cnninit='glorot_uniform', cnnact='tanh',
                 cdim={1: 1/2, 2: 1/2, 3: 1/2, 4: 1/2, 5: 1/2},
                 inputs=['e0_', 'e1_'], pfx=''):
    """ An CNN pooling layer that takes sequence of embeddings e0_, e1_ and
    processes them using a CNN + max-pooling to produce a single "summary
    embedding" (*NOT* a sequence of embeddings).

    The layer can apply multiple convolutions of different widths; the
    convolution dimensionality is denoted by the cdim dict, keyed by width
    and containing the number of filters.  The resulting summary embedding
    dimensionality is sum of N*cdim values (the convolutions are concatenated),
    returned by this function for your convenience.

    The output layers are e0s_, e1s_.
    """
    Nc = 0
    for fl, cd in cdim.items():
        nb_filter = int(N*cd)
        model.add_shared_node(name=pfx+'aconv%d'%(fl,),
                              inputs=inputs, outputs=[pfx+'e0c%d'%(fl,), pfx+'e1c%d'%(fl,)],
                              layer=Convolution1D(input_shape=(spad, N),
                                                  nb_filter=nb_filter, filter_length=fl,
                                                  activation=cnnact, W_regularizer=l2(l2reg),
                                                  init=cnninit))
        model.add_shared_node(name=pfx+'apool%d[0]'%(fl,),
                              inputs=[pfx+'e0c%d'%(fl,), pfx+'e1c%d'%(fl,)],
                              outputs=[pfx+'e0s%d[0]'%(fl,), pfx+'e1s%d[0]'%(fl,)],
                              layer=MaxPooling1D(pool_length=int(spad - fl + 1)))
        model.add_shared_node(name=pfx+'apool%d[1]'%(fl,),
                              inputs=[pfx+'e0s%d[0]'%(fl,), pfx+'e1s%s[0]'%(fl,)],
                              outputs=[pfx+'e0s%d'%(fl,), pfx+'e1s%d'%(fl,)],
                              layer=Flatten(input_shape=(1, nb_filter)))
        Nc += nb_filter

    if len(cdim) > 1:
        model.add_node(name=pfx+'e0s', inputs=[pfx+'e0s%d'%(fl,) for fl in cdim.keys()], merge_mode='concat', layer=Activation('linear'))
        model.add_node(name=pfx+'e1s', inputs=[pfx+'e1s%d'%(fl,) for fl in cdim.keys()], merge_mode='concat', layer=Activation('linear'))
    else:
        model.add_node(name=pfx+'e0s', input=pfx+'e0s%d'%(cdim.keys()[0],), layer=Activation('linear'))
        model.add_node(name=pfx+'e1s', input=pfx+'e1s%d'%(cdim.keys()[0],), layer=Activation('linear'))
    model.add_node(name=pfx+'e0s_', input=pfx+'e0s', layer=Dropout(dropout))
    model.add_node(name=pfx+'e1s_', input=pfx+'e1s', layer=Dropout(dropout))

    return Nc


# Match point scoring (scalar output) callables.  Each returns the layer name.
# This is primarily meant as an output layer, but could be used also for
# example as an attention mechanism.

def dot_ptscorer(model, inputs, Ddim, N, l2reg, pfx='out'):
    """ Score the pair using just dot-product, that is elementwise
    multiplication and then sum.  The dot-product is natural because it
    measures the relative directions of vectors, being essentially
    a non-normalized cosine similarity. """
    # (The Activation is a nop, merge_mode is the important part)
    model.add_node(name=pfx+'dot', inputs=inputs, layer=Activation('linear'), merge_mode='dot', dot_axes=1)
    return pfx+'dot'


def cos_ptscorer(model, inputs, Ddim, N, l2reg, pfx='out'):
    """ Score the pair using just cosine similarity. """
    # (The Activation is a nop, merge_mode is the important part)
    model.add_node(name=pfx+'cos', inputs=inputs, layer=Activation('linear'), merge_mode='cos', dot_axes=1)
    return pfx+'cos'


def mlp_ptscorer(model, inputs, Ddim, N, l2reg, pfx='out', sum_mode='sum', extra_inp=[]):
    """ Element-wise features from the pair fed to an MLP. """
    if sum_mode == 'absdiff':
        # model.add_node(name=pfx+'sum', layer=absdiff_merge(model, inputs))
        absdiff_merge(model, inputs, pfx, "sum")
    else:
        model.add_node(name=pfx+'sum', inputs=inputs, layer=Activation('linear'), merge_mode='sum')
    model.add_node(name=pfx+'mul', inputs=inputs, layer=Activation('linear'), merge_mode='mul')

    model.add_node(name=pfx+'hdn', inputs=[pfx+'sum', pfx+'mul'] + extra_inp, merge_mode='concat',
                   layer=Dense(output_dim=int(N*Ddim), W_regularizer=l2(l2reg), activation='sigmoid'))
    model.add_node(name=pfx+'mlp', input=pfx+'hdn',
                   layer=Dense(output_dim=1, W_regularizer=l2(l2reg)))
    return pfx+'mlp'


def mulsum_ptscorer(model, inputs, Ddim, N, l2reg, pfx='out', sum_mode='sum', extra_inp=[]):
    """ Element-wise features from the pair fed to a concat of sum and multiplication of inputs.
     """
    if sum_mode == 'absdiff':
        absdiff_merge(model, inputs, pfx, "sum")
    else:
        model.add_node(name=pfx+'sum', inputs=inputs, layer=Activation('linear'), merge_mode='sum')
    model.add_node(name=pfx+'mul', inputs=inputs, layer=Activation('linear'), merge_mode='mul')

    model.add_node(name=pfx+'mulsum', inputs=[pfx+'sum', pfx+'mul'], merge_mode='concat',
                   layer=Dense(output_dim=1, W_regularizer=l2(l2reg), activation='linear'))
    return pfx+'mulsum'


def cat_ptscorer(model, inputs, Ddim, N, l2reg, pfx='out'):
    """ Just train a linear classifier (weighed sum of elements) on concatenation
    of inputs.  You may pass also just a single input (which may make sense
    if you for example process s1 "with regard to s0"). """
    if len(inputs) > 1:
        model.add_node(name=pfx+'cat', inputs=inputs, merge_mode='concat',
                       layer=Dense(output_dim=1, W_regularizer=l2(l2reg)))
    else:
        model.add_node(name=pfx+'cat', input=inputs[0],
                       layer=Dense(output_dim=1, W_regularizer=l2(l2reg)))
    return pfx+'cat'



def absdiff_merge(model, inputs, pfx="out", layer_name="absdiff"):
    """ Merging two layers into one, via element-wise subtraction and then taking absolute value.

    Example of usage: layer_name = absdiff_merge(model, inputs=["e0_", "e1_"])

    TODO: The more modern way appears to be to use "join" merge mode and Lambda layer.
    """
    if len(inputs) != 2:
        raise ValueError("absdiff_merge has to got exactly 2 inputs")

    def diff(X):
        return K.abs(X[0] - X[1])

    def output_shape(input_shapes):
        return input_shapes[0]

    full_name = "%s%s" % (pfx, layer_name)
    model.add_node(name=layer_name, inputs=inputs,
                   layer=LambdaMerge([model.nodes[l] for l in inputs], diff, output_shape))
    return full_name


def dot_time_distributed_merge(model, layers, cos_norm=False):
    """ Merging two time series layers into one, producing a new time series that
    contains a dot-product scalar for each time step.

    If cos_norm=True, actually computes cosine similarity. """
    def batched_batched_dot(s):
        """ from (x,y,z)-shaped pair, produce (x,y)-shaped pair that replaces the z-vector pairs by their dot-products """
        import theano
        import theano.tensor as T
        return theano.scan(fn=lambda xm, ym: T.batched_dot(xm, ym),
                           outputs_info=None, sequences=s, non_sequences=None)[0]

    def batched_cos_sim(s):
        """ from (x,y,z)-shaped pair, produce (x,y)-shaped pair that replaces the z-vector pairs by their cosine similarities """
        import theano
        import theano.tensor as T
        return theano.scan(fn=lambda xm, ym: T.batched_dot(xm, ym) / T.sqrt(T.batched_dot(xm, xm) * T.batched_dot(ym, ym)),
                           outputs_info=None, sequences=s, non_sequences=None)[0]

    if cos_norm:
        lmb = batched_cos_sim
    else:
        lmb = batched_batched_dot

    return LambdaMerge([model.nodes[l] for l in layers], lmb,
                       lambda s: (s[1][0], s[1][1]))
