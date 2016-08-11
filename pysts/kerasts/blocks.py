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

    Declare inputs si0, si1, se0, se1, f0, f1 (vectorized sentences and NLP flags)
    and generate outputs e0, e1.  Returns the vector dimensionality.

    The word vectors are passed either as non-zero **si** element and zero **se**
    vector, or vice versa.  **si** are indices to a trainable embedding matrix,
    while **se** are hardcoded embeddings.  The motivation of this is that most
    frequent words (which represent semantic operators, like "and", "not",
    "how" etc.) as well as special words (OOV, interpunction) are passed as
    indices and therefore have adaptable embeddings, while the long tail of
    less used words is substituted to hardcoded embeddings on input, so that
    the full GloVe matrix does not need to be in GPU memory and we generalize
    to words unseen at training time.  At any rate, after the embedding layer
    the inputs are summed, so for the rest of the models it makes no difference
    how each word is passed.

    With trainable=True, allows adaptation of the embedding matrix during
    training.  With add_flags=True, append the NLP flags to the embeddings. """

    if create_inputs:
        for m, p in [(0, s0pad), (1, s1pad)]:
            model.add_input('si%d'%(m,), input_shape=(p,), dtype='int')
            model.add_input('se%d'%(m,), input_shape=(p, glove.N))
            if add_flags:
                model.add_input('f%d'%(m,), input_shape=(p, nlp.flagsdim))

    emb = vocab.embmatrix(glove)
    model.add_shared_node(name='emb', inputs=['si0', 'si1'], outputs=['e0[0]', 'e1[0]'],
                          layer=Embedding(input_dim=emb.shape[0], input_length=s1pad,
                                          output_dim=glove.N, mask_zero=True,
                                          weights=[emb], trainable=trainable,
                                          dropout=dropout_w))
    model.add_node(name='e0[1]', inputs=['e0[0]', 'se0'], merge_mode='sum', layer=Activation('linear'))
    model.add_node(name='e1[1]', inputs=['e1[0]', 'se1'], merge_mode='sum', layer=Activation('linear'))
    eputs = ['e0[1]', 'e1[1]']
    if add_flags:
        for m in [0, 1]:
            model.add_node(name='e%d[f]'%(m,), inputs=[eputs[m], 'f%d'%(m,)], merge_mode='concat', layer=Activation('linear'))
        eputs = ['e0[f]', 'e1[f]']
        N = glove.N + nlp.flagsdim
    else:
        N = glove.N

    model.add_shared_node(name='embdrop', inputs=eputs, outputs=['e0', 'e1'],
                          layer=Dropout(dropout, input_shape=(N,)))

    return N


def rnn_input(model, N, spad, dropout=3/4, dropoutfix_inp=0, dropoutfix_rec=0,
              sdim=2, rnnbidi=True, return_sequences=False,
              rnn=GRU, rnnact='tanh', rnninit='glorot_uniform', rnnbidi_mode='sum',
              rnnlevels=1,
              inputs=['e0', 'e1'], pfx=''):
    """ An RNN layer that takes sequence of embeddings e0, e1 and
    processes them using an RNN + dropout.

    If return_sequences=False, it returns just the final hidden state of the RNN;
    otherwise, it return a sequence of contextual token embeddings instead.
    At any rate, the output layers are e0s_, e1s_.

    If rnnlevels>1, a multi-level stacked RNN architecture like in Wang&Nyberg
    http://www.aclweb.org/anthology/P15-2116 is applied, however with skip-connections
    i.e. the inner RNNs have both the outer RNN and original embeddings as inputs.
    """
    deep_inputs = inputs
    for i in range(1, rnnlevels):
        rnn_input(model, N, spad, dropout=0, sdim=sdim, rnnbidi=rnnbidi, return_sequences=True,
                  rnn=rnn, rnnact=rnnact, rnninit=rnninit, rnnbidi_mode=rnnbidi_mode,
                  rnnlevels=1, inputs=deep_inputs, pfx=pfx+'L%d'%(i,))
        model.add_node(name=pfx+'L%de0s_j'%(i,), inputs=[inputs[0], pfx+'L%de0s_'%(i,)], merge_mode='concat', layer=Activation('linear'))
        model.add_node(name=pfx+'L%de1s_j'%(i,), inputs=[inputs[1], pfx+'L%de1s_'%(i,)], merge_mode='concat', layer=Activation('linear'))
        deep_inputs = ['L%de0s_j'%(i,), 'L%de1s_j'%(i,)]

    if rnnbidi:
        if rnnbidi_mode == 'concat':
            sdim /= 2
        model.add_shared_node(name=pfx+'rnnf', inputs=deep_inputs, outputs=[pfx+'e0sf', pfx+'e1sf'],
                              layer=rnn(input_dim=N, output_dim=int(N*sdim), input_length=spad,
                                        init=rnninit, activation=rnnact,
                                        return_sequences=return_sequences,
                                        dropout_W=dropoutfix_inp, dropout_U=dropoutfix_rec))
        model.add_shared_node(name=pfx+'rnnb', inputs=deep_inputs, outputs=[pfx+'e0sb', pfx+'e1sb'],
                              layer=rnn(input_dim=N, output_dim=int(N*sdim), input_length=spad,
                                        init=rnninit, activation=rnnact,
                                        return_sequences=return_sequences, go_backwards=True,
                                        dropout_W=dropoutfix_inp, dropout_U=dropoutfix_rec))
        model.add_node(name=pfx+'e0s', inputs=[pfx+'e0sf', pfx+'e0sb'], merge_mode=rnnbidi_mode, layer=Activation('linear'))
        model.add_node(name=pfx+'e1s', inputs=[pfx+'e1sf', pfx+'e1sb'], merge_mode=rnnbidi_mode, layer=Activation('linear'))

    else:
        model.add_shared_node(name=pfx+'rnn', inputs=deep_inputs, outputs=[pfx+'e0s', pfx+'e1s'],
                              layer=rnn(input_dim=N, output_dim=int(N*sdim), input_length=spad,
                                        init=rnninit, activation=rnnact,
                                        return_sequences=return_sequences,
                                        dropout_W=dropoutfix_inp, dropout_U=dropoutfix_rec))

    model.add_shared_node(name=pfx+'rnndrop', inputs=[pfx+'e0s', pfx+'e1s'], outputs=[pfx+'e0s_', pfx+'e1s_'],
                          layer=Dropout(dropout, input_shape=(spad, int(N*sdim)) if return_sequences else (int(N*sdim),)))


def add_multi_node(model, name, inputs, outputs, layer_class,
        layer_args, siamese=True, **kwargs):
    if siamese:
        layer = layer_class(**layer_args)
        model.add_shared_node(name=name, inputs=inputs, outputs=outputs,
                layer=layer, **kwargs)
    else:
        for inp, outp in zip(inputs, outputs):
            layer = layer_class(**layer_args)
            model.add_node(name=outp, input=inp, layer=layer, **kwargs)


def cnnsum_input(model, N, spad, dropout=3/4, l2reg=1e-4,
                 cnninit='glorot_uniform', cnnact='tanh',
                 cdim={1: 1/2, 2: 1/2, 3: 1/2, 4: 1/2, 5: 1/2},
                 inputs=['e0', 'e1'], pfx='', siamese=True):
    """ An CNN pooling layer that takes sequence of embeddings e0, e1 and
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
        add_multi_node(model, name=pfx+'aconv%d'%(fl,), siamese=siamese,
                              inputs=inputs, outputs=[pfx+'e0c%d'%(fl,), pfx+'e1c%d'%(fl,)],
                              layer_class=Convolution1D,
                              layer_args={'input_shape':(spad, N),
                                  'nb_filter':nb_filter,
                                  'filter_length':fl,
                                  'activation':cnnact,
                                  'W_regularizer':l2(l2reg),
                                  'init':cnninit})
        add_multi_node(model, name=pfx+'apool%d[0]'%(fl,), siamese=siamese,
                              inputs=[pfx+'e0c%d'%(fl,), pfx+'e1c%d'%(fl,)],
                              outputs=[pfx+'e0s%d[0]'%(fl,), pfx+'e1s%d[0]'%(fl,)],
                              layer_class=MaxPooling1D,
                              layer_args={'pool_length':int(spad - fl + 1)})
        add_multi_node(model, name=pfx+'apool%d[1]'%(fl,), siamese=siamese,
                              inputs=[pfx+'e0s%d[0]'%(fl,), pfx+'e1s%s[0]'%(fl,)],
                              outputs=[pfx+'e0s%d'%(fl,), pfx+'e1s%d'%(fl,)],
                              layer_class=Flatten, layer_args={'input_shape':(1, nb_filter)})
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

def dot_ptscorer(model, inputs, Ddim, N, l2reg, pfx='out', extra_inp=[]):
    """ Score the pair using just dot-product, that is elementwise
    multiplication and then sum.  The dot-product is natural because it
    measures the relative directions of vectors, being essentially
    a non-normalized cosine similarity. """
    # (The Activation is a nop, merge_mode is the important part)
    model.add_node(name=pfx+'dot', inputs=inputs, layer=Activation('linear'), merge_mode='dot', dot_axes=1)
    if extra_inp:
        model.add_node(name=pfx+'mlp', inputs=[pfx+'dot'] + extra_inp, merge_mode='concat',
                       layer=Dense(output_dim=1, W_regularizer=l2(l2reg)))
        return pfx+'mlp'
    else:
        return pfx+'dot'


def cos_ptscorer(model, inputs, Ddim, N, l2reg, pfx='out', extra_inp=[]):
    """ Score the pair using just cosine similarity. """
    # (The Activation is a nop, merge_mode is the important part)
    model.add_node(name=pfx+'cos', inputs=inputs, layer=Activation('linear'), merge_mode='cos', dot_axes=1)
    if extra_inp:
        model.add_node(name=pfx+'mlp', inputs=[pfx+'cos'] + extra_inp, merge_mode='concat',
                       layer=Dense(output_dim=1, W_regularizer=l2(l2reg)))
        return pfx+'mlp'
    else:
        return pfx+'cos'


def mlp_ptscorer(model, inputs, Ddim, N, l2reg, pfx='out', Dinit='glorot_uniform', sum_mode='sum', extra_inp=[]):
    """ Element-wise features from the pair fed to an MLP. """
    if sum_mode == 'absdiff':
        # model.add_node(name=pfx+'sum', layer=absdiff_merge(model, inputs))
        absdiff_merge(model, inputs, pfx, "sum")
    else:
        model.add_node(name=pfx+'sum', inputs=inputs, layer=Activation('linear'), merge_mode='sum')
    model.add_node(name=pfx+'mul', inputs=inputs, layer=Activation('linear'), merge_mode='mul')
    mlp_inputs = [pfx+'sum', pfx+'mul'] + extra_inp

    def mlp_args(mlp_inputs):
        """ return model.add_node() args that are good for mlp_inputs list
        of both length 1 and more than 1. """
        mlp_args = dict()
        if len(mlp_inputs) > 1:
            mlp_args['inputs'] = mlp_inputs
            mlp_args['merge_mode'] = 'concat'
        else:
            mlp_args['input'] = mlp_inputs[0]
        return mlp_args

    # Ddim may be either 0 (no hidden layer), scalar (single hidden layer) or
    # list (multiple hidden layers)
    if Ddim == 0:
        Ddim = []
    elif not isinstance(Ddim, list):
        Ddim = [Ddim]
    if Ddim:
        for i, D in enumerate(Ddim):
            model.add_node(name=pfx+'hdn[%d]'%(i,),
                           layer=Dense(output_dim=int(N*D), W_regularizer=l2(l2reg), activation='tanh', init=Dinit),
                           **mlp_args(mlp_inputs))
            mlp_inputs = [pfx+'hdn[%d]'%(i,)]

    model.add_node(name=pfx+'mlp',
                   layer=Dense(output_dim=1, W_regularizer=l2(l2reg)), **mlp_args(mlp_inputs))
    return pfx+'mlp'


def cat_ptscorer(model, inputs, Ddim, N, l2reg, pfx='out', extra_inp=[]):
    """ Just train a linear classifier (weighed sum of elements) on concatenation
    of inputs.  You may pass also just a single input (which may make sense
    if you for example process s1 "with regard to s0"). """
    if len(list(inputs) + extra_inp) > 1:
        model.add_node(name=pfx+'cat', inputs=list(inputs) + extra_inp, merge_mode='concat',
                       layer=Dense(output_dim=1, W_regularizer=l2(l2reg)))
    else:
        model.add_node(name=pfx+'cat', input=inputs[0],
                       layer=Dense(output_dim=1, W_regularizer=l2(l2reg)))
    return pfx+'cat'



def absdiff_merge(model, inputs, pfx="out", layer_name="absdiff"):
    """ Merging two layers into one, via element-wise subtraction and then taking absolute value.

    Example of usage: layer_name = absdiff_merge(model, inputs=["e0", "e1"])

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
