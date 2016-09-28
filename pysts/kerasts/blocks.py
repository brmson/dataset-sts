"""
Predefined Keras model blocks that represent common model components.

The block-layers are tiny Keras models.  This for example means that
parameter sharing is implied by instantiating a block-layer once and
calling it multiple times!
"""

from __future__ import division
from __future__ import print_function

from keras.layers import Input, merge
from keras.layers.convolutional import Convolution1D, MaxPooling1D
from keras.layers.core import Dropout, Flatten
from keras.layers.embeddings import Embedding
from keras.layers.recurrent import GRU
from keras.layers.wrappers import Bidirectional
from keras.models import Model
from keras.regularizers import l2


#from keras.layers.core import Activation, Dense, Dropout, Flatten, LambdaMerge

import pysts.nlp as nlp


class SentenceInputs(object):
    """ A set of inputs that together represent the m-th input sequence
    (sentence).

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
    how each word is passed. """
    def __init__(self, m, N, spad, add_flags=True):
        # word indices
        self.si = Input(shape=(spad,), dtype='int32', name='si%d'%(m,))
        # external embeddings
        self.se = Input(shape=(spad, N), name='se%d'%(m,))
        self.inputs = [self.si, self.se]
        if add_flags:
            self.f = Input(shape=(spad, nlp.flagsdim), name='f%d'%(m,))
            self.inputs.append(self.f)
        else:
            self.f = None


def inputs_pair(glove, spad, add_flags=True):
    return (SentenceInputs(0, glove.N, spad, add_flags=add_flags),
            SentenceInputs(1, glove.N, spad, add_flags=add_flags))


class WordsEmbedding(object):
    """ The sequence input block-layer that merges all SentenceInputs inputs
    to a single neural channel.

    The constructor builds the model and sets up a callable that behaves like
    applying the model, only that it expects SentenceInputs as an argument
    rather than an ordinary Keras variable(s).  The output dimensionality is
    available as self.N.

    With trainable=True, allows adaptation of the embedding matrix during
    training.  With add_flags=True, append the NLP flags to the embeddings. """
    def __init__(self, spad, glove, vocab, dropout, dropout_w, trainable=True, add_flags=True):
        s = SentenceInputs(99, glove.N, spad, add_flags=add_flags)
        inputs = [s.si, s.se]

        embmatrix = vocab.embmatrix(glove)
        emb = Embedding(embmatrix.shape[0], glove.N,  # mask_zero=True,
                        weights=[embmatrix], trainable=trainable,
                        dropout=dropout_w)
        si_emb = emb(s.si)
        s_emb = merge([si_emb, s.se], mode='sum')
        self.N = glove.N

        if add_flags:
            inputs.append(s.f)
            s_emb = merge([s_emb, s.f], mode='concat')
            self.N += nlp.flagsdim

        e = Dropout(dropout)(s_emb)

        self.model = Model(input=inputs, output=e)

    def __call__(self, s):
        assert s.si._keras_shape == self.model.inputs[0]._keras_shape, 'si %s != %s' % (s.si._keras_shape, self.model.inputs[0]._keras_shape)
        assert s.se._keras_shape == self.model.inputs[1]._keras_shape, 'se %s != %s' % (s.si._keras_shape, self.model.inputs[1]._keras_shape)
        return self.model(s.inputs)


class SentenceRNN(object):
    """ An RNN layer that takes a sequence of word embeddings as returned
    from WordsEmbedding and processes them using an RNN + dropout.

    If return_sequences=False, it returns just the final hidden state of the RNN;
    otherwise, it return a sequence of contextual token embeddings instead.
    At any rate, the output layers are e0s_, e1s_.

    If rnnlevels>1, a multi-level stacked RNN architecture like in Wang&Nyberg
    http://www.aclweb.org/anthology/P15-2116 is applied, however with skip-connections
    i.e. the inner RNNs have both the level above and original embeddings as inputs.
    """
    def __init__(self, spad, N, dropout=3/4, dropoutfix_inp=0, dropoutfix_rec=0,
                 sdim=2, rnnbidi=True, return_sequences=False,
                 rnn=GRU, rnnact='tanh', rnninit='glorot_uniform', rnnbidi_mode='sum',
                 rnnlevels=1):
        e = Input(shape=(spad, N))

        deep_input = e
        for i in range(1, rnnlevels):
            # XXX: is it okay to trim dropout to zero?
            upper_layer = SentenceRNN(N, dropout=0, dropoutfix_inp=dropoutfix_inp, dropoutfix_rec=dropoutfix_rec,
                                      sdim=sdim, rnnbidi=rnnbidi, return_sequences=True,
                                      rnn=rnn, rnnact=rnnact, rnninit=rnninit, rnnbidi_mode=rnnbidi_mode)
            deep_input = upper_layer(e)
            # skip-connections to layers below
            deep_input = merge([deep_input, e], mode='concat')

        if rnnbidi and rnnbidi_mode == 'concat':
            sdim /= 2
        self.N = int(N*sdim)
        rnn_layer = rnn(self.N, init=rnninit, activation=rnnact,
                        return_sequences=return_sequences,
                        dropout_W=dropoutfix_inp, dropout_U=dropoutfix_rec)
        if rnnbidi:
            rnn_layer = Bidirectional(rnn_layer, merge_mode=rnnbidi_mode)

        r = rnn_layer(e)
        r = Dropout(dropout)(r)

        self.model = Model(input=e, output=r)

    def __call__(self, x):
        assert x._keras_shape == self.model.inputs[0]._keras_shape, '%s != %s' % (x._keras_shape, self.model.inputs[0]._keras_shape)
        return self.model(x)


class SentenceCNN(object):
    """ An CNN pooling layer that takes sequence of word embeddings and
    processes them using a CNN + max-pooling to produce a single "summary
    embedding" (*NOT* a sequence of embeddings).

    The layer can apply multiple convolutions of different widths; the
    convolution dimensionality is denoted by the cdim dict, keyed by width
    and containing the number of filters.  The resulting summary embedding
    dimensionality is sum of N*cdim values (the convolutions are concatenated),
    returned by this function for your convenience.
    """
    def __init__(self, spad, N, dropout=3/4, l2reg=1e-4,
                 cnninit='glorot_uniform', cnnact='tanh',
                 cdim={1: 1/2, 2: 1/2, 3: 1/2, 4: 1/2, 5: 1/2}):
        e = Input(shape=(spad, N))

        self.N = 0
        aout = []
        for fl, cd in cdim.items():
            nb_filter = int(N*cd)
            aconv = Convolution1D(nb_filter=nb_filter, filter_length=fl,
                                  activation=cnnact, init=cnninit,
                                  W_regularizer=l2(l2reg))(e)
            apool = Flatten()(MaxPooling1D(pool_length=int(spad - fl + 1))(aconv))
            aout.append(apool)
            self.N += nb_filter

        out = merge(aout, mode='concat') if len(aout) > 1 else aout[0]
        out = Dropout(dropout)(out)

        self.model = Model(input=e, output=out)

    def __call__(self, x):
        assert x._keras_shape == self.model.inputs[0]._keras_shape, '%s != %s' % (x._keras_shape, self.model.inputs[0]._keras_shape)
        return self.model(x)


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
