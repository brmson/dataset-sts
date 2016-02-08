"""
Predefined Keras Graph blocks that represent common model components.
"""

from keras.layers.core import Activation, Dense, Dropout
from keras.layers.embeddings import Embedding

import pysts.nlp as nlp


def embedding(model, glove, vocab, s0pad, s1pad, dropout, trainable=True):
    """ The universal sequence input layer.

    Declare inputs si0, si1, f0, f1 (vectorized sentences and NLP flags)
    and generate outputs e0, e1 representing vector sequences, and e0_, e1_
    with dropout applied.  Returns the vector dimensionality. """

    for m, p in [(0, s0pad), (1, s1pad)]:
        model.add_input('si%d'%(m,), input_shape=(p,), dtype='int')
        model.add_input('f%d'%(m,), input_shape=(p, nlp.flagsdim))
    model.add_shared_node(name='emb', inputs=['si0', 'si1'], outputs=['e0[0]', 'e1[0]'],
                          layer=Embedding(input_dim=vocab.size(), input_length=p,
                                          output_dim=glove.N, mask_zero=True,
                                          weights=[vocab.embmatrix(glove)], trainable=trainable))
    for m in [0, 1]:
        model.add_node(name='e%d'%(m,), inputs=['e%d[0]'%(m,), 'f%d'%(m,)], merge_mode='concat', layer=Activation('linear'))
    N = glove.N + nlp.flagsdim

    model.add_shared_node(name='embdrop', inputs=['e0', 'e1'], outputs=['e0_', 'e1_'],
                          layer=Dropout(dropout, input_shape=(N,)))

    return N


# Match point scoring (scalar output) callables.  Each returns the layer name.
# This is primarily meant as an output layer, but could be used also for
# example as an attention mechanism.

def dot_ptscorer(model, inputs, Ddim, N, l2reg, pfx='out'):
    """ Score the pair using just dot-product, that is elementwise
    multiplication and sum.  The dot-product is natural because it
    measures the relative directions of vectors, being essentially
    a non-normalized cosine similarity. """
    # (The Activation is a nop, merge_mode is the important part)
    model.add_node(name=pfx+'dot', inputs=inputs, layer=Activation('linear'), merge_mode='dot', dot_axes=1)
    return pfx+'dot'


def mlp_ptscorer(model, inputs, Ddim, N, l2reg, pfx='out'):
    """ Element-wise features from the pair fed to an MLP. """
    model.add_node(name=pfx+'sum', inputs=inputs, layer=Activation('linear'), merge_mode='sum')
    model.add_node(name=pfx+'mul', inputs=inputs, layer=Activation('linear'), merge_mode='mul')

    model.add_node(name=pfx+'hdn', inputs=[pfx+'sum', pfx+'mul'], merge_mode='concat',
                   layer=Dense(output_dim=int(N*Ddim), W_regularizer=l2(l2reg), activation='sigmoid'))
    model.add_node(name=pfx+'mlp', input=pfx+'hdn',
                   layer=Dense(output_dim=1, W_regularizer=l2(l2reg)))
    return pfx+'mlp'


def cat_ptscorer(model, inputs, Ddim, N, l2reg, pfx='out'):
    """ Just train a linear classifier (weighed sum of elements) on concatenation
    of inputs.  You may pass also just a single input (which may make sense
    if you for example process s1 "with regard to s0"). """
    model.add_node(name=pfx+'cat', inputs=inputs, merge_mode='concat',
                   layer=Dense(output_dim=1, W_regularizer=l2(l2reg)))
