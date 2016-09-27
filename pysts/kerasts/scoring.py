"""
Predefined Keras model blocks that "point score" pairs of sentence
embeddings, i.e. producing a real value from two N-length vectors
summarizing the sentence meaning.

The output value is deliberately non-normalized since for ranking
purposes, not applying a sigmoid works better than applying it.
"""

from __future__ import division
from __future__ import print_function

from keras.layers import Dense, Input, merge
from keras.models import Model
from keras.regularizers import l2


class MergePtScorer(object):
    """ Produce score as a simple combination of inputs, first
    optionally projecting them to a merge-friendly linear space. """
    def __init__(self, N, l2reg, merge_mode, pdim=0, extra_inp=[]):
        se0, se1 = Input(shape=(N,)), Input(shape=(N,))
        e0, e1 = se0, se1
        extra_inp_var = [Input(e.shape) for e in extra_inp]

        # Projection
        if pdim > 0:
            proj = Dense(int(N*pdim), W_regularizer=l2(l2reg))
            e0, e1 = proj(e0), proj(e1)

        score = merge([e0, e1], mode=merge_mode, dot_axes=1)
        if extra_inp_var:
            score_hidden = merge([score] + extra_inp_var, mode='concat')
            score = Dense(1, W_regularizer=l2(l2reg))(score_hidden)

        self.model = Model(input=[se0, se1] + extra_inp_var, output=score)

    def __call__(self, se0, se1, extra_inp=[]):
        assert se0._keras_shape == self.model.inputs[0]._keras_shape, 'se0 %s != %s' % (se0._keras_shape, self.model.inputs[0]._keras_shape)
        assert se1._keras_shape == self.model.inputs[1]._keras_shape, 'se1 %s != %s' % (se1._keras_shape, self.model.inputs[1]._keras_shape)
        return self.model([se0, se1] + extra_inp)


class DotPtScorer(MergePtScorer):
    """ Score the pair using just dot-product, that is elementwise
    multiplication and then sum.  The dot-product is natural because it
    measures the relative directions of vectors, being essentially
    a non-normalized cosine similarity. """
    def __init__(self, N, l2reg, pdim=0, extra_inp=[]):
        super(N, l2reg, 'dot', pdim=pdim, extra_inp=extra_inp)


class CosPtScorer(MergePtScorer):
    """ Score the pair using just cosine similarity. """
    def __init__(self, N, l2reg, pdim=0, extra_inp=[]):
        super(N, l2reg, 'cos', pdim=pdim, extra_inp=extra_inp)


class CatPtScorer(MergePtScorer):
    """ Just train a linear classifier (weighed sum of elements) on concatenation
    of inputs. """
    def __init__(self, N, l2reg, pdim=0, extra_inp=[]):
        super(N, l2reg, 'concat', pdim=pdim, extra_inp=extra_inp)


class MLPPtScorer(object):
    """ Element-wise features (element-wise sum/absdiff and element-wise
    product) from the pair fed to an MLP. """
    def __init__(self, N, l2reg, pdim=0, Ddim=1, Dinit='glorot_uniform', sum_mode='sum', extra_inp=[]):
        se0, se1 = Input(shape=(N,)), Input(shape=(N,))
        e0, e1 = se0, se1
        extra_inp_var = [Input(e.shape) for e in extra_inp]

        # Projection
        if pdim > 0:
            proj = Dense(int(N*pdim), W_regularizer=l2(l2reg))
            e0, e1 = proj(e0), proj(e1)

        assert sum_mode == 'sum', 'TODO absdiff'
        e_sum = merge([e0, e1], mode='sum')
        e_mul = merge([e0, e1], mode='mul')
        mlp_input = merge([e_sum, e_mul] + extra_inp_var, mode='concat')

        if Ddim == 0:
            Ddim = []
        elif not isinstance(Ddim, list):
            Ddim = [Ddim]

        for i, D in enumerate(Ddim):
            mlp_input = Dense(int(N*D), W_regularizer=l2(l2reg), activation='tanh', init=Dinit)(mlp_input)

        score = Dense(1, W_regularizer=l2(l2reg))(mlp_input)

        self.se0 = se0
        self.model = Model(input=[se0, se1] + extra_inp_var, output=score)

    def __call__(self, se0, se1, extra_inp=[]):
        assert se0._keras_shape == self.model.inputs[0]._keras_shape, 'se0 %s != %s' % (se0._keras_shape, self.model.inputs[0]._keras_shape)
        assert se1._keras_shape == self.model.inputs[1]._keras_shape, 'se1 %s != %s' % (se1._keras_shape, self.model.inputs[1]._keras_shape)
        return self.model([se0, se1] + extra_inp)


class S1PtScorer(object):
    """ Score just based on s1 embedding.  This may make sense if your
    s1 embedding depends on s0 embedding e.g. via attention.  """
    def __init__(self, N, l2reg, Ddim=0, pdim=0, extra_inp=[]):
        se = Input(shape=(N,))
        e = se
        extra_inp_var = [Input(ex.shape) for ex in extra_inp]

        # Projection
        if pdim > 0:
            proj = Dense(int(N*pdim), W_regularizer=l2(l2reg))
            e = proj(e)

        if extra_inp_var:
            mlp_input = merge([e] + extra_inp_var, mode='concat')
        else:
            mlp_input = e

        if Ddim == 0:
            Ddim = []
        elif not isinstance(Ddim, list):
            Ddim = [Ddim]
        for i, D in enumerate(Ddim):
            mlp_input = Dense(int(N*D), W_regularizer=l2(l2reg), activation='tanh')(mlp_input)

        score = Dense(1, W_regularizer=l2(l2reg))(mlp_input)

        self.model = Model(input=[se] + extra_inp_var, output=score)

    def __call__(self, se0, se1, extra_inp=[]):
        assert se1._keras_shape == self.model.inputs[0]._keras_shape, '%s != %s' % (se1._keras_shape, self.model.inputs[0]._keras_shape)
        return self.model([se1] + extra_inp)
