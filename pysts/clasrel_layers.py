from keras.layers.wrappers import TimeDistributed as td

from keras import activations, initializations
import keras.constraints
import keras.regularizers
from keras.layers.core import MaskedLayer, Layer, TimeDistributedDense, TimeDistributedMerge, Activation
import keras.backend as K
import theano.tensor as T
import numpy as np

def relu(x):
    return K.switch(x > 0, x + 0.01, 0.01)


class WeightedMean(MaskedLayer):

    input_ndim = 3

    def __init__(self, w_dim, q_dim, max_sentences=100, output_dim=1,
                 activation='linear', **kwargs):
        self.max_sentences = max_sentences
        self.w_dim = w_dim
        self.q_dim = q_dim
        self.input_dim = self.w_dim + self.q_dim
        self.activation = activations.get(activation)
        self.output_dim = output_dim

        kwargs['input_shape'] = (self.max_sentences, self.w_dim + self.q_dim,)
        super(WeightedMean, self).__init__(**kwargs)

    def build(self):
        pass

    @property
    def output_shape(self):
        input_shape = self.input_shape
        return (input_shape[0], input_shape[1], 1)

    def get_output(self, train=False):
        e = 1e-6  # constant used for numerical stability
        X = self.get_input(train)
        x = K.reshape(X, (-1, self.input_shape[-1]))
        mask = 1#x[:, 2]
        f = x[:, 0] * mask
        r = x[:, 1] * mask

        # s_ = K.dot(f, self.W)
        # t_ = K.dot(r, self.Q)
        # mask = K.switch(s_, 1, 0)
        # s = self.activation_w(s_ + self.w[0]) * mask
        # t = self.activation_q(t_ + self.q[0]) * mask
        s = K.reshape(f, (-1, self.input_shape[1]))
        t = K.reshape(r, (-1, self.input_shape[1]))

        output = self.activation(K.sum(s * t, axis=1) / (T.sum(t, axis=-1)) + e)
        output = K.reshape(output, (-1, 1))
        return output

    def get_config(self):
        config = {'name': self.__class__.__name__,
                  'output_dim': self.output_dim,
                  'activation': self.activation.__name__,
                  'input_dim': self.input_dim}
        base_config = super(WeightedMean, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

    def fill_regulizers(self):
        regularizers = []
        if self.W_regularizer:
            self.W_regularizer.set_param(self.W)
            regularizers.append(self.W_regularizer)

        if self.w_regularizer:
            self.w_regularizer.set_param(self.w)
            regularizers.append(self.w_regularizer)

        if self.Q_regularizer:
            self.Q_regularizer.set_param(self.Q)
            regularizers.append(self.Q_regularizer)

        if self.q_regularizer:
            self.q_regularizer.set_param(self.q)
            regularizers.append(self.q_regularizer)

        return regularizers


class Reshape_(MaskedLayer):
    """Copy of keras core Reshape layer, does NOT check
    if array changes size.
    """
    def __init__(self, dims, **kwargs):
        super(Reshape_, self).__init__(**kwargs)
        self.dims = tuple(dims)

    def _fix_unknown_dimension(self, input_shape, output_shape):

        output_shape = list(output_shape)

        known, unknown = 1, None
        for index, dim in enumerate(output_shape):
            if dim < 0:
                if unknown is None:
                    unknown = index
                else:
                    raise ValueError('can only specify one unknown dimension')
            else:
                known *= dim

        return tuple(output_shape)

    @property
    def output_shape(self):
        return (self.input_shape[0],) + self._fix_unknown_dimension(self.input_shape[1:], self.dims)

    def get_output(self, train=False):
        X = self.get_input(train)
        return K.reshape(X, (-1,) + self.output_shape[1:])

    def get_config(self):
        config = {'name': self.__class__.__name__,
                  'dims': self.dims}
        base_config = super(Reshape_, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


class SumMask(Layer):
    """Copy of keras core Reshape layer, does NOT check
    if array changes size.
    """
    def __init__(self, **kwargs):
        super(SumMask, self).__init__(**kwargs)

    @property
    def output_shape(self):
        return (self.input_shape[0], self.input_shape[1], 1)

    def get_output(self, train=False):
        X = self.get_input(train)
        return K.expand_dims(K.switch(K.sum(X, -1), 1, 0))

    def get_config(self):
        config = {'name': self.__class__.__name__,
                  'input_dim': self.input_dim,
                  'output_dim': self.output_dim}
        base_config = super(SumMask, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))
