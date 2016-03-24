"""
A model that aims to reproduce the attention-based
RNN/CNN combined architecture of http://arxiv.org/abs/1511.04108.

The architecture uses gated RNN for input preprocessing, CNN to generate
aggregate sentence representations, perceptron-softmax to focus attention within
answer sentence based on the question aggregate, and MLP with elementwise dot/sum
for final classification.

There are differences to the original paper, some of them more important, others
less so:

    * Word embedding matrix is preinitialized with 300D GloVe and is adaptable
      during training

    * Tokens are annotated by trivial sentence overlap and lexical features and
      these features are appended to their embedidngs

    * Dropout is heavier than usual in such models

    * Bidi-RNN uses sum instead of concatenation to merge directions

    * GRU is used instead of LSTM

    * The Ranknet loss function is used as an objective

    * Output is a multi-layer perceptron with input inspired by
      Tai et al., 2015 http://arxiv.org/abs/1503.00075 rather than cosinesim

    * There ought to be a lot of differences in numeric parameters, in some
      cases possibly significant (in original, cdim=~10)

Plus, a lot of the components are optional and easy to disable or tweak here.
Go wild!


This will be a part of our upcoming paper; meanwhile, if you need to cite this,
refer to the dataset-sts GitHub repo, please.

Performance:
    * anssel-wang:  (*BEFORE* any parameter tuning, just estimates from simpler models)
      devMRR=0.888205, testMRR=0.836141, testMAP=0.7490

      This beats state-of-art as of Jan 2016:
      http://www.aclweb.org/aclwiki/index.php?title=Question_Answering_(State_of_the_art)

    * anssel-yodaqa:  (same parameters as wang)
      valMRR=0.451608

"""

from __future__ import print_function
from __future__ import division

import keras.activations as activations
from keras.layers.convolutional import Convolution1D, MaxPooling1D, AveragePooling1D
from keras.layers.core import Activation, Dense, Dropout, Flatten, MaskedLayer, Permute, RepeatVector, TimeDistributedDense
from keras.layers.recurrent import SimpleRNN, GRU, LSTM
from keras.regularizers import l2

import pysts.kerasts.blocks as B


def config(c):
    c['dropout'] = 1/2
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

    c['pool_layer'] = MaxPooling1D
    c['cnnact'] = 'tanh'
    c['cnninit'] = 'glorot_uniform'
    c['cdim'] = 2
    c['cfiltlen'] = 3

    c['project'] = True
    c['adim'] = 1/2
    c['attn_mode'] = 'sum'
    c['focus_act'] = 'softmax'

    # model-external:
    c['inp_e_dropout'] = 1/2
    c['inp_w_dropout'] = 0
    # anssel-specific:
    c['ptscorer'] = B.mlp_ptscorer
    c['mlpsum'] = 'sum'
    c['Ddim'] = 2


def pool(model, e_in, e_out, nsteps, width, pool_layer, dropout):
    model.add_node(name=e_out+'[0]', input=e_in,
                   layer=pool_layer(pool_length=nsteps))
    model.add_node(name=e_out+'[1]', input=e_out+'[0]',
                   layer=Flatten(input_shape=(1, width)))
    model.add_node(name=e_out, input=e_out+'[1]',
                   layer=Dropout(dropout))


def aggregate(model, e_in, pfx, N, spad, pool_layer,
              dropout, l2reg, sdim, cnnact, cdim, cfiltlen, project):
    if pool_layer is None:
        return (e_in, int(N*sdim))

    if cnnact is not None:
        if not project:
            # force cdim <- sdim so that e0c is compatible with attention space
            cdim = sdim
        model.add_node(name=pfx+'c', input=e_in,
                       layer=Convolution1D(input_shape=(spad, int(N*sdim)),
                                nb_filter=int(N*cdim), filter_length=cfiltlen,
                                activation=cnnact, W_regularizer=l2(l2reg)))
        nsteps = spad - cfiltlen + 1
        width = int(N*cdim)
        pool(model, pfx+'c', pfx+'g', nsteps, width, pool_layer, dropout=dropout)
    else:
        (nsteps, width) = (spad, int(N*sdim))
        pool(model, e_in, pfx+'g', nsteps, width, pool_layer, dropout=0)
    return (pfx+'g', width)


class NormalizedActivation(MaskedLayer):
    '''Apply an activation function to an output, normalizing the output.

    This is like softmax for other activations than exp.

    # Input shape
        Arbitrary. Use the keyword argument `input_shape`
        (tuple of integers, does not include the samples axis)
        when using this layer as the first layer in a model.

    # Output shape
        Same shape as input.

    # Arguments:
        activation: name of activation function to use
            (see: [activations](../activations.md)),
            or alternatively, a Theano or TensorFlow operation.
    '''
    def __init__(self, activation, norm_mode, **kwargs):
        super(NormalizedActivation, self).__init__(**kwargs)
        self.activation = activations.get(activation)
        self.norm_mode = norm_mode

    def get_output(self, train=False):
        import keras.backend as K
        X = self.get_input(train)
        a = self.activation(X)
        if self.norm_mode == 'norm':
            s = K.sum(a, axis=-1, keepdims=True)
        elif self.norm_mode == 'maxnorm':
            s = K.max(a, axis=-1, keepdims=True)
        else:
            raise ValueError
        return a / s

    def get_config(self):
        config = {'name': self.__class__.__name__,
                  'activation': self.activation.__name__}
        base_config = super(NormalizedActivation, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


def focus_activation(focus_act):
    """ .../norm: normalize activation to sum to 1;
    .../maxnorm: normalize activation to peak at 1 """
    if '/' in focus_act:
        focus_act, norm_mode = focus_act.split('/')
        return NormalizedActivation(focus_act, norm_mode)
    else:
        return Activation(focus_act)


def focus(model, N, input_aggreg, input_seq, orig_seq, attn_name, output_name,
          s1pad, sdim, awidth, attn_mode, focus_act, l2reg):
    model.add_node(name=input_aggreg+'[rep]', input=input_aggreg,
                   layer=RepeatVector(s1pad))
    if attn_mode == 'dot' or attn_mode == 'cos':
        # model attention by dot-product, i.e. similarity measure of question
        # aggregate and answer token in attention space
        model.add_node(name=attn_name+'[1]',
                       layer=B.dot_time_distributed_merge(model, [input_aggreg+'[rep]', input_seq],
                                                          cos_norm=(attn_mode == 'cos')))
    else:
        # traditional attention model from Hermann et al., 2015 and Tan et al., 2015
        # we want to model attention as w*tanh(e0a + e1sa[i])
        model.add_node(name=attn_name+'[0]', inputs=[input_aggreg+'[rep]', input_seq], merge_mode='sum',
                       layer=Activation('tanh'))
        model.add_node(name=attn_name+'[1]', input=attn_name+'[0]',
                       layer=TimeDistributedDense(input_dim=awidth, output_dim=1, W_regularizer=l2(l2reg)))
    model.add_node(name=attn_name+'[2]', input=attn_name+'[1]',
                   layer=Flatten(input_shape=(s1pad, 1)))

    # *Focus* e1 by softmaxing (by default) attention and multiplying tokens
    # by their attention.
    model.add_node(name=attn_name+'[3]', input=attn_name+'[2]',
                   layer=focus_activation(focus_act))
    model.add_node(name=attn_name+'[4]', input=attn_name+'[3]',
                   layer=RepeatVector(int(N*sdim)))
    model.add_node(name=attn_name, input=attn_name+'[4]',
                   layer=Permute((2,1)))
    model.add_node(name=output_name, inputs=[orig_seq, attn_name], merge_mode='mul',
                   layer=Activation('linear'))


def prep_model(model, N, s0pad, s1pad, c):
    # FIXME: pool_layer=None is in fact not supported, since this RNN
    # would return a scalar for e1s too; instead, we'l need to manually
    # pick the first&last element of the returned sequence from e0s
    B.rnn_input(model, N, s0pad, return_sequences=(c['pool_layer'] is not None),
                dropout=c['dropout'], dropoutfix_inp=c['dropoutfix_inp'], dropoutfix_rec=c['dropoutfix_rec'],
                sdim=c['sdim'], rnnlevels=c['rnnlevels'],
                rnnbidi=c['rnnbidi'], rnnbidi_mode=c['rnnbidi_mode'],
                rnn=c['rnn'], rnnact=c['rnnact'], rnninit=c['rnninit'])

    # Generate e0s aggregate embedding
    e0_aggreg, gwidth = aggregate(model, 'e0s_', 'e0', N, s0pad, c['pool_layer'],
                                  dropout=c['dropout'], l2reg=c['l2reg'], sdim=c['sdim'],
                                  cnnact=c['cnnact'], cdim=c['cdim'], cfiltlen=c['cfiltlen'],
                                  project=c['project'])

    if c['project']:
        # ...and re-embed e0, e1 in attention space
        awidth = int(N*c['adim'])
        model.add_node(name='e0a', input=e0_aggreg,
                       layer=Dense(input_dim=gwidth, output_dim=awidth, W_regularizer=l2(c['l2reg'])))
        e0_aggreg_attn = 'e0a'

        model.add_node(name='e1sa_', input='e1s',
                       layer=TimeDistributedDense(input_dim=int(N*c['sdim']), output_dim=awidth, W_regularizer=l2(c['l2reg'])))
        # XXX: this dummy works around a mysterious theano error
        model.add_node(name='e1sa', input='e1sa_', layer=Activation('linear'))
        e1_attn = 'e1sa'
    else:
        awidth = int(N*c['sdim'])
        e1_attn = 'e1s'
        e0_aggreg_attn = e0_aggreg

    # Now, build an attention function f(e0a, e1sa) -> e1a, producing an
    # (s1pad,) vector of scalars denoting the attention for each e1 token
    focus(model, N, e0_aggreg_attn, e1_attn, 'e1s_', 'e1a', 'e1sm', s1pad, c['sdim'], awidth,
          c['attn_mode'], c['focus_act'], c['l2reg'])

    # Generate e1sm aggregate embedding
    e1_aggreg, gwidth = aggregate(model, 'e1sm', 'e1', N, s1pad, c['pool_layer'],
                                  dropout=c['dropout'], l2reg=c['l2reg'], sdim=c['sdim'],
                                  cnnact=c['cnnact'], cdim=c['cdim'], cfiltlen=c['cfiltlen'],
                                  project=c['project'])

    return (e0_aggreg, e1_aggreg)
