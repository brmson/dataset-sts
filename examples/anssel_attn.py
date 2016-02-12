#!/usr/bin/python3
"""
An Answer Sentence Selection classifier that aims to reproduce the attention-based
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


Prerequisites:
    * Get glove.6B.300d.txt from http://nlp.stanford.edu/projects/glove/

Performance:
    * wang:  (*BEFORE* any parameter tuning, just estimates from simpler models)
      devMRR=0.888205, testMRR=0.836141, testMAP=0.7490

      This beats state-of-art as of Jan 2016:
      http://www.aclweb.org/aclwiki/index.php?title=Question_Answering_(State_of_the_art)

    * yodaqa:  (same parameters as wang)
      valMRR=0.451608

"""

from __future__ import print_function
from __future__ import division

import argparse

from keras.callbacks import ModelCheckpoint
from keras.layers.convolutional import Convolution1D, MaxPooling1D, AveragePooling1D
from keras.layers.core import Activation, Dense, Dropout, Flatten, Permute, RepeatVector, TimeDistributedDense
from keras.layers.recurrent import SimpleRNN, GRU, LSTM
from keras.models import Graph
from keras.regularizers import l2

import pysts.embedding as emb
import pysts.eval as ev
import pysts.loader as loader
import pysts.nlp as nlp
from pysts.vocab import Vocabulary

from pysts.kerasts import graph_input_anssel
import pysts.kerasts.blocks as B
from pysts.kerasts.callbacks import AnsSelCB
from pysts.kerasts.objectives import ranknet


s0pad = 60
s1pad = 60


def load_set(fname, vocab=None):
    s0, s1, y, t = loader.load_anssel(fname)

    if vocab is None:
        vocab = Vocabulary(s0 + s1)

    si0 = vocab.vectorize(s0)
    si1 = vocab.vectorize(s1)
    f0, f1 = nlp.sentence_flags(s0, s1, s0pad, s1pad)
    gr = graph_input_anssel(si0, si1, y, f0, f1)

    return (s0, s1, y, vocab, gr)


def pool(model, e_in, e_out, nsteps, width, pool_layer, dropout):
    model.add_node(name=e_out+'[0]', input=e_in,
                   layer=pool_layer(pool_length=nsteps))
    model.add_node(name=e_out+'[1]', input=e_out+'[0]',
                   layer=Flatten(input_shape=(1, width)))
    model.add_node(name=e_out, input=e_out+'[1]',
                   layer=Dropout(dropout))


def aggregate(model, e_in, pfx, N, spad, pool_layer, dropout, l2reg, sdim, cnnact, cdim, cfiltlen, project):
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


def prep_model(glove, vocab, dropout=3/4, dropout_in=None, l2reg=1e-4,
               rnnbidi=True, rnn=GRU, rnnbidi_mode='sum', rnnact='tanh', rnninit='glorot_uniform',
               sdim=2, rnnlevels=1,
               pool_layer=MaxPooling1D, cnnact='tanh', cnninit='glorot_uniform', cdim=2, cfiltlen=3,
               project=True, adim=1/2, attn_mode='sum', fact='softmax',
               ptscorer=B.mlp_ptscorer, mlpsum='sum', Ddim=2,
               oact='sigmoid'):
    model = Graph()
    N = B.embedding(model, glove, vocab, s0pad, s1pad, dropout)

    if dropout_in is None:
        dropout_in = dropout

    # FIXME: pool_layer=None is in fact not supported, since this RNN
    # would return a scalar for e1s too; instead, we'l need to manually
    # pick the first&last element of the returned sequence from e0s
    B.rnn_input(model, N, s0pad, return_sequences=(pool_layer is not None),
                rnnlevels=rnnlevels, dropout=dropout_in, sdim=sdim,
                rnnbidi=rnnbidi, rnnbidi_mode=rnnbidi_mode,
                rnn=rnn, rnnact=rnnact, rnninit=rnninit)

    # Generate e0s aggregate embedding
    e0_aggreg, gwidth = aggregate(model, 'e0s_', 'e0', N, s0pad, pool_layer,
                                  dropout=dropout_in, l2reg=l2reg, sdim=sdim,
                                  cnnact=cnnact, cdim=cdim, cfiltlen=cfiltlen,
                                  project=project)

    if project:
        # ...and re-embed e0, e1 in attention space
        awidth = int(N*adim)
        model.add_node(name='e0a', input=e0_aggreg,
                       layer=Dense(input_dim=gwidth, output_dim=awidth, W_regularizer=l2(l2reg)))
        e0_aggreg_attn = 'e0a'

        model.add_node(name='e1sa_', input='e1s',
                       layer=TimeDistributedDense(input_dim=int(N*sdim), output_dim=awidth, W_regularizer=l2(l2reg)))
        # XXX: this dummy works around a mysterious theano error
        model.add_node(name='e1sa', input='e1sa_', layer=Activation('linear'))
        e1_attn = 'e1sa'
    else:
        e1_attn = 'e1s'
        e0_aggreg_attn = e0_aggreg

    # Now, build an attention function f(e0a, e1sa) -> e1a, producing an
    # (s1pad,) vector of scalars denoting the attention for each e1 token
    model.add_node(name='e0sa', input=e0_aggreg_attn,
                   layer=RepeatVector(s1pad))
    if attn_mode == 'dot':
        # model attention by dot-product, i.e. similarity measure of question
        # aggregate and answer token in attention space
        model.add_node(name='e1a[1]',
                       layer=B.dot_time_distributed_merge(model, ['e0sa', e1_attn]))
    else:
        # traditional attention model from Hermann et al., 2015 and Tan et al., 2015
        # we want to model attention as w*tanh(e0a + e1sa[i])
        model.add_node(name='e1a[0]', inputs=['e0sa', e1_attn], merge_mode='sum',
                       layer=Activation('tanh'))
        model.add_node(name='e1a[1]', input='e1a[0]',
                       layer=TimeDistributedDense(input_dim=awidth, output_dim=1, W_regularizer=l2(l2reg)))
    model.add_node(name='e1a[2]', input='e1a[1]',
                   layer=Flatten(input_shape=(s1pad, 1)))

    # *Focus* e1 by softmaxing (by default) attention and multiplying tokens
    # by their attention.
    model.add_node(name='e1a[3]', input='e1a[2]',
                   layer=Activation(fact))
    model.add_node(name='e1a[4]', input='e1a[3]',
                   layer=RepeatVector(int(N*sdim)))
    model.add_node(name='e1a', input='e1a[4]',
                   layer=Permute((2,1)))
    model.add_node(name='e1sm', inputs=['e1s_', 'e1a'], merge_mode='mul',
                   layer=Activation('linear'))

    # Generate e1sm aggregate embedding
    e1_aggreg, gwidth = aggregate(model, 'e1sm', 'e1', N, s1pad, pool_layer,
                                  dropout=dropout_in, l2reg=l2reg, sdim=sdim,
                                  cnnact=cnnact, cdim=cdim, cfiltlen=cfiltlen,
                                  project=project)

    if ptscorer == '1':
        # special scoring mode just based on the answer
        # (assuming that the question match is carried by the attention)
        ptscorer = B.cat_ptscorer
        final_outputs = [e1_aggreg]
    else:
        final_outputs = [e0_aggreg, e1_aggreg]

    # Measurement
    kwargs = dict()
    if ptscorer == B.mlp_ptscorer:
        kwargs['sum_mode'] = mlpsum
    model.add_node(name='scoreS', input=ptscorer(model, final_outputs, Ddim, N, l2reg, **kwargs),
                   layer=Activation(oact))
    model.add_output(name='score', input='scoreS')
    return model


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Benchmark CNN+RNN on a bipartite ranking task (answer selection)")
    parser.add_argument("-N", help="GloVe dim", type=int, default=300)
    parser.add_argument("--wang", help="whether to run on Wang inst. of YodaQA dataset", type=int, default=0)
    parser.add_argument("--params", help="additional training parameters", type=str, default='')
    args = parser.parse_args()

    glove = emb.GloVe(N=args.N)
    if args.wang == 1:
        s0, s1, y, vocab, gr = load_set('anssel-wang/train-all.csv')
        s0t, s1t, yt, _, grt = load_set('anssel-wang/dev.csv', vocab)
    else:
        s0, s1, y, vocab, gr = load_set('anssel-yodaqa/curatedv1-training.csv')
        s0t, s1t, yt, _, grt = load_set('anssel-yodaqa/curatedv1-val.csv', vocab)

    kwargs = eval('dict(' + args.params + ')')
    model = prep_model(glove, vocab, oact='linear', **kwargs)
    model.compile(loss={'score': ranknet}, optimizer='adam')  # for 'binary_crossentropy', drop the custom oact
    model.fit(gr, validation_data=grt,
              callbacks=[AnsSelCB(s0t, grt),
                         ModelCheckpoint('weights-attn-bestval.h5', save_best_only=True, monitor='mrr', mode='max')],
              batch_size=160, nb_epoch=16, samples_per_epoch=5000)
    model.save_weights('weights-attn-final.h5', overwrite=True)
    ev.eval_anssel(model.predict(gr)['score'][:,0], s0, y, 'Train')
    ev.eval_anssel(model.predict(grt)['score'][:,0], s0t, yt, 'Val')
