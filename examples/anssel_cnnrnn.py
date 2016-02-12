#!/usr/bin/python3
"""
An Answer Sentence Selection classifier that follows Fariz Rahman's approach
popular in Keras examples of combining CNN + RNN for text classification.


Prerequisites:
    * Get glove.6B.300d.txt from http://nlp.stanford.edu/projects/glove/

Performance:
    * wang:  (the model parameters were tuned to maximize devMRR on wang, but
      only briefly)
      devMRR=0.853077, testMRR=0.770822, testMAP=0.7047
    * yodaqa:  (same parameters as on wang)
      valMRR=0.380500

"""

from __future__ import print_function
from __future__ import division

import argparse

from keras.callbacks import ModelCheckpoint
from keras.layers.convolutional import Convolution1D, MaxPooling1D
from keras.layers.core import Activation, Dense, Dropout
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


def cnn_input(model, N, spad, l2reg=1e-4,
              cnninit='glorot_uniform', cnnact='tanh',
              cdim=2, cfiltlen=3):
    """ A CNN pooling layer that takes sequence of embeddings e0_, e1_ and
    processes them using a CNN to produce a new sequence of fixed-length-context
    sensitive embeddings.  Returns output tensor shape.

    The output layers are e0c, e1c.
    """
    nb_filter = int(N*cdim)
    model.add_shared_node(name='aconv',
                          inputs=['e0_', 'e1_'], outputs=['e0c', 'e1c'],
                          layer=Convolution1D(input_shape=(spad, N),
                                              nb_filter=nb_filter, filter_length=cfiltlen,
                                              activation=cnnact, W_regularizer=l2(l2reg),
                                              init=cnninit))

    return (spad - cfiltlen + 1, nb_filter)


def prep_model(glove, vocab, dropout=1/2, dropout_in=4/5, l2reg=1e-4,
               cnnact='tanh', cnninit='glorot_uniform', cdim=1, cfiltlen=3,
               maxpool=True, maxpool_len=2,
               rnnbidi=True, rnn=GRU, rnnbidi_mode='sum', rnnact='tanh', rnninit='glorot_uniform', sdim=2,
               project=False, pdim=2.5,
               ptscorer=B.mlp_ptscorer, mlpsum='sum', Ddim=2,
               oact='sigmoid'):
    model = Graph()
    N = B.embedding(model, glove, vocab, s0pad, s1pad, dropout)

    if dropout_in is None:
        dropout_in = dropout

    (sc, Nc) = cnn_input(model, N, s0pad, l2reg=l2reg,
                         cnninit=cnninit, cnnact=cnnact, cdim=cdim, cfiltlen=cfiltlen)

    if maxpool:
        model.add_shared_node(name='pool', inputs=['e0c', 'e1c'], outputs=['e0g', 'e1g'],
                              layer=MaxPooling1D(pool_length=maxpool_len))
        sc /= maxpool_len
        cnn_outputs = ['e0g', 'e1g']
    else:
        cnn_outputs = ['e0c', 'e1c']
    model.add_node(name='e0c_', input=cnn_outputs[0], layer=Dropout(dropout))
    model.add_node(name='e1c_', input=cnn_outputs[1], layer=Dropout(dropout))

    B.rnn_input(model, Nc, sc, inputs=['e0c_', 'e1c_'],
                dropout=dropout_in, sdim=sdim,
                rnnbidi=rnnbidi, rnn=rnn, rnnact=rnnact, rnninit=rnninit,
                rnnbidi_mode=rnnbidi_mode)

    # Projection
    if project:
        model.add_shared_node(name='proj', inputs=['e0s_', 'e1s_'], outputs=['e0p', 'e1p'],
                              layer=Dense(input_dim=int(N*sdim), output_dim=int(N*pdim), W_regularizer=l2(l2reg)))
        # model.add_shared_node(name='projdrop', inputs=['e0p', 'e1p'], outputs=['e0p_', 'e1p_'],
        #                       layer=Dropout(dropout_in, input_shape=(N,)))
        # final_outputs = ['e0p_', 'e1p_']
        final_outputs = ['e0p', 'e1p']
    else:
        final_outputs = ['e0s_', 'e1s_']

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
                         ModelCheckpoint('weights-cnnrnn-bestval.h5', save_best_only=True, monitor='mrr', mode='max')],
              batch_size=160, nb_epoch=16, samples_per_epoch=5000)
    model.save_weights('weights-cnnrnn-final.h5', overwrite=True)
    ev.eval_anssel(model.predict(gr)['score'][:,0], s0, y, 'Train')
    ev.eval_anssel(model.predict(grt)['score'][:,0], s0t, yt, 'Val')
