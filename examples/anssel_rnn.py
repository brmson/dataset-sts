#!/usr/bin/python3
"""
An Answer Sentence Selection classifier that uses full-fledged features
of the pysts Keras toolkit (KeraSTS) and even with a very simple architecture
pretty closely approaches 2015-state-of-art results on the task.

The architecture uses shared bidirectional GRU to produce sentence embeddings,
adaptable word embedding matrix preinitialized with 300D GloVe, projection
matrix (MemNN-like - applied to both sentences to project them to a common
external similarity space) and MLP (TODO: cite) similarity measure.

Rather than relying on the hack of using the word overlap counts as additional
features for final classification, individual tokens are annotated by overlap
features and that's passed to the GRU along with the embeddings.

The Ranknet loss function is used as an objective, instead of binary
crossentropy.

This will be a part of our upcoming paper; meanwhile, if you need to cite this,
refer to the dataset-sts GitHub repo, please.


Prerequisites:
    * Get glove.6B.300d.txt from http://nlp.stanford.edu/projects/glove/

Performance:
    * wang:  (the model parameters were tuned to maximize devMRR on wang)
      * project=False, dot_ptscorer
                rnnbidi=False - devMRR=0.773352, testMRR=0.745151
                rnnbidi=True - devMRR=0.796154, testMRR=0.774527

      * project=True, dot_ptscorer (no dropout after project)
                rnnbidi=False - devMRR=0.818654, testMRR=0.709342
                rnnbidi=True - devMRR=0.840403, testMRR=0.762395

      * project=False, mlp_ptscorer
                rnnbidi=False - devMRR=0.829890, testMRR=0.756679
                rnnbidi=True - devMRR=0.869744, testMRR=0.787051

      * project=True, mlp_ptscorer (no dropout after project)
                rnnbidi=False - devMRR=0.844359, testMRR=0.813130, testMAP=0.7249
                rnnbidi=True - devMRR=0.857949, testMRR=0.797496, testMAP=0.7275

      * project=True, mlp_ptscorer (CURRENT)
                rnnbidi=False - devMRR=0.830641, testMRR=0.797059, testMAP=0.7097
                rnnbidi=True - devMRR=0.833887, testMRR=0.808252, testMAP=0.7262

"""

from __future__ import print_function
from __future__ import division

import argparse

from keras.callbacks import ModelCheckpoint
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


def prep_model(glove, vocab, dropout=3/4, dropout_in=None, l2reg=1e-4,
               rnnbidi=True, rnn=GRU, rnnact='tanh', rnninit='glorot_uniform', sdim=2,
               project=True, pdim=2.5,
               ptscorer=B.mlp_ptscorer, Ddim=2,
               oact='sigmoid'):
    model = Graph()
    N = B.embedding(model, glove, vocab, s0pad, s1pad, dropout)

    if dropout_in is None:
        dropout_in = dropout

    B.rnn_input(model, N, s0pad, dropout=dropout_in, sdim=sdim,
                rnnbidi=rnnbidi, rnn=rnn, rnnact=rnnact, rnninit=rnninit)

    # Projection
    if project:
        model.add_shared_node(name='proj', inputs=['e0s_', 'e1s_'], outputs=['e0p', 'e1p'],
                              layer=Dense(input_dim=int(N*sdim), output_dim=int(N*pdim), W_regularizer=l2(l2reg)))
        model.add_shared_node(name='projdrop', inputs=['e0p', 'e1p'], outputs=['e0p_', 'e1p_'],
                              layer=Dropout(dropout_in, input_shape=(N,)))
        final_outputs = ['e0p_', 'e1p_']
    else:
        final_outputs = ['e0s_', 'e1s_']

    # Measurement
    model.add_node(name='scoreS', input=ptscorer(model, final_outputs, Ddim, N, l2reg),
                   layer=Activation(oact))
    model.add_output(name='score', input='scoreS')
    return model


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Benchmark RNN on a bipartite ranking task (answer selection)")
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
                         ModelCheckpoint('weights-bestval.h5', save_best_only=True, monitor='mrr', mode='max')],
              batch_size=160, nb_epoch=8)
    model.save_weights('weights-final.h5', overwrite=True)
    ev.eval_anssel(model.predict(gr)['score'][:,0], s0, y, 'Train')
    ev.eval_anssel(model.predict(grt)['score'][:,0], s0t, yt, 'Val')
