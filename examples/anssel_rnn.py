#!/usr/bin/python3
"""
An Answer Sentence Selection classifier that uses full-fledged features
of the pysts Keras toolkit (KeraSTS) and even with a very simple architecture
achieves 2015-state-of-art results on the task.

The architecture uses shared one-directional GRU to produce sentence embeddings,
adaptable word embedding matrix preinitialized with 300D GloVe, projection
matrix (MemNN-like - applied to both sentences to project them to a common
external similarity space) and dot-product similarity measure.

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
    * wang: devMRR=0.84364, testMRR=0.81342

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
    s0, s1, y, t = loader.load_anssel(fname, subsample0=1)

    if vocab is None:
        vocab = Vocabulary(s0 + s1)

    si0 = vocab.vectorize(s0)
    si1 = vocab.vectorize(s1)
    f0, f1 = nlp.sentence_flags(s0, s1, s0pad, s1pad)
    gr = graph_input_anssel(si0, si1, y, f0, f1)

    return (s0, s1, y, vocab, gr)


def prep_model(glove, vocab, dropout=1/2, l2reg=1e-3,
               rnn=GRU, rnnact='tanh', rnninit='glorot_uniform', sdim=2,
               pdim=2, Ddim=2, ptscorer=B.dot_ptscorer, oact='sigmoid'):
    model = Graph()
    N = B.embedding(model, glove, vocab, s0pad, s1pad, dropout)
    
    # RNN
    model.add_shared_node(name='rnn', inputs=['e0_', 'e1_'], outputs=['e0s', 'e1s'],
                          layer=rnn(input_dim=N, output_dim=int(N*sdim), input_length=s0pad, init=rnninit, activation=rnnact))
    model.add_shared_node(name='rnndrop', inputs=['e0s', 'e1s'], outputs=['e0s_', 'e1s_'],
                          layer=Dropout(dropout))
    
    # Projection
    model.add_shared_node(name='proj', inputs=['e0s_', 'e1s_'], outputs=['e0p', 'e1p'],
                          layer=Dense(input_dim=int(N*sdim), output_dim=int(N*pdim), W_regularizer=l2(l2reg)))

    # Measurement
    model.add_node(name='scoreS', input=ptscorer(model, ['e0p', 'e1p'], Ddim, N, l2reg),
                   layer=Activation(oact))
    model.add_output(name='score', input='scoreS')
    return model


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Benchmark RNN on a bipartite ranking task (answer selection)")
    parser.add_argument("-N", help="GloVe dim", type=int, default=300)
    parser.add_argument("--wang", help="whether to run on Wang inst. of YodaQA dataset", type=int, default=0)
    args = parser.parse_args()

    glove = emb.GloVe(N=args.N)
    if args.wang == 1:
        s0, s1, y, vocab, gr = load_set('anssel-wang/train-all.csv')
        s0t, s1t, yt, _, grt = load_set('anssel-wang/dev.csv', vocab)
    else:
        s0, s1, y, vocab, gr = load_set('anssel-yodaqa/curatedv1-training.csv')
        s0t, s1t, yt, _, grt = load_set('anssel-yodaqa/curatedv1-val.csv', vocab)

    model = prep_model(glove, vocab, oact='linear', dropout=2/3, l2reg=1e-4)
    model.compile(loss={'score': ranknet}, optimizer='adam')  # for 'binary_crossentropy', drop the custom oact
    model.fit(gr, validation_data=grt,
              callbacks=[AnsSelCB(s0t, grt),
                         ModelCheckpoint('weights-bestval.h5', save_best_only=True, monitor='mrr', mode='max')],
              batch_size=80, nb_epoch=4)
    model.save_weights('weights-final.h5')
    ev.eval_anssel(model.predict(gr)['score'][:,0], s0, y, 'Train')
    ev.eval_anssel(model.predict(grt)['score'][:,0], s0t, yt, 'Val')
