#!/usr/bin/python3
"""
Tool for hyperparameter tuning of the answer selection CNN model.
"""

from __future__ import print_function
from __future__ import division

import sys

import anssel_cnn as E
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.layers.recurrent import SimpleRNN, GRU, LSTM
import pysts.embedding as emb
import pysts.eval as ev
import pysts.kerasts.blocks as B
from pysts.kerasts.callbacks import AnsSelCB
from pysts.kerasts.objectives import ranknet
from pysts.hyperparam import RandomSearch


if __name__ == "__main__":
    s0, s1, y, vocab, gr = E.load_set('anssel-wang/train-all.csv')
    s0t, s1t, yt, _, grt = E.load_set('anssel-wang/dev.csv', vocab)

    glove = emb.GloVe(300)  # XXX hardcoded N

    # XXX: hardcoded loss function

    rs = RandomSearch('cnn_rlog.txt',
                      dropout=[1/4, 1/2, 2/3, 3/4], l2reg=[1e-4, 1e-3, 1e-2],
                      cnnact=['tanh', 'tanh', 'relu'], cnninit=['glorot_uniform', 'glorot_uniform', 'normal'],
                      cdim={1: [0,0,1/2,1,2], 2: [0,0,1/2,1,2], 3: [0,0,1/2,1,2], 4: [0,0,1/2,1,2], 5: [0,0,1/2,1,2]},
                      project=[True, True, False], pdim=[1, 2, 2.5],
                      ptscorer=[B.dot_ptscorer, B.mlp_ptscorer], Ddim=[1, 2, 2.5])
    for ps, h, pardict in rs():
        print(' ...... %s .................... %s' % (h, ps))
        model = E.prep_model(glove, vocab, oact='linear', **pardict)
        model.compile(loss={'score': ranknet}, optimizer='adam')
        model.fit(gr, validation_data=grt,
                  callbacks=[AnsSelCB(s0t, grt),
                             ModelCheckpoint('weights-cnn_%s.h5' % (h,), save_best_only=True, monitor='mrr', mode='max'),
                             EarlyStopping(monitor='mrr', mode='max', patience=1)],
                  batch_size=160, nb_epoch=12)
        # mrr = max(hist.history['mrr'])
        model.load_weights('weights-cnn_%s.h5' % (h,))
        ev.eval_anssel(model.predict(gr)['score'][:,0], s0, y, 'Train')
        mrr = ev.eval_anssel(model.predict(grt)['score'][:,0], s0t, yt, 'Test')
        rs.report(ps, h, mrr)
