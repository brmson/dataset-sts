#!/usr/bin/python3
"""
Tool for hyperparameter tuning of the answer selection model.

Usage: tools/anssel_tuning.py MODEL TRAINDATA VALDATA

XXX this is a preliminary tool that has the parameter set hardcoded.
"""

from __future__ import print_function
from __future__ import division

import importlib
import sys
import time

from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.layers.recurrent import SimpleRNN, GRU, LSTM
import pysts.embedding as emb
import pysts.eval as ev
import pysts.kerasts.blocks as B
from pysts.kerasts.callbacks import AnsSelCB
from pysts.kerasts.objectives import ranknet
from pysts.hyperparam import RandomSearch, hash_params

import anssel_train
import models  # importlib python3 compatibility requirement


if __name__ == "__main__":
    modelname, trainf, valf = sys.argv[1:4]

    module = importlib.import_module('.'+modelname, 'models')

    s0, s1, y, vocab, gr = anssel_train.load_set(trainf)
    s0t, s1t, yt, _, grt = anssel_train.load_set(valf, vocab)

    glove = emb.GloVe(300)  # XXX hardcoded N

    rs = RandomSearch(modelname+'_rlog.txt',
                      dropout=[1/2, 2/3, 3/4], inp_e_dropout=[1/2, 3/4, 4/5], l2reg=[1e-4, 1e-3, 1e-2],
                      cnnact=['tanh', 'tanh', 'relu'], cnninit=['glorot_uniform', 'glorot_uniform', 'normal'],
                      cdim={1: [0,0,1/2,1,2], 2: [0,0,1/2,1,2,0], 3: [0,0,1/2,1,2,0], 4: [0,0,1/2,1,2,0], 5: [0,0,1/2,1,2]},
                      project=[True, True, False], pdim=[1, 2, 2.5, 3],
                      ptscorer=[B.mlp_ptscorer], Ddim=[1, 2, 2.5, 3])

    for ps, h, pardict in rs():
        conf, ps, h = anssel_train.config(module.config, [])
        for k, v in pardict.items():
            conf[k] = v
        ps, h = hash_params(conf)
        runid = '%s_%x' % (modelname, h)
        print(' ...... %x .................... %s' % (h, ps))

        try:
            model = anssel_train.build_model(glove, vocab, module.prep_model, conf)
            model.fit(gr, validation_data=grt,
                      callbacks=[AnsSelCB(s0t, grt),
                                 ModelCheckpoint('weights-'+runid+'.h5', save_best_only=True, monitor='mrr', mode='max'),
                                 EarlyStopping(monitor='mrr', mode='max', patience=4)],
                      batch_size=160, nb_epoch=16, samples_per_epoch=int(len(s0)/4))
            # mrr = max(hist.history['mrr'])
            model.load_weights('weights-'+runid+'.h5')
            ev.eval_anssel(model.predict(gr)['score'][:,0], s0, y, 'Train')
            mrr = ev.eval_anssel(model.predict(grt)['score'][:,0], s0t, yt, 'Val')
            rs.report(ps, h, mrr)
        except Exception as e:
            print(e)
            time.sleep(1)
