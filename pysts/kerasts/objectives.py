"""
A variety of objectives that may make sense in the sentence pair setting.

  * Bipartite ranking convex surrogate objectives (for ranking tasks like anssel).
    (NOTE - this works only with linear output, *not* sigmoid)
"""

import keras.backend as K


def ranknet(y_true, y_pred):
    """ Bipartite ranking surrogate """
    return K.mean(K.log(1. + K.exp(-(y_true * y_pred - (1-y_true) * y_pred))), axis=-1)

def ranksvm(y_true, y_pred):
    """ Bipartite ranking surrogate """
    return K.mean(K.maximum(1. - (y_true * y_pred - (1-y_true) * y_pred), 0.), axis=-1)

def cicerons_1504(y_true, y_pred):
    """ Bipartite ranking surrogate - http://arxiv.org/pdf/1504.06580v2.pdf """
    return K.mean(K.log(1. + K.exp(2*(2.5 - y_true*y_pred))) +
                  K.log(1. + K.exp(2*(0.5 + (1-y_true)*y_pred))), axis=-1)



def _y2num(y):
    """ theano-friendly class-to-score conversion """
    return y[:,1] + 2*y[:,2] + 3*y[:,3] + 4*y[:,4] + 5*y[:,5]

def pearsonobj(y_true, y_pred):
    """ Pearson's r objective for STS grade correlation """
    ny_true = _y2num(y_true)
    ny_pred = _y2num(y_pred)
    my_true = K.mean(ny_true)
    my_pred = K.mean(ny_pred)
    var_true = (ny_true - my_true)**2
    var_pred = (ny_pred - my_pred)**2
    return - K.sum((ny_true - my_true) * (ny_pred - my_pred), axis=-1) / \
             (K.sqrt(K.sum(var_true, axis=-1) * K.sum(var_pred, axis=-1)))
