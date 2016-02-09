"""
Task-specific callbacks for the fit() function.
"""

from keras.callbacks import Callback

import pysts.eval as ev
import pysts.loader as loader


class AnsSelCB(Callback):
    """ A callback that monitors answer selection validation MRR after each epoch """
    def __init__(self, val_s0, val_gr):
        self.val_s0 = val_s0
        self.val_gr = val_gr  # graph_input()

    def on_epoch_end(self, epoch, logs={}):
        ypred = self.model.predict(self.val_gr)['score'][:,0]
        mrr = ev.mrr(self.val_s0, self.val_gr['score'], ypred)
        print('                                                       val mrr %f' % (mrr,))
        logs['mrr'] = mrr


class STSPearsonCB(Callback):
    def __init__(self, train_gr, val_gr):
        self.train_gr = train_gr
        self.val_gr = val_gr
    def on_epoch_end(self, epoch, logs={}):
        prtr = ev.eval_sts(self.model.predict(self.train_gr)['classes'],
                           loader.sts_categorical2labels(self.train_gr['classes']), 'Train', quiet=True)
        prval = ev.eval_sts(self.model.predict(self.val_gr)['classes'],
                            loader.sts_categorical2labels(self.val_gr['classes']), 'Val', quiet=True)
        print('                  train Pearson %f    val Pearson %f' % (prtr, prval))
        logs['pearson'] = prval
