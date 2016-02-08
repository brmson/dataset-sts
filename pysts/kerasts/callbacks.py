"""
Task-specific callbacks for the fit() function.
"""

from keras.callbacks import Callback

import pysts.eval as ev


class AnsSelCB(Callback):
    """ A callback that monitors answer selection validation MRR after each epoch """
    def __init__(self, val_s0, val_gr):
        self.val_s0 = val_s0
        self.val_gr = val_gr  # graph_input()

    def on_epoch_end(self, epoch, logs={}):
        ypred = self.model.predict(self.val_gr)['score'][:,0]
        mrr = ev.mrr(self.val_s0, self.val_gr['score'], ypred)
        print('                                                       val mrr %f' % (mrr,))
