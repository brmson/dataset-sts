"""
A simple model based on skipthoughts sentence embeddings.

To set up:
    * Execute the "Getting started" wgets in its README
    * set config['skipthoughts_datadir'] to directory with downloaded files
    * make skipthoughts.py from https://github.com/ryankiros/skip-thoughts/blob/master/skipthoughts.py 
        available via import skipthoughts

Inner working: First we compute skipthought embedding of both inputs; then we merge them (multiply & subtract), cancatenate, and compute result (1 MLP layer).
"""

from __future__ import print_function
from __future__ import division


from keras.models import Graph
from keras.layers.core import Activation, Dense, Dropout
from keras.regularizers import l2
from keras.optimizers import Adam

import pysts.embedding as emb
import pysts.loader as loader
import pysts.kerasts.blocks as B
from pysts.kerasts.objectives import pearsonobj

import numpy as np


def config(c):
    # XXX:
    c['skipthoughts_datadir'] = "/storage/ostrava1/home/nadvorj1/skip-thoughts/"

    # disable GloVe
    c['embdim'] = None
    # disable Keras training
    c['ptscorer'] = None

    # Which version of precomputed ST vectors to use
    c["skipthoughts_uni_bi"] = "combined"

    # loss is set in __init__
    c["loss"] = None

    # Values from original code (ryankiros/skip-thoughts/eval_sick.py):
    c['merge_sum'] = True
    c['merge_mul'] = False
    c['merge_diff'] = False
    c['merge_absdiff'] = True
    # l2=0 is used in eval_sick.py. They used some value in paper
    c['l2reg'] = 0.0
    c['dropout'] = 0.0

    # Add <End-Of-Sentence> mark to inputs. If inputs have correct
    # punctuation it tend to be better without EOS.
    c['use_eos'] = True

    # appending boolean flags to ST vectors
    c["use_flags"] = False


class STModel:
    """ Quacks (a little) like a Keras model. """

    def __init__(self, c, output):
        self.weights_to_load = None
        self.c = c
        self.output = output

        if c.get("clipnorm"):
            c["opt"] = Adam(clipnorm=c["clipnorm"])

        # xxx: this will probably break soon
        if output == 'classes':
            self.output_width = 6  # xxx: sick only needs 5
            self.output = 'classes'
            if not self.c.get("loss"):
                # note: this can be overwritten from shell, but not from task config
                self.c["loss"] = "categorical_crossentropy"  # (used in orig paper)

            if not self.c.get("output_activation"):
                self.c["output_activation"] = "softmax"

            c['balance_class'] = False

        else:  # output == binary
            self.output_width = 1
            self.output = 'score'

            if not self.c.get("loss"):
                self.c['loss'] = 'binary_crossentropy'

            if not self.c.get("output_activation"):
                c["output_activation"] = "sigmoid"

            c['balance_class'] = True

        if not self.c.get("use_eos"):
            self.c["use_eos"] = output == 'classes'

        self.st = emb.SkipThought(c=self.c)
        self.N = self.st.N

    def prep_model(self, do_compile=True, load_weights=True):
        if hasattr(self, "model"):
            return
        dropout = self.c["dropout"]

        self.model = Graph()
        self.model.add_input(name='e0', input_shape=(self.N,))
        self.model.add_input(name='e1', input_shape=(self.N,))
        self.model.add_node(name="e0_", input="e0", layer=Dropout(dropout))
        self.model.add_node(name="e1_", input="e1", layer=Dropout(dropout))

        merges = []
        if self.c.get("merge_sum"):
            self.model.add_node(name='sum', inputs=['e0_', 'e1_'], layer=Activation('linear'), merge_mode='sum')
            self.model.add_node(name="sum_", input="sum", layer=Dropout(dropout))
            merges.append("sum_")

        if self.c.get("merge_mul"):
            self.model.add_node(name='mul', inputs=['e0_', 'e1_'], layer=Activation('linear'), merge_mode='mul')
            self.model.add_node(name="mul_", input="mul", layer=Dropout(dropout))
            merges.append("mul_")

        if self.c.get("merge_absdiff"):
            merge_name = B.absdiff_merge(self.model, ["e0_", "e1_"], pfx="", layer_name="absdiff", )
            self.model.add_node(name="%s_" % merge_name, input=merge_name, layer=Dropout(dropout))
            merges.append("%s_" % merge_name)

        if self.c.get("merge_diff"):
            merge_name = B.absdiff_merge(self.model, ["e0_", "e1_"], pfx="", layer_name="diff")
            self.model.add_node(name="%s_" % merge_name, input=merge_name, layer=Dropout(dropout))
            merges.append("%s_" % merge_name)

        self.model.add_node(name='hidden', inputs=merges, merge_mode='concat',
                            layer=Dense(self.output_width, W_regularizer=l2(self.c['l2reg'])))
        self.model.add_node(name='out', input='hidden', layer=Activation(self.c['output_activation']))
        self.model.add_output(name=self.output, input='out')

        if do_compile:
            self.model.compile(loss={self.output: self.c['loss']}, optimizer=self.c["opt"])

        if self.weights_to_load and load_weights:
            self.model.load_weights(*self.weights_to_load[0], **self.weights_to_load[1])
 
    def add_flags(self, e, f):
        f = np.asarray(f, dtype="float32")
        flags_n = f.shape[1] * f.shape[2]
        f = f.reshape(e.shape[0], flags_n)
        e = np.concatenate((e, f), axis=1)
        return e

    def prepare_data(self, gr, balance=False):
        self.precompute_embeddings(gr)

        e0, e1, _, _, y = loader.load_embedded(self.st, gr["s0"], gr["s1"], gr[self.output], balance=False, ndim=1)

        if self.c.get("use_flags"):
            e0 = self.add_flags(e0, gr["f0"])
            e1 = self.add_flags(e1, gr["f1"])
            self.N = e0.shape[1]

        if balance:
            e0, e1, y = loader.balance_dataset((e0, e1, gr[self.output]))
        return np.array(e0), np.array(e1), y

    def fit(self, gr, **kwargs):
        e0, e1, y = self.prepare_data(gr, balance=self.c["balance_class"])
        self.prep_model()

        self.model.fit({'e0': e0, 'e1': e1, self.output: y},
                       batch_size=self.c["batch_size"], nb_epoch=self.c["nb_epoch"],
                       verbose=2)

    def load_weights(self, *args, **kwargs):
        self.weights_to_load = (args, kwargs)

    def save_weights(self, *args, **kwargs):
        self.model.save_weights(*args, **kwargs)

    def precompute_embeddings(self, gr):
        sentences = [" ".join(words) for words in gr["s0"] + gr["s1"]]
        self.st.batch_embedding(sentences)

    def predict(self, gr):
        e0, e1, _ = self.prepare_data(gr, balance=False)
        self.prep_model()
        result = self.model.predict({'e0': e0, 'e1': e1})
        return result


def prep_model(vocab, c, output='score'):
    return STModel(c, output)
