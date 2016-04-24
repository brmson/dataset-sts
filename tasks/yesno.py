
from __future__ import print_function
from __future__ import division

import pickle

from keras.callbacks import EarlyStopping, ModelCheckpoint
import keras.preprocessing.sequence as prep
import numpy as np

from pysts.kerasts import graph_input_anssel
import pysts.loader as loader
import pysts.nlp as nlp
from pysts.vocab import Vocabulary
from keras.layers.core import Activation, Dropout, TimeDistributedDense, Dense, Masking
from keras.regularizers import l2
from keras.layers.embeddings import Embedding
from keras.models import Graph
from pysts.clasrel_layers import Reshape_, WeightedMean, SumMask
import pysts.nlp as nlp

from . import AbstractTask


class Container:
    """Container for merging questions together."""
    def __init__(self, q_text, s0, s1, si0, si1, f0, f1, y):
        self.q_text = q_text  # str of question
        self.s0 = s0
        self.s1 = s1
        self.si0 = si0
        self.si1 = si1
        self.f0 = f0
        self.f1 = f1
        self.y = y



class YesNoTask(AbstractTask):
    def __init__(self):
        self.name = 'yesno'
        # self.s0pad = 60
        # self.s1pad = 60
        # self.max_sentences = 50
        self.emb = None
        self.vocab = None

    def config(self, c):
        c['task>model'] = True
        c['loss'] = 'binary_crossentropy'
        c['max_sentences'] = 50
        c['spad'] = 60
        c['embdim'] = 50
        c['nb_epoch'] = 30

        c['opt'] = 'adam'
        c['inp_e_dropout'] = 0.
        c['dropout'] = 0.
        c['l2reg'] = 0.01
        c['e_add_flags'] = True
        c['ptscorer'] = mlp_ptscorer
        c['mlpsum'] = 'sum'
        c['Ddim'] = .1
        c['loss'] = 'binary_crossentropy'
        c['nb_epoch'] = 100
        c['batch_size'] = 10
        c['class_mode'] = 'binary'

        # old rnn
        c['pdim'] = 2.5
        c['pact'] = 'tanh'

    def load_set(self, fname, cache_dir=None):
        save_cache = False
        if cache_dir:
            import os.path
            fname_abs = os.path.abspath(fname)
            from hashlib import md5
            cache_filename = "%s/%s.p" % (cache_dir, md5(fname_abs.encode("utf-8")).hexdigest())
            try:
                with open(cache_filename, "rb") as f:
                    return pickle.load(f)
            except (IOError, TypeError, KeyError):
                save_cache = True

        s0, s1, y, kw, akw, t = loader.load_anssel(fname, skip_oneclass=False)

        if self.vocab is None:
            vocab = Vocabulary(s0 + s1)  # FIXME: lower?
        else:
            vocab = self.vocab

        si0 = vocab.vectorize(s0, spad=self.s0pad)
        si1 = vocab.vectorize(s1, spad=self.s1pad)
        f0, f1 = nlp.sentence_flags(s0, s1, self.s0pad, self.s1pad)
        gr = graph_input_anssel(si0, si1, y, f0, f1, s0, s1, kw=kw, akw=akw)
        gr, y = self.merge_questions(gr)
        if save_cache:
            with open(cache_filename, "wb") as f:
                pickle.dump((s0, s1, y, vocab, gr), f)
                print("save")

        return gr, y, vocab

    def build_model(self, module_prep_model, do_compile=True):

        model = build_model(self.emb, self.vocab, module_prep_model, self.c)

        # for lname in self.c['fix_layers']:  # whats this??
        #     model.nodes[lname].trainable = False

        if do_compile:
            model.compile(loss={'score': self.c['loss']}, optimizer=self.c['opt'])
        return model

    def fit_callbacks(self, weightsf):
        return [ModelCheckpoint(weightsf, save_best_only=True, monitor='val_loss', mode='min'),
                EarlyStopping(monitor='val_loss', mode='min', patience=10)]

    def eval(self, model):
        res = []
        for gr, fname in [(self.gr, self.trainf), (self.grv, self.valf), (self.grt, self.testf)]:
            if gr is None:
                res.append(None)
                continue
            loss, acc = model.evaluate(gr, show_accuracy=True)
            res.append(YesNoRes(loss, acc))
        return tuple(res)

    def res_columns(self, mres, pfx=' '):
        """ Produce README-format markdown table row piece summarizing
        important statistics """
        return('%s%.6f    |%s%.6f |%s%.6f |%s%.6f'
               % (pfx, mres[self.trainf]['Loss'],
                  pfx, mres[self.valf]['Precision'],
                  pfx, mres[self.testf].get('Loss', np.nan),
                  pfx, mres[self.testf].get('Precision', np.nan)))

    def merge_questions(self, gr):
        # s0=questions, s1=sentences
        q_t = ''
        ixs = []
        for q_text, i in zip(gr['s0'], range(len(gr['s0']))):
            if q_t != q_text:
                ixs.append(i)
                q_t = q_text

        containers = []
        for i, i_ in zip(ixs, ixs[1:]+[len(gr['s0'])]):
            container = Container(gr['s0'][i], gr['s0'][i:i_], gr['s1'][i:i_],
                                  gr['si0'][i:i_], gr['si1'][i:i_],
                                  gr['f0'][i:i_], gr['f1'][i:i_], gr['score'][i])
            containers.append(container)

        si03d, si13d, f04d, f14d = [], [], [], []
        for c in containers:
            si0 = prep.pad_sequences(c.si0.T, maxlen=self.c['max_sentences'],
                                     padding='post', truncating='post').T
            si1 = prep.pad_sequences(c.si1.T, maxlen=self.c['max_sentences'],
                                     padding='post', truncating='post').T
            si03d.append(si0)
            si13d.append(si1)

            f0 = prep.pad_sequences(c.f0.transpose((1, 0, 2)), maxlen=self.c['max_sentences'],
                                    padding='post',
                                    truncating='post', dtype='bool').transpose((1, 0, 2))
            f1 = prep.pad_sequences(c.f1.transpose((1, 0, 2)), maxlen=self.c['max_sentences'],
                                    padding='post',
                                    truncating='post', dtype='bool').transpose((1, 0, 2))
            f04d.append(f0)
            f14d.append(f1)

        y = np.array([c.y for c in containers])
        gr = {'si03d': np.array(si03d), 'si13d': np.array(si13d),
              'f04d': np.array(f04d), 'f14d': np.array(f14d), 'score': y}

        return gr, y

from collections import namedtuple
YesNoRes = namedtuple('YesNoRes', ['Loss', 'Precision'])


def _prep_model(model, glove, vocab, module_prep_model, c, oact, s0pad, s1pad):
    # Input embedding and encoding
    N = embedding(model, glove, vocab, s0pad, s1pad, c['inp_e_dropout'], add_flags=c['e_add_flags'])
    # Sentence-aggregate embeddings
    final_outputs = module_prep_model(model, N, s0pad, s1pad, c)

    model.add_node(name='scoreS1', input=mlp_ptscorer(model, final_outputs, c['Ddim'], N, c['l2reg'], pfx='S1_'),
                   layer=Activation(oact))
    model.add_node(name='scoreS2', input=mlp_ptscorer(model, final_outputs, c['Ddim'], N, c['l2reg'], pfx='S2_'),
                   layer=Activation(oact))


def embedding(model, glove, vocab, s0pad, s1pad, dropout, trainable=False,
              add_flags=False):
    """ Sts embedding layer, without creating inputs. """
    # TODO: add switch to original -> no code duplication

    if add_flags:
        outputs = ['e0[0]', 'e1[0]']
    else:
        outputs = ['e0', 'e1']

    # model.add_shared_node(name='emb', inputs=['si0', 'si1'], outputs=outputs,
    #                       layer=Embedding(input_dim=vocab.size(), input_length=s1pad,
    #                                       output_dim=glove.N, mask_zero=True,
    #                                       weights=[vocab.embmatrix(glove)], trainable=trainable))
    model.add_node(name='e0[0]x', input='si0',
                  layer=Embedding(input_dim=vocab.size(), input_length=s0pad,
                                          output_dim=glove.N, mask_zero=True,
                                          weights=[vocab.embmatrix(glove)], trainable=trainable))
    model.add_node(name='e1[0]', input='si1',
                  layer=Embedding(input_dim=vocab.size(), input_length=s1pad,
                                          output_dim=glove.N, mask_zero=True,
                                          weights=[vocab.embmatrix(glove)], trainable=trainable))

    model.add_node(name='e0[0]', input='e0[0]x', layer=Activation('linear'))

    if add_flags:
        for m in [0, 1]:
            model.add_node(name='e%d'%(m,), inputs=['e%d[0]'%(m,), 'f%d'%(m,)],
                           merge_mode='concat', layer=Activation('linear'))
        N = glove.N + nlp.flagsdim
    else:
        N = glove.N

    model.add_shared_node(name='embdrop', inputs=['e0', 'e1'], outputs=['e0_', 'e1_'],
                          layer=Dropout(dropout, input_shape=(N,)))

    return N


import theano
def layer_fun(model, layer_name):
    thf = theano.function([model.inputs[name].input for name in model.input_order],
                          model.nodes[layer_name].get_output(train=False),
                          on_unused_input='ignore', allow_input_downcast=True)
    return thf

def layer_fun_mask(model, layer_name):
    thf = theano.function([model.inputs[name].input for name in model.input_order],
                          model.nodes[layer_name].get_output_mask(train=False),
                          on_unused_input='ignore', allow_input_downcast=True)
    return thf
    # return thf(*[gr[name] for name in model.input_order])

testing_function = None

def build_model(glove, vocab, module_prep_model, c):
    s0pad = s1pad = c['spad']
    max_sentences = c['max_sentences']
    rnn_dim = 1
    print('Model')
    model = Graph()
    # ===================== inputs of size (batch_size, max_sentences, s_pad)
    model.add_input('si03d', (max_sentences, s0pad), dtype=int)  # XXX: cannot be cast to int->problem?
    model.add_input('si13d', (max_sentences, s1pad), dtype=int)
    if True:  # TODO: if flags
        model.add_input('f04d', (max_sentences, s0pad, nlp.flagsdim))
        model.add_input('f14d', (max_sentences, s1pad, nlp.flagsdim))
        model.add_node(Reshape_((s0pad, nlp.flagsdim)), 'f0', input='f04d')
        model.add_node(Reshape_((s1pad, nlp.flagsdim)), 'f1', input='f14d')

    # model.add_node(Masking(mask_value=0., input_shape=(max_sentences, s0pad)),'masked', 'si03d')

    # ===================== reshape to (batch_size * max_sentences, s_pad)
    model.add_node(Reshape_((s0pad,)), 'si0', input='si03d')
    model.add_node(Reshape_((s1pad,)), 'si1', input='si13d')

    # ===================== outputs from sts
    oact = 'linear'
    _prep_model(model, glove, vocab, module_prep_model, c, oact, s0pad, s1pad)  # out = ['scoreS1', 'scoreS2']
    # ===================== reshape (batch_size * max_sentences,) -> (batch_size, max_sentences, rnn_dim)
    model.add_node(Reshape_((max_sentences, rnn_dim)), 'sts_in1', input='scoreS1')
    model.add_node(Reshape_((max_sentences, rnn_dim)), 'sts_in2', input='scoreS2')

    # ===================== [w_full_dim, q_full_dim] -> [class, rel]
    model.add_node(TimeDistributedDense(1, activation='sigmoid',
                                        W_regularizer=l2(c['l2reg']),
                                        b_regularizer=l2(c['l2reg'])),
                   'c', input='sts_in1')
    model.add_node(TimeDistributedDense(1, activation='sigmoid',
                                        W_regularizer=l2(c['l2reg']),
                                        b_regularizer=l2(c['l2reg'])),
                   'r', input='sts_in2')

    model.add_node(SumMask(), 'mask', input='si03d')
    # ===================== mean of class over rel
    model.add_node(WeightedMean(max_sentences=max_sentences),
                   name='weighted_mean', inputs=['c', 'r', 'mask'])
    model.add_output(name='score', input='weighted_mean')
    return model


def mlp_ptscorer(model, inputs, Ddim, N, l2reg, pfx='out', sum_mode='sum'):
    """ Element-wise features from the pair fed to an MLP. """
    if sum_mode == 'absdiff':
        model.add_node(name=pfx+'sum', layer=absdiff_merge(model, inputs))
    else:
        model.add_node(name=pfx+'sum', inputs=inputs, layer=Activation('linear'), merge_mode='sum')
    model.add_node(name=pfx+'mul', inputs=inputs, layer=Activation('linear'), merge_mode='mul')

    # model.add_node(name=pfx+'hdn', inputs=[pfx+'sum', pfx+'mul'], merge_mode='concat',
    #                layer=Dense(output_dim=int(N*Ddim), W_regularizer=l2(l2reg), activation='sigmoid'))
    # model.add_node(name=pfx+'mlp', input=pfx+'hdn',
    #                layer=Dense(output_dim=1, W_regularizer=l2(l2reg)))
    model.add_node(name=pfx+'mlp', inputs=[pfx+'sum', pfx+'mul'], merge_mode='concat',
                   layer=Dense(output_dim=1, W_regularizer=l2(l2reg), activation='linear'))
    return pfx+'mlp'

from models.pysts.kerasts.blocks import LambdaMerge
def absdiff_merge(model, layers):
    """ Merging two layers into one, via element-wise subtraction and then taking absolute value.

    Example of usage: model.add_node(name="diff", layer=absdiff_merge(["e0_", "e1_"]))

    TODO: The more modern way appears to be to use "join" merge mode and Lambda layer.
    """
    def diff(X):
        if len(X)!=2:
            raise ValueError("")
        return K.abs(X[0]-X[1])

    def output_shape(input_shapes):
        return input_shapes[0]

    return LambdaMerge([model.nodes[l] for l in layers], diff, output_shape)


def task():
    return YesNoTask()