"""
KeraSTS interface for datasets of the Hypothesis Evaluation task,
i.e. producing an aggregate s0 classification based on the set of its
s1 evidence (typically mix of relevant and irrelevant).
See data/hypev/... for details and actual datasets.
Training example:
    tools/train.py cnn hypev data/hypev/argus/argus_train.csv data/hypev/argus/argus_test.csv dropout=0

Class/rel mode selection:

    * rel_mode can be either 'scoreS2' (default), f_add value
      or None (for no relevancy weighing)
    * class_mode can be either 'scoreS1' (default), f_add value
      or None

Prescoring example (BM25 for relevancy and as an extra classification feature
and pruning least related evidence):

    "prescoring='termfreq'" "prescoring_weightsf='weights-anssel-termfreq-3368350fbcab42e4-bestval.h5'" "prescoring_input='bm25'" "f_add=['bm25']" "f_add_S1=['bm25']" "rel_mode='bm25'" prescoring_prune=20 max_sentences=20

N.B. if you use prescoring as an input (either for _mode or as extra MLP input),
you must list it as both f_add and _mode or f_add_S1 or f_add_S2 (f_add does
not imply MLP input by itself, contrary to other tasks).

"""

from __future__ import division
from __future__ import print_function

import csv
import numpy as np
import pickle
import random
import re
import traceback

import keras.preprocessing.sequence as prep
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.layers.core import Activation, TimeDistributedDense, Dense
from keras.models import Graph
from keras.regularizers import l2

import pysts.eval as ev
import pysts.loader as loader
import pysts.nlp as nlp
from pysts.kerasts import graph_input_anssel, graph_input_slice
import pysts.kerasts.blocks as B
from pysts.kerasts.callbacks import HypEvCB
from pysts.kerasts.clasrel_layers import Reshape_, WeightedMean, SumMask
from pysts.vocab import Vocabulary
from . import AbstractTask
from .anssel import AnsSelTask


class Container:
    """Container for merging question's sentences together."""
    def __init__(self, q_text, s0, s1, si0, si1, sj0, sj1, f0, f1, y, qid, xc, xr, fx):
        self.q_text = q_text  # str of question
        self.s0 = s0
        self.s1 = s1
        self.si0 = si0
        self.si1 = si1
        self.sj0 = sj0
        self.sj1 = sj1
        self.f0 = f0
        self.f1 = f1
        self.y = y
        self.qid = qid
        self.xc = xc
        self.xr = xr
        # fx is a dict of "extra" features
        # TODO: Make this whole thing a dict that self-describes the
        # required transformations
        self.fx = fx


class HypEvTask(AbstractTask):
    def __init__(self):
        self.name = 'hypev'
        self.emb = None
        self.vocab = None
        self.gr = None

        # Prescore individual htext,mtext pairs using anssel model.
        self.prescoring_task = AnsSelTask

    def config(self, c):
        c['task>model'] = True
        c['loss'] = 'binary_crossentropy'
        c['max_sentences'] = 50
        c['class_mode'] = 'scoreS1'
        c['rel_mode'] = 'scoreS2'
        c['aux_c'] = False
        c['aux_r'] = False
        c['spad'] = 60
        c['embdim'] = 50
        c['embicase'] = True
        c['nb_epoch'] = 100
        c['batch_size'] = 10

        c['inp_e_dropout'] = 0.
        c['inp_w_dropout'] = 0.
        c['dropout'] = 0.
        c['e_add_flags'] = True
        c['ptscorer'] = B.mlp_ptscorer
        c['mlpsum'] = 'sum'
        c['Ddim'] = 0
        c['oact'] = 'linear'

        # old rnn
        c['pdim'] = 2.5
        c['pact'] = 'tanh'

        # which question classes of mctest to load
        c['mcqtypes'] = ['one', 'multiple']

    def load_set(self, fname, cache_dir=None, lists=None):
        # TODO: Make the cache-handling generic,
        # and offer a way to actually pass cache_dir
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

        if lists is not None:
            s0, s1, y, qids, xtra, types = lists
        else:
            xtra = None
            if '/mc' in fname:
                s0, s1, y, qids, types = loader.load_mctest(fname)
            else:
                s0, s1, y, qids = loader.load_hypev(fname)
                try:
                    dsfile = re.sub('\.([^.]*)$', '_aux.tsv', fname)  # train.tsv -> train_aux.tsv
                    with open(dsfile) as f:
                        rows = csv.DictReader(f, delimiter='\t')
                        xtra = loader.load_hypev_xtra(rows)
                        print(dsfile + ' loaded and available')
                except Exception as e:
                    if self.c['aux_r'] or self.c['aux_c']:
                        raise e
                types = None

        if self.vocab is None:
            vocab = Vocabulary(s0 + s1, prune_N=self.c['embprune'], icase=self.c['embicase'])
        else:
            vocab = self.vocab

        # mcqtypes pruning must happen *after* Vocabulary has been constructed!
        if types is not None:
            s0 = [x for x, t in zip(s0, types) if t in self.c['mcqtypes']]
            s1 = [x for x, t in zip(s1, types) if t in self.c['mcqtypes']]
            y = [x for x, t in zip(y, types) if t in self.c['mcqtypes']]
            qids = [x for x, t in zip(qids, types) if t in self.c['mcqtypes']]
            print('Retained %d questions, %d hypotheses (%s types)' % (len(set(qids)), len(set([' '.join(s) for s in s0])), self.c['mcqtypes']))

        si0, sj0 = vocab.vectorize(s0, self.emb, spad=self.s0pad)
        si1, sj1 = vocab.vectorize(s1, self.emb, spad=self.s1pad)
        f0, f1 = nlp.sentence_flags(s0, s1, self.s0pad, self.s1pad)
        gr = graph_input_anssel(si0, si1, sj0, sj1, None, None, y, f0, f1, s0, s1)
        if qids is not None:
            gr['qids'] = qids
        if xtra is not None:
            gr['#'] = xtra['#']
            gr['@'] = xtra['@']
        gr, y = self.merge_questions(gr)
        if save_cache:
            with open(cache_filename, "wb") as f:
                pickle.dump((s0, s1, y, vocab, gr), f)
                print("save")

        return (gr, y, vocab)

    def sample_pairs(self, gr, batch_size, shuffle=True, once=False):
        """ A generator that produces random pairs from the dataset """
        try:
            id_N = int((len(gr['si03d']) + batch_size-1) / batch_size)
            ids = list(range(id_N))
            while True:
                if shuffle:
                    # XXX: We never swap samples between batches, does it matter?
                    random.shuffle(ids)
                for i in ids:
                    sl = slice(i * batch_size, (i+1) * batch_size)
                    ogr = graph_input_slice(gr, sl)
                    # s0, s1 are larger than the rest, unnerving keras
                    ogr.pop('s0', None)
                    ogr.pop('s1', None)
                    ogr['se03d'] = self.emb.map_jset(ogr['sj03d'])
                    ogr['se13d'] = self.emb.map_jset(ogr['sj13d'])
                    # print(sl)
                    # print('<<0>>', ogr['sj0'], ogr['se0'])
                    # print('<<1>>', ogr['sj1'], ogr['se1'])
                    yield ogr
                if once:
                    break
        except Exception:
            traceback.print_exc()

    def build_model(self, module_prep_model, do_compile=True, classrel_outputs=False):
        xcdim = len(loader.hypev_xtra_c) if self.c['aux_c'] else None
        xrdim = len(loader.hypev_xtra_r) if self.c['aux_r'] else None

        model = build_model(self.emb, self.vocab, module_prep_model, self.c, xcdim, xrdim, classrel_outputs)

        for lname in self.c['fix_layers']:
            model.nodes[lname].trainable = False

        if do_compile:
            xloss = {}
            if classrel_outputs:
                xloss['class'] = self.c['loss']
                xloss['rel'] = self.c['loss']
            model.compile(loss=dict(score=self.c['loss'], **xloss), optimizer=self.c['opt'])
        return model

    def fit_callbacks(self, weightsf):
        return [HypEvCB(self, self.grv),
                ModelCheckpoint(weightsf, save_best_only=True, monitor='acc', mode='max'),
                EarlyStopping(monitor='acc', mode='max', patience=10)]

    def eval(self, model):
        res = []
        for gr, fname in [(self.gr, self.trainf), (self.grv, self.valf), (self.grt, self.testf)]:
            if gr is None:
                res.append(None)
                continue
            ypred = self.predict(model, gr)
            res.append(ev.eval_hypev(gr.get('qids', None), ypred, gr['score'], fname))
        return tuple(res)

    def res_columns(self, mres, pfx=' '):
        """ Produce README-format markdown table row piece summarizing
        important statistics """
        if 'QAccuracy' in mres[self.trainf]:  # ev.HypEvRes
            return('%s%.6f |%s%.6f |%s%.6f |%s%.6f |%s%.6f '
                   % (pfx, mres[self.trainf]['QAccuracy'],
                      pfx, mres[self.valf]['QAccuracy'],
                      pfx, mres[self.valf]['QF1'],
                      pfx, mres[self.testf].get('QAccuracy', np.nan),
                      pfx, mres[self.testf].get('QF1', np.nan)))
        else:  # ev.AbcdRes
            return('%s%.6f |%s%.6f |%s%.6f |%s%.6f |%s%.6f '
                   % (pfx, mres[self.trainf]['AbcdAccuracy'],
                      pfx, mres[self.valf]['AbcdAccuracy'],
                      pfx, mres[self.valf]['AbcdMRR'],
                      pfx, mres[self.testf].get('AbcdAccuracy', np.nan),
                      pfx, mres[self.testf].get('AbcdMRR', np.nan)))

    def merge_questions(self, gr):
        # First, apply prescoring
        gr = self.prescoring_apply(gr)

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
                                  gr['sj0'][i:i_], gr['sj1'][i:i_],
                                  gr['f0'][i:i_], gr['f1'][i:i_], gr['score'][i],
                                  gr['qids'][i] if 'qids' in gr else None,
                                  gr['#'][i:i_] if '#' in gr else None,
                                  gr['@'][i:i_] if '@' in gr else None,
                                  dict([(k, gr[k][i:i_]) for k in self.c.get('f_add', [])]))
            containers.append(container)

        si03d, si13d, sj03d, sj13d, f04d, f14d, xc3d, xr3d, mask = [], [], [], [], [], [], [], [], []
        gr_extra = dict()
        for k in self.c.get('f_add', []):
            gr_extra[k] = []

        for c in containers:
            si0 = prep.pad_sequences(c.si0.T, maxlen=self.c['max_sentences'],
                                     padding='post', truncating='post').T
            si1 = prep.pad_sequences(c.si1.T, maxlen=self.c['max_sentences'],
                                     padding='post', truncating='post').T
            si03d.append(si0)
            si13d.append(si1)

            sj0 = prep.pad_sequences(c.sj0.T, maxlen=self.c['max_sentences'],
                                     padding='post', truncating='post').T
            sj1 = prep.pad_sequences(c.sj1.T, maxlen=self.c['max_sentences'],
                                     padding='post', truncating='post').T
            sj03d.append(sj0)
            sj13d.append(sj1)

            f0 = prep.pad_sequences(c.f0.transpose((1, 0, 2)), maxlen=self.c['max_sentences'],
                                    padding='post',
                                    truncating='post', dtype='bool').transpose((1, 0, 2))
            f1 = prep.pad_sequences(c.f1.transpose((1, 0, 2)), maxlen=self.c['max_sentences'],
                                    padding='post',
                                    truncating='post', dtype='bool').transpose((1, 0, 2))
            f04d.append(f0)
            f14d.append(f1)

            if c.xc is not None:
                #print(c.xc, c.xc.T.shape)
                xc = prep.pad_sequences(c.xc.T, maxlen=self.c['max_sentences'],
                                        padding='post', truncating='post').T
                xr = prep.pad_sequences(c.xr.T, maxlen=self.c['max_sentences'],
                                        padding='post', truncating='post').T
                xc3d.append(xc)
                xr3d.append(xr)

            for k in self.c.get('f_add', []):
                fx = prep.pad_sequences(c.fx[k].T, maxlen=self.c['max_sentences'],
                                        padding='post', truncating='post').T
                gr_extra[k].append(fx)

            m = np.where(np.sum(si1 + sj1, axis=1) > 0, 1., 0.)
            mask.append(m)

        y = np.array([c.y for c in containers])
        gr3d = {'si03d': np.array(si03d), 'si13d': np.array(si13d),
                'sj03d': np.array(sj03d), 'sj13d': np.array(sj13d),
                'f04d': np.array(f04d), 'f14d': np.array(f14d),
                'mask': np.array(mask),
                'score': y,
                's0': gr['s0'], 's1': gr['s1']}
        gr3d['c'] = containers

        if self.c['aux_c']:
            gr3d['xc3d'] = np.array(xc3d)
            #print(gr3d['xc3d'].shape, gr3d['si03d'].shape)
        if self.c['aux_r']:
            gr3d['xr3d'] = np.array(xr3d)
            #print(gr3d['xr3d'].shape)

        if 'qids' in gr and gr['qids'] is not None:
            gr3d['qids'] = [c.qid for c in containers]

        for k in gr_extra.keys():
            #gr3d[k] = np.reshape(np.array(gr_extra[k]), (len(containers), self.c['max_sentences']))
            gr3d[k + '3d'] = np.array(gr_extra[k])

        return gr3d, y


def _prep_model(model, glove, vocab, module_prep_model, c, oact, s0pad, s1pad, rnn_dim, make_S1, make_S2):
    if not make_S1 and not make_S2:
        return

    # Input embedding and encoding
    N = B.embedding(model, glove, vocab, s0pad, s1pad, c['inp_e_dropout'],
                    c['inp_w_dropout'], add_flags=c['e_add_flags'], create_inputs=False)
    # Sentence-aggregate embeddings
    final_outputs = module_prep_model(model, N, s0pad, s1pad, c)

    kwargs_S1 = dict()
    kwargs_S2 = dict()
    if c['ptscorer'] == B.mlp_ptscorer:
        kwargs_S1['sum_mode'] = c['mlpsum']
        kwargs_S2['sum_mode'] = c['mlpsum']
        kwargs_S1['Dinit'] = c['Dinit']
        kwargs_S2['Dinit'] = c['Dinit']
    if 'f_add_S1' in c:
        kwargs_S1['extra_inp'] = c['f_add_S1']
    if 'f_add_S2' in c:
        kwargs_S2['extra_inp'] = c['f_add_S2']

    if c['ptscorer'] == '1':
        if 'extra_inp' in kwargs_S1 or 'extra_inp' in kwargs_S1:
            print("Warning: Ignoring extra_inp with ptscorer '1'")
        if make_S1:
            model.add_node(name='scoreS1', input=final_outputs[1],
                           layer=Dense(rnn_dim, activation=oact, W_regularizer=l2(c['l2reg'])))
        if make_S2:
            model.add_node(name='scoreS2', input=final_outputs[1],
                           layer=Dense(rnn_dim, activation=oact, W_regularizer=l2(c['l2reg'])))
    else:
        if make_S1:
            next_input = c['ptscorer'](model, final_outputs, c['Ddim'], N, c['l2reg'], pfx='S1_', **kwargs_S1)
            model.add_node(name='scoreS1', input=next_input, layer=Activation(oact))
        if make_S2:
            next_input = c['ptscorer'](model, final_outputs, c['Ddim'], N, c['l2reg'], pfx='S2_', **kwargs_S2)
            model.add_node(name='scoreS2', input=next_input, layer=Activation(oact))


def build_model(glove, vocab, module_prep_model, c, xcdim=None, xrdim=None, classrel_outputs=False):
    s0pad = s1pad = c['spad']
    max_sentences = c['max_sentences']
    rnn_dim = 1
    print('Model')
    model = Graph()
    # ===================== inputs of size (batch_size, max_sentences, s_pad)
    model.add_input('si03d', (max_sentences, s0pad), dtype=int)  # XXX: cannot be cast to int->problem?
    model.add_input('si13d', (max_sentences, s1pad), dtype=int)
    model.add_input('se03d', (max_sentences, s0pad, glove.N))
    model.add_input('se13d', (max_sentences, s1pad, glove.N))
    if True:  # TODO: if flags
        model.add_input('f04d', (max_sentences, s0pad, nlp.flagsdim))
        model.add_input('f14d', (max_sentences, s1pad, nlp.flagsdim))
        model.add_node(Reshape_((s0pad, nlp.flagsdim)), 'f0', input='f04d')
        model.add_node(Reshape_((s1pad, nlp.flagsdim)), 'f1', input='f14d')
    model.add_input('mask', (max_sentences,))
    model.add_node(Reshape_((max_sentences, 1)), 'mask1', input='mask')
    for inp in c.get('f_add', []):
        model.add_input(inp+'3d', input_shape=(c['max_sentences'], 1))  # assumed scalar
        model.add_node(Reshape_((1,)), inp, input=inp+'3d')  # disperse to individual pairs

    # ===================== reshape to (batch_size * max_sentences, s_pad)
    model.add_node(Reshape_((s0pad,)), 'si0', input='si03d')
    model.add_node(Reshape_((s1pad,)), 'si1', input='si13d')
    model.add_node(Reshape_((s0pad, glove.N)), 'se0', input='se03d')
    model.add_node(Reshape_((s1pad, glove.N)), 'se1', input='se13d')

    # ===================== outputs from sts  # out = ['scoreS1', 'scoreS2']
    _prep_model(model, glove, vocab, module_prep_model, c,
                c['oact'], s0pad, s1pad, rnn_dim,
                c['class_mode'] == 'scoreS1', c['rel_mode'] == 'scoreS2')
    # ===================== reshape (batch_size * max_sentences,) -> (batch_size, max_sentences, rnn_dim)
    if c['class_mode']:
        model.add_node(Reshape_((max_sentences, rnn_dim)), 'sts_in1', input=c['class_mode'])
    if c['rel_mode']:
        model.add_node(Reshape_((max_sentences, rnn_dim)), 'sts_in2', input=c['rel_mode'])
    c_full = 'sts_in1'
    r_full = 'sts_in2'

    # ===================== append auxiliary features
    if xcdim is not None:
        if c['class_mode']:
            model.add_input('xc3d', (max_sentences, xcdim))
            model.add_node(Activation('linear'), 'c_full', inputs=[c_full, 'xc3d'], merge_mode='concat', concat_axis=-1)
            c_full = 'c_full'
    if xrdim is not None:
        if c['rel_mode']:
            model.add_input('xr3d', (max_sentences, xrdim))
            model.add_node(Activation('linear'), 'r_full', inputs=[r_full, 'xr3d'], merge_mode='concat', concat_axis=-1)
            r_full = 'r_full'

    # ===================== [w_full_dim, q_full_dim] -> [class, rel]
    if c['class_mode']:
        model.add_node(TimeDistributedDense(1, activation='sigmoid',
                                            W_regularizer=l2(c['l2reg']),
                                            b_regularizer=l2(c['l2reg'])),
                       'c', input=c_full)
        if classrel_outputs:
            model.add_output(name='class', input='c')
    if c['rel_mode']:
        model.add_node(TimeDistributedDense(1, activation='sigmoid',
                                            W_regularizer=l2(c['l2reg']),
                                            b_regularizer=l2(c['l2reg'])),
                       'r', input=r_full)
        if classrel_outputs:
            model.add_output(name='rel', input='r')

    #model.add_node(SumMask(), 'mask', input='si03d')  # XXX: needs to take se03d into account too
    # ===================== mean of class over rel
    model.add_node(WeightedMean(max_sentences=max_sentences),
                   name='weighted_mean', inputs=['c' if c['class_mode'] else 'mask1', 'r' if c['rel_mode'] else 'mask1', 'mask1'])
    model.add_output(name='score', input='weighted_mean')
    return model


def task():
    return HypEvTask()
