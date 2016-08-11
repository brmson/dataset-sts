from __future__ import print_function
from __future__ import division

import importlib
from keras.layers.core import Activation
from keras.models import Graph
import numpy as np
import random
import traceback

import pysts.loader as loader
from pysts.kerasts import graph_input_slice, graph_input_prune
import pysts.kerasts.blocks as B


def default_config(model_config, task_config):
    # TODO: Move this to AbstractTask()?
    c = dict()
    c['embdim'] = 300
    c['embprune'] = 100
    c['embicase'] = False
    c['inp_e_dropout'] = 1/2
    c['inp_w_dropout'] = 0
    c['e_add_flags'] = True

    c['ptscorer'] = B.mlp_ptscorer
    c['mlpsum'] = 'sum'
    c['Ddim'] = 2
    c['Dinit'] = 'glorot_uniform'
    c['f_add_kw'] = False

    c['loss'] = 'mse'  # you really want to override this in each task's config()
    c['balance_class'] = False

    c['opt'] = 'adam'
    c['fix_layers'] = []  # mainly useful for transfer learning, or 'emb' to fix embeddings
    c['batch_size'] = 160
    c['nb_epoch'] = 16
    c['nb_runs'] = 1
    c['epoch_fract'] = 1

    c['prescoring'] = None
    c['prescoring_prune'] = None
    c['prescoring_input'] = None

    task_config(c)
    if c.get('task>model', False):  # task config has higher priority than model
        model_config(c)
        task_config(c)
    else:
        model_config(c)
    return c


class AbstractTask(object):
    def set_conf(self, c):
        self.c = c

        if 's0pad' in self.c:
            self.s0pad = self.c['s0pad']
            self.s1pad = self.c['s1pad']
        elif 'spad' in self.c:
            self.spad = self.c['spad']
            self.s0pad = self.c['spad']
            self.s1pad = self.c['spad']

    def load_vocab(self, vocabf):
        _, _, self.vocab = self.load_set(vocabf)
        return self.vocab

    def load_data(self, trainf, valf, testf=None):
        self.trainf = trainf
        self.valf = valf
        self.testf = testf

        self.gr, self.y, self.vocab = self.load_set(trainf)
        self.grv, self.yv, _ = self.load_set(valf)
        if testf is not None:
            self.grt, self.yt, _ = self.load_set(testf)
        else:
            self.grt, self.yt = (None, None)

        if self.c.get('adapt_ubuntu', False):
            self.vocab.add_word('__eou__')
            self.vocab.add_word('__eot__')
            self.gr = loader.graph_adapt_ubuntu(self.gr, self.vocab)
            self.grv = loader.graph_adapt_ubuntu(self.grv, self.vocab)
            if self.grt is not None:
                self.grt = loader.graph_adapt_ubuntu(self.grt, self.vocab)

    def sample_pairs(self, gr, batch_size, shuffle=True, once=False):
        """ A generator that produces random pairs from the dataset """
        try:
            id_N = int((len(gr['si0']) + batch_size-1) / batch_size)
            ids = list(range(id_N))
            while True:
                if shuffle:
                    # XXX: We never swap samples between batches, does it matter?
                    random.shuffle(ids)
                for i in ids:
                    sl = slice(i * batch_size, (i+1) * batch_size)
                    ogr = graph_input_slice(gr, sl)
                    ogr['se0'] = self.emb.map_jset(ogr['sj0'])
                    ogr['se1'] = self.emb.map_jset(ogr['sj1'])
                    # print(sl)
                    # print('<<0>>', ogr['sj0'], ogr['se0'])
                    # print('<<1>>', ogr['sj1'], ogr['se1'])
                    yield ogr
                if once:
                    break
        except Exception:
            traceback.print_exc()

    def prescoring_apply(self, gr, skip_oneclass=False):
        """ Given a gr, prescore the pairs and do either pruning (for each s0,
        keep only top N s1 based on the prescoring) or add the prescore as
        an input. """
        def prescoring_model(prescoring_task, model, conf, weightsf):
            """ Setup and return a pre-scoring model """
            # We just make a task instance with the prescoring model
            # specific config, build the model and apply it;
            # we sometimes don't want the *real* task to do prescoring, e.g. in
            # case of hypev which evaluates whole clusters of same-s0 pairs but
            # prescoring should be done on the individual tasks
            prescore_task = prescoring_task()

            # We also set up the config appropriately to the model + task.
            prescoring_module = importlib.import_module('.'+model, 'models')
            c = default_config(prescoring_module.config, prescore_task.config)
            for k, v in conf.items():
                c[k] = v
            prescore_task.set_conf(c)

            print('[Prescoring] Model')
            model = prescore_task.build_model(prescoring_module.prep_model)

            print('[Prescoring] ' + weightsf)
            model.load_weights(weightsf)
            return model

        if 'prescoring' not in self.c or (self.c['prescoring_prune'] is None and self.c['prescoring_input'] is None):
            return gr  # nothing to do

        if 'prescoring_model_inst' not in self.c:
            # cache the prescoring model instance
            self.c['prescoring_model_inst'] = prescoring_model(self.prescoring_task, self.c['prescoring'],
                    self.c.get('prescoring_conf', {}), self.c['prescoring_weightsf'])
        print('[Prescoring] Predict')
        ypred = self.c['prescoring_model_inst'].predict(gr)['score'][:,0]

        if self.c['prescoring_input'] is not None:
            inp = self.c['prescoring_input']
            gr[inp] = np.reshape(ypred, (len(ypred), 1))  # 1D to 2D

        if self.c['prescoring_prune'] is not None:
            N = self.c['prescoring_prune']
            print('[Prescoring] Prune')
            gr = graph_input_prune(gr, ypred, N, skip_oneclass=skip_oneclass)

        return gr

    def prep_model(self, module_prep_model, oact='sigmoid'):
        # Input embedding and encoding
        model = Graph()
        N = B.embedding(model, self.emb, self.vocab, self.s0pad, self.s1pad,
                        self.c['inp_e_dropout'], self.c['inp_w_dropout'], add_flags=self.c['e_add_flags'])

        # Sentence-aggregate embeddings
        final_outputs = module_prep_model(model, N, self.s0pad, self.s1pad, self.c)

        # Measurement

        if self.c['ptscorer'] == '1':
            # special scoring mode just based on the answer
            # (assuming that the question match is carried over to the answer
            # via attention or another mechanism)
            ptscorer = B.cat_ptscorer
            final_outputs = [final_outputs[1]]
        else:
            ptscorer = self.c['ptscorer']

        kwargs = dict()
        if ptscorer == B.mlp_ptscorer:
            kwargs['sum_mode'] = self.c['mlpsum']
            kwargs['Dinit'] = self.c['Dinit']
        if 'f_add' in self.c:
            for inp in self.c['f_add']:
                model.add_input(inp, input_shape=(1,))  # assumed scalar
            kwargs['extra_inp'] = self.c['f_add']
        model.add_node(name='scoreS', input=ptscorer(model, final_outputs, self.c['Ddim'], N, self.c['l2reg'], **kwargs),
                       layer=Activation(oact))
        model.add_output(name='score', input='scoreS')
        return model

    def fit_model(self, model, **kwargs):
        if self.c['ptscorer'] is None:
            return model.fit(self.gr, **kwargs)
        batch_size = kwargs.pop('batch_size')
        kwargs['callbacks'] = self.fit_callbacks(kwargs.pop('weightsf'))
        return model.fit_generator(self.sample_pairs(self.gr, batch_size), **kwargs)

    def predict(self, model, gr):
        if self.c['ptscorer'] is None:
            return model.predict(gr)
        batch_size = 16384  # XXX: hardcoded
        ypred = []
        for ogr in self.sample_pairs(gr, batch_size, shuffle=False, once=True):
            ypred += list(model.predict(ogr)['score'][:,0])
        return np.array(ypred)
