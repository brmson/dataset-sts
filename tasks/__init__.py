from keras.layers.core import Activation
from keras.models import Graph
import pysts.loader as loader
import pysts.kerasts.blocks as B


class AbstractTask:
    def set_conf(self, c):
        self.c = c

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
            self.gr = loader.graph_adapt_ubuntu(self.gr, self.vocab)
            self.grv = loader.graph_adapt_ubuntu(self.grv, self.vocab)
            if self.grt is not None:
                self.grt = loader.graph_adapt_ubuntu(self.grt, self.vocab)

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
            final_outputs = final_outputs[1]
        else:
            ptscorer = self.c['ptscorer']

        kwargs = dict()
        if ptscorer == B.mlp_ptscorer:
            kwargs['sum_mode'] = self.c['mlpsum']
        if self.c['f_add_kw']:
            model.add_input('kw', input_shape=(1,))
            model.add_input('akw', input_shape=(1,))
            kwargs['extra_inp'] = ['kw', 'akw']
        model.add_node(name='scoreS', input=ptscorer(model, final_outputs, self.c['Ddim'], N, self.c['l2reg'], **kwargs),
                       layer=Activation(oact))
        model.add_output(name='score', input='scoreS')
        return model

    def fit_model(self, model, **kwargs):
        return model.fit(self.gr, validation_data=self.grv, **kwargs)
