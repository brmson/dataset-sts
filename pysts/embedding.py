"""
A toolkit you may find useful for mapping sentences to embeddings.

Download and unzip the standard GloVe embeddings to use this.

Skip-thoughts use unigram/bigram information from the Children Book dataset.
"""


from __future__ import print_function

import numpy as np


class Embedder(object):
    """ Generic embedding interface.

    Required: attributes g and N """

    def map_tokens(self, tokens, ndim=2):
        """ for the given list of tokens, return a list of GloVe embeddings,
        or a single plain bag-of-words average embedding if ndim=1.

        Unseen words (that's actually *very* rare) are mapped to 0-vectors. """
        gtokens = [self.g[t] for t in tokens if t in self.g]
        if not gtokens:
            return np.zeros((1, self.N)) if ndim == 2 else np.zeros(self.N)
        gtokens = np.array(gtokens)
        if ndim == 2:
            return gtokens
        else:
            return gtokens.mean(axis=0)

    def map_set(self, ss, ndim=2):
        """ apply map_tokens on a whole set of sentences """
        return [self.map_tokens(s, ndim=ndim) for s in ss]

    def pad_set(self, ss, spad, N=None):
        """ Given a set of sentences transformed to per-word embeddings
        (using glove.map_set()), convert them to a 3D matrix with fixed
        sentence sizes - padded or trimmed to spad embeddings per sentence.

        Output is a tensor of shape (len(ss), spad, N).

        To determine spad, use something like
            np.sort([np.shape(s) for s in s0], axis=0)[-1000]
        so that typically everything fits, but you don't go to absurd lengths
        to accomodate outliers.
        """
        ss2 = []
        if N is None:
            N = self.N
        for s in ss:
            if spad > s.shape[0]:
                if s.ndim == 2:
                    s = np.vstack((s, np.zeros((spad - s.shape[0], N))))
                else:  # pad non-embeddings (e.g. toklabels) too
                    s = np.hstack((s, np.zeros(spad - s.shape[0])))
            elif spad < s.shape[0]:
                s = s[:spad]
            ss2.append(s)
        return np.array(ss2)


class GloVe(Embedder):
    """ A GloVe dictionary and the associated N-dimensional vector space """
    def __init__(self, N=300, glovepath='glove.6B.%dd.txt'):
        """ Load GloVe dictionary from the standard distributed text file.

        Glovepath should contain %d, which is substituted for the embedding
        dimension N. """
        self.N = N
        self.g = dict()
        self.glovepath = glovepath % (N,)

        with open(self.glovepath, 'r') as f:
            for line in f:
                l = line.split()
                word = l[0]
                self.g[word] = np.array(l[1:]).astype(float)


class Word2Vec(Embedder):
    """ A word2vec dictionary and the associated N-dimensional vector space """
    def __init__(self, N=300, w2vpath='GoogleNews-vectors-negative%d.bin.gz'):
        """ Load word2vec pretrained dictionary from the binary archive.
        """
        self.N = N
        self.w2vpath = w2vpath % (N,)

        import gensim
        self.g = gensim.models.Word2Vec.load_word2vec_format(self.w2vpath, binary=True)
        assert self.N == self.g.vector_size


class SkipThought(Embedder):
    """Embedding of sentences, using precomputed skip-thought model [1506.06726].
    To set up:
    * Get skipthoughts.py file from https://github.com/ryankiros/skip-thoughts
    * Execute the "Getting started" wgets in its README
    * set up config['skipthoughts_datadir'] with path to dir where these files
        were downloaded
    
    Skip-thoughts use embeddings build from the Children Book dataset.

    Config:
    * config['skipthoughts_uni_bi'] = 'uni' or 'bi' or 'combined'; Two different 
        skipthought versions, or their combination (see original paper for details)"""

    def __init__(self, c=None):
        """Load precomputed model."""
        if not c:
            c = {}
        self.c = c

        import skipthoughts
        self.encode = skipthoughts.encode

        if self.c.get("skipthoughts_datadir"):
            datadir = self.c["skipthoughts_datadir"]
        else:
            raise KeyError("config['skipthoughts_datadir'] is not set")

        # table for memoizing embeddings
        self.cache_table = {}

        self.uni_bi = self.c["skipthoughts_uni_bi"]
        if self.uni_bi in ("uni", "bi"):
            self.N = 2400
        elif self.uni_bi == "combined":
            self.N = 4800
        else:
            raise KeyError("config['skipthoughts_uni_bi'] has invalid value. Possible values: 'uni', 'bi', 'combined'")

        skipthoughts.path_to_models = datadir
        skipthoughts.path_to_tables = datadir
        skipthoughts.path_to_umodel = skipthoughts.path_to_models + 'uni_skip.npz'
        skipthoughts.path_to_bmodel = skipthoughts.path_to_models + 'bi_skip.npz'
        self.st = skipthoughts.load_model()

    def batch_embedding(self, sentences):
        """Precompute batch embeddings of sentences, and remember them for use 
        later (during this run; ie: without saving into file).
        sentences is list of strings."""

        new_sentences = list(set(sentences) - set(self.cache_table.keys()))
        new_sentences = filter(lambda sen: len(sen) > 0, new_sentences)
        embeddings = self.encode(self.st, new_sentences, verbose=False, use_eos=self.c.get("use_eos"))
        assert len(new_sentences) == len(embeddings)
        self.cache_table.update(zip(new_sentences, embeddings))

    def map_tokens(self, tokens, ndim=2):
        """
        Args:
            tokens list of words, together forming a sentence.

        Returns:
            its embedding as a ndarray."""

        assert ndim == 1, "ndim has to be equal to 1 for skipthoughts embedding"

        sentence = " ".join(tokens)
        if sentence in self.cache_table:
            output_vector = self.cache_table[sentence]
        else:
            output_vector, = self.encode(self.st, [sentence, ], verbose=False)
            self.cache_table[sentence] = output_vector
        if self.uni_bi == 'combined':
            return output_vector
        elif self.uni_bi == 'uni':
            return output_vector[:self.N]
        elif self.uni_bi == 'bi':
            return output_vector[self.N:]
        else:
            raise ValueError("skipthoughts_uni_bi has invalid value")


