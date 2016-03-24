KeraSTS Baseline Models
=======================

You can find some baseline deep learning models for text similarity here.
The models are meant to be task-invariant, that is, you should be able
to train the same model on Answer Sentence Selection, STS, SICK, or the
Ubuntu dialogue.  To this end, the models use the KeraSTS toolkit.

We say these are "baselines", but they are supported to replicate
state-of-art results.

You are encouraged to contribute more strong models, especially if you
publish a paper on them with the help of this repository.

Performance of the models on individual tasks are listed in the tasks
respective READMEs.

Model API
---------

Each model should provide two methods:

  * ``config(conf)`` takes a dict with the model configuration and initializes
    it with default parameter values; some parameters are used external
    to the model itself (like ``inp_e_dropout``)

  * ``prep_model(model, N, s0pad, s1pad, conf)`` takes a model that is
    a Graph with ``e0_``, ``e1_`` layers already included that represent
    the compared s0pad (s1pad) long sequences of (flagged) N-dimensional
    embeddings.
    The output of the model should be another pair of vectors (layer names
    returned as a tuple from this function) that are further compared in
    a task-specific way (often also parametrizable) to generate the score.

It's probably much easier to just look at the examples below.

Models
------

To try out any given model, use task-specific training tools in the **tools/**
directory.

  * **termfreq.py** is a trivial non-neural baseline that just determines
    word-level overlap between sentences - TFIDF or BM25
  * **avg.py** is a trivial baseline that produces bag-of-words average
    embedding across all input words, then projects it to a similarity
    vector space; this is analogous to [(Yu, 1412.1632)](http://arxiv.org/abs/1412.1632)
    as well as the MemN2N model; it also supports the Deep Averaging Networks
  * **rnn.py** is a simple model that generates summary sentence embeddings
    based on GRU hidden states
  * **cnn.py** is another simple model that generates summary sentence
    embeddings using a CNN and max-pooling
  * **cnnrnn.py** uses a model popular in Keras examples, using CNN to
    "smear" the input sentence, then RNN on top of that to generate the
    summary sentence embedding
  * **rnncnn.py** uses a model introduced in
    [Tan et al., 1511.04108](http://arxiv.org/abs/1511.04108)), CNN on
    top of an RNN sequence output - simple combination of rnn and cnn
  * **attn1511.py** is a modular and configurable pipeline using RNNs, CNNs
    and attention to generate sentence embeddings that depend on the other
    sentence as well (reimplementation and extension of
    [Tan et al., 1511.04108](http://arxiv.org/abs/1511.04108))

Model Wishlist
--------------

Roughly with (imho) the most interesting coming first in each category.

### "Simple" Models

  * [Sentence Similarity Learning by Lexical Decomposition and Composition](http://arxiv.org/pdf/1602.07019v1.pdf) (anssel)
  * [Tree-LSTM](http://arxiv.org/abs/1503.00075) (anssel)
  * [Denoising Bodies to Titles: Retrieving Similar Questions with Recurrent Convolutional Models](http://arxiv.org/abs/1512.05726) (para)
  * [Learning to Rank Short Text Pairs with Convolutional Deep Neural Networks](http://disi.unitn.it/~severyn/papers/sigir-2015-long.pdf) (anssel)
  * [Learning Hybrid Representations to Retrieve Semantically Equivalent Questions](http://www.aclweb.org/anthology/P15-2114) (para)
  * [Siamese Recurrent Architectures for Learning Sentence Similarity](https://pdfs.semanticscholar.org/6812/fb9ef1c2dad497684a9020d8292041a639ff.pdf) (sts, ent)

### Attention-based Models

  * [Reasoning about Entailment with Neural Attention](http://arxiv.org/abs/1509.06664) and [followups](http://nlp.stanford.edu/projects/snli/) (ent)
  * [Attentive Pooling Networks](http://arxiv.org/abs/1602.03609) (anssel)
  * [ABCNN: Attention-Based Convolutional Neural Network for Modeling Sentence Pairs](http://arxiv.org/pdf/1512.05193v2.pdf) (anssel, para, ent)
    and [Attention-Based Convolutional Neural Network for Machine Comprehension](http://www.aclweb.org/anthology/P15-2114) (hypev)
    (what are the differences? also seems similar to attentive pooling above)
  * [Multi-Perspective Sentence Similarity Modeling with Convolutional Neural Networks](http://ttic.uchicago.edu/~kgimpel/papers/he+etal.emnlp15.pdf) (para, sts)
