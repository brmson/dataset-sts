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

  * **avg.py** is a trivial baseline that produces bag-of-words average
    embedding across all input words, then projects it to a similarity
    vector space; this is analogous to [(Yu, 1412.1632)](http://arxiv.org/abs/1412.1632)
    as well as the MemN2N model
  * **rnn.py** is a simple model that generates summary sentence embeddings
    based on GRU hidden states
  * **cnn.py** is another simple model that generates summary sentence
    embeddings using a CNN and max-pooling
  * **cnnrnn.py** uses a model popular in Keras examples, using CNN to
    "smear" the input sentence, then RNN on top of that to generate the
    summary sentence embedding
  * **attn1511.py** is a modular and configurable pipeline using RNNs, CNNs
    and attention to generate sentence embeddings that depend on the other
    sentence as well (reimplementation and extension of
    [Tan et al., 1511.04108](http://arxiv.org/abs/1511.04108))

Model Wishlist
--------------

  * [Attentive Pooling Networks](http://arxiv.org/abs/1602.03609)
  * [Reasoning about Entailment with Neural Attention](http://arxiv.org/abs/1509.06664)
  * [Denoising Bodies to Titles: Retrieving Similar Questions with Recurrent Convolutional Models](http://arxiv.org/abs/1512.05726)
  * [Learning Hybrid Representations to Retrieve Semantically Equivalent Questions](http://www.aclweb.org/anthology/P15-2114)
  * [Learning to Rank Short Text Pairs with Convolutional Deep Neural Networks](http://disi.unitn.it/~severyn/papers/sigir-2015-long.pdf)
  * [Tree-LSTM](http://arxiv.org/abs/1503.00075)
  * [Deep Averaging Networks](http://cs.umd.edu/~miyyer/pubs/2015_acl_dan.pdf)
    should be a trivial extension of avg; just word-level dropout is required
