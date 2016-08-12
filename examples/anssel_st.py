#!/usr/bin/python3
"""
Simple example of using skipthoughts for answer sentence selection.
example of usage:
python examples/anssel_st.py  --skipthoughts_datadir skip-thoughts/

todo:
- add example of cache_dir

To set up the skipthoughts datadir contents:
    * git clone https://github.com/ryankiros/skip-thoughts
    * Execute the "Getting started" wgets in its README

When starting this example script for the first time, the sentence
level embeddings are generated (about 2 sentences per second, so
long time); they are cached and reused the next time.

"""

from __future__ import print_function

import argparse
import pickle
import os

from keras.models import Model
from keras.layers import Dense, Dropout, Input, merge
from keras.regularizers import l2
from keras.optimizers import Adam

import pysts.embedding as emb
import pysts.eval as ev
import pysts.loader as loader

def load_set(fname, emb, cache_dir=None):
    save_cache = False
    if cache_dir:
        fname_abs = os.path.abspath(fname)
        from hashlib import md5
        cache_filename = "%s/%s.p" % (cache_dir, md5(fname_abs.encode("utf-8")).hexdigest())

        try:
            with open(cache_filename, "rb") as f:
                return pickle.load(f)
        except (IOError, TypeError, KeyError):
            save_cache=True

    s0, s1, y, _, _, _ = loader.load_anssel(fname)
    e0, e1, s0, s1, y = loader.load_embedded(emb, s0, s1, y, balance=True, ndim=1)

    if save_cache:
        with open(cache_filename, "wb") as f:
            pickle.dump((e0, e1, y), f)
    return (e0, e1, y)


def prep_model(N, dropout=0, l2reg=1e-4):
    # Process sentence embeddings
    e0 = Input(shape=(N,), name='e0')
    e1 = Input(shape=(N,), name='e1')
    # dropout here triggers keras error, wtf?

    # Generate element-wise features from the pair
    # (the Activation is a nop, merge_mode is the important part)
    ew_sum = merge([e0, e1], mode='sum')
    ew_mul = merge([e0, e1], mode='mul')
    ew = Dropout(dropout)(merge([ew_sum, ew_mul], mode='concat', concat_axis=-1))

    # Use MLP to generate classes
    hidden = Dense(50, activation='sigmoid', W_regularizer=l2(l2reg))(Dropout(dropout)(ew))
    score = Dense(1, activation='sigmoid', W_regularizer=l2(l2reg), name='score')(hidden)

    return Model(input=[e0, e1], output=score)

if __name__ == "__main__":
    #todo: add all possible arguments (inspire in other examples)
    parser = argparse.ArgumentParser(
        description="Benchmark skip-thoughts on binary classification / point ranking task (anssel-yodaqa)")
    parser.add_argument("--balance", help="whether to manually balance the dataset", type=int, default=1)
    parser.add_argument("--wang", help="whether to run on Wang inst. of YodaQA dataset", type=int, default=0)

    parser.add_argument("--cache_dir", help="directory where to save/load cached datasets", type=str, default="")

    # possible: /storage/ostrava1/home/nadvorj1/skip-thoughts/
    parser.add_argument("--skipthoughts_datadir", help="directory with precomputed Skip_thoughts embeddings (containing bi_skip.npz...)", type=str, default="")
    args = parser.parse_args()

    if args.wang == 1:
        train_filename = "data/anssel/wang/train-all.csv"
        val_filename = "data/anssel/wang/test.csv"
    else:
        train_filename = "data/anssel/yodaqa/curatedv2-training.csv"
        val_filename = "data/anssel/yodaqa/curatedv2-val.csv"

    st = emb.SkipThought(datadir=args.skipthoughts_datadir, uni_bi="combined")
    N = st.N

    e0, e1, y = load_set(train_filename, st, args.cache_dir)
    e0t, e1t, yt = load_set(val_filename, st, args.cache_dir)

    model = prep_model(N)

    model.compile(loss={'score': 'binary_crossentropy'}, optimizer=Adam(lr=0.001), metrics=['accuracy'])
    hist = model.fit({'e0': e0, 'e1': e1}, {'score': y},
                     batch_size=20, nb_epoch=2000,
                     validation_data=({'e0': e0t, 'e1': e1t}, {'score': yt}))

    ev.eval_anssel(model.predict({'e0': e0, 'e1': e1})['score'][:, 0], e0, e1, yt, 'Train')
    ev.eval_anssel(model.predict({'e0': e0t, 'e1': e1t})['score'][:, 0], e0t, e1t, yt, 'Test')
