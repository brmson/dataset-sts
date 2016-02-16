#!/usr/bin/python3
"""
An example sts2015 classifier using the (Yu, 2014) inspired approach
http://arxiv.org/abs/1412.1632 of training (M, b) such that:

    f(q, a) = sigmoid(q * M * a.T + b)

We use a trivial keras model to train an embedding projection M
while using bag-of-words average for the two sentenes.

The big downside of this approach is that the model doesn't see both
sentences at once, producing only a coarse measure of similarity based
purely the very limited embedding capacity.  This seems to be the
reason the model overfits a lot.

Prerequisites:
    * Get glove.6B.300d.txt from http://nlp.stanford.edu/projects/glove/

Performance (200 iters):
    * sts (2012-2014 train, 2015 test)
        9092/9092 [==============================] - 2s - loss: 1.4630 - acc: 0.4040 - val_loss: 2.1378 - val_acc: 0.1737
        Train Pearson: 0.835212
        Train Spearman: 0.796621
        Train MSE: 0.693983
        Test Pearson: 0.265433
        Test Spearman: 0.285071
        Test MSE: 3.056868
    * sick2014
        4500/4500 [==============================] - 1s - loss: 1.1657 - acc: 0.5389 - val_loss: 1.2519 - val_acc: 0.4589
        Train Pearson: 0.871274
        Train Spearman: 0.807043
        Train MSE: 0.291123
        Test Pearson: 0.690318
        Test Spearman: 0.571486
        Test MSE: 0.572896
"""

from __future__ import print_function

import argparse
import glob

from keras.models import Sequential
from keras.layers.core import Activation, Dense, Dropout, Flatten, Merge, RepeatVector
from keras.regularizers import l2

import pysts.embedding as emb
import pysts.loader as loader
import pysts.eval as ev


def load_set(glove, globmask, loadfun=loader.load_sts):
    s0, s1, labels = loader.concat_datasets([loadfun(d) for d in glob.glob(globmask)])
    print('(%s) Loaded dataset: %d' % (globmask, len(s0)))
    e0, e1, s0, s1, labels = loader.load_embedded(glove, s0, s1, labels)
    return ([e0, e1], labels)


def prep_model(glove, dropout=0.3, l2reg=1e-3):
    # XXX: this is a bit hacky to combine dot-product with six-wise output
    model0 = Sequential()  # s0
    model1 = Sequential()  # s1
    model0.add(Dropout(dropout, input_shape=(glove.N,)))
    model1.add(Dropout(dropout, input_shape=(glove.N,)))
    model0.add(Dense(input_dim=glove.N, output_dim=glove.N, W_regularizer=l2(l2reg)))  # M matrix
    model0.add(RepeatVector(6))  # [nclass]
    model1.add(RepeatVector(6))  # [nclass]

    model = Sequential()
    model.add(Merge([model0, model1], mode='dot', dot_axes=([2], [2])))
    model.add(Flatten())  # 6x6 matrix with cross-activations -> 36 vector
    model.add(Dense(6, W_regularizer=l2(l2reg)))  # 36 vector -> 6 vector, ugh
    model.add(Activation('softmax'))
    return model


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Benchmark yu1412 on semantic relatedness regression-classification (sts)")
    parser.add_argument("-N", help="GloVe dim", type=int, default=300)
    parser.add_argument("--sick", help="whether to run on SICK2014 inst. of sts2012-14/15 dataset", type=int, default=0)
    args = parser.parse_args()

    glove = emb.GloVe(N=args.N)
    if args.sick == 1:
        Xtrain, ytrain = load_set(glove, 'data/sts/sick2014/SICK_train.txt', loader.load_sick2014)
        Xtest, ytest = load_set(glove, 'data/sts/sick2014/SICK_test_annotated.txt', loader.load_sick2014)
    else:
        Xtrain, ytrain = load_set(glove, 'data/sts/semeval-sts/all/201[0-4]*')
        Xtest, ytest = load_set(glove, 'data/sts/semeval-sts/all/2015*')

    model = prep_model(glove)
    model.compile(loss='categorical_crossentropy', optimizer='adam')
    model.fit(Xtrain, loader.sts_labels2categorical(ytrain), batch_size=80, nb_epoch=200, show_accuracy=True,
              validation_data=(Xtest, loader.sts_labels2categorical(ytest)))
    ev.eval_sts(model.predict_proba(Xtrain), ytrain, 'Train')
    ev.eval_sts(model.predict_proba(Xtest), ytest, 'Test')
