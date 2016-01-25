#!/usr/bin/python3
"""
An example sts2015/sick2014 classifier using the (Tai, 2015) approach
http://arxiv.org/abs/1503.00075 with mean vectors and model as in 4.2,
using the awesome Keras deep learning library.

Play with it to see effect of GloVe dimensionality, hidden layer
(and its size), various regularization etc.!

TODO: Add sum-features, or even better abs-delta features

TODO: KL cost function

Prerequisites:
    * Get glove.6B.300d.txt from http://nlp.stanford.edu/projects/glove/

Performance (100 iters):
    * sick2014
        4500/4500 [==============================] - 0s - loss: 1.2332 - acc: 0.5016 - val_loss: 1.1926 - val_acc: 0.4770
        Train Pearson: 0.721662
        Train Spearman: 0.593711
        Train MSE: 0.529364
        Test Pearson: 0.701661
        Test Spearman: 0.571329
        Test MSE: 0.540898
    * sts
        9092/9092 [==============================] - 0s - loss: 1.5331 - acc: 0.3917 - val_loss: 1.8354 - val_acc: 0.2080
        Train Pearson: 0.581443
        Train Spearman: 0.525810
        Train MSE: 1.450468
        Test Pearson: 0.345727
        Test Spearman: 0.340871
        Test MSE: 2.489101
"""

from __future__ import print_function

import argparse
import glob

from keras.models import Sequential
from keras.layers.core import Activation, Dense, Dropout, Merge
from keras.regularizers import l2

import pysts.embedding as emb
import pysts.loader as loader
import pysts.eval as ev


def load_set(glove, globmask, loadfun=loader.load_sts):
    s0, s1, labels = loader.concat_datasets([loadfun(d) for d in glob.glob(globmask)])
    print('(%s) Loaded dataset: %d' % (globmask, len(s0)))
    e0, e1, s0, s1, labels = loader.load_embedded(glove, s0, s1, labels)
    return ([e0, e1], labels)


def prep_model(glove, dropout=0, l2reg=1e-4):
    model0 = Sequential()  # s0
    model1 = Sequential()  # s1
    model0.add(Dropout(dropout, input_shape=(glove.N,)))
    model1.add(Dropout(dropout, input_shape=(glove.N,)))

    model = Sequential()
    model.add(Merge([model0, model1], mode='mul'))
    model.add(Dense(50, W_regularizer=l2(l2reg)))
    model.add(Activation('sigmoid'))
    model.add(Dense(6, W_regularizer=l2(l2reg)))
    model.add(Activation('softmax'))
    return model


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Benchmark yu1412 on semantic relatedness regression-classification (sts)")
    parser.add_argument("-N", help="GloVe dim", type=int, default=300)
    parser.add_argument("--sick", help="whether to run on SICK2014 inst. of sts2012-14/15 dataset", type=int, default=0)
    args = parser.parse_args()

    glove = emb.GloVe(N=args.N)
    if args.sick == 1:
        Xtrain, ytrain = load_set(glove, 'sick2014/SICK_train.txt', loader.load_sick2014)
        Xtest, ytest = load_set(glove, 'sick2014/SICK_test_annotated.txt', loader.load_sick2014)
    else:
        Xtrain, ytrain = load_set(glove, 'sts/all/201[0-4]*')
        Xtest, ytest = load_set(glove, 'sts/all/2015*')

    model = prep_model(glove)
    model.compile(loss='categorical_crossentropy', optimizer='adam')
    model.fit(Xtrain, loader.sts_labels2categorical(ytrain), batch_size=80, nb_epoch=100, show_accuracy=True,
              validation_data=(Xtest, loader.sts_labels2categorical(ytest)))
    ev.eval_sts(model, Xtrain, ytrain, 'Train')
    ev.eval_sts(model, Xtest, ytest, 'Test')


"""
Performance tuning (100 iters) on sick2014:

  * Just elementwise-mul:

    4500/4500 [==============================] - 0s - loss: 1.2332 - acc: 0.5016 - val_loss: 1.1926 - val_acc: 0.4770
    Train Pearson: 0.721662
    Train Spearman: 0.593711
    Train MSE: 0.529364
    Test Pearson: 0.701661
    Test Spearman: 0.571329
    Test MSE: 0.540898
"""
