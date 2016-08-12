#!/usr/bin/python3
"""
An example sts2015/sick2014 classifier using the (Tai, 2015) approach
http://arxiv.org/abs/1503.00075 with mean vectors and model as in 4.2,
using the awesome Keras deep learning library.

Play with it to see effect of GloVe dimensionality, hidden layer
(and its size), various regularization etc.!

TODO: KL cost function

Prerequisites:
    * Get glove.6B.300d.txt from http://nlp.stanford.edu/projects/glove/

Performance (2000 iters):
    * sick2014
        4500/4500 [==============================] - 0s - loss: 0.9471 - val_loss: 1.2636
        Train Pearson: 0.943327
        Train Spearman: 0.933806
        Train MSE: 0.131103
        Test Pearson: 0.747579
        Test Spearman: 0.630783
        Test MSE: 0.450212
    * sts
        9092/9092 [==============================] - 1s - loss: 1.1706 - val_loss: 2.1772
        Train Pearson: 0.919535
        Train Spearman: 0.902290
        Train MSE: 0.381750
        Test Pearson: 0.396603
        Test Spearman: 0.385356
        Test MSE: 2.353272
"""

from __future__ import print_function

import argparse
import glob

from keras.models import Model
from keras.layers import Dense, Dropout, Input, merge
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
    # Process sentence embeddings
    e0 = Input(shape=(glove.N,), name='e0')
    e1 = Input(shape=(glove.N,), name='e1')
    # dropout here triggers keras error, wtf?

    # Generate element-wise features from the pair
    # (the Activation is a nop, merge_mode is the important part)
    ew_sum = merge([e0, e1], mode='sum')
    ew_mul = merge([e0, e1], mode='mul')
    ew = merge([ew_sum, ew_mul], mode='concat', concat_axis=-1)

    # Use MLP to generate classes
    hidden = Dense(glove.N*2, activation='sigmoid', W_regularizer=l2(l2reg), init='identity')(Dropout(dropout)(ew))
    classes = Dense(6, activation='softmax', W_regularizer=l2(l2reg), name='classes')(hidden)

    return Model(input=[e0, e1], output=classes)


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
    model.compile(loss={'classes': 'categorical_crossentropy'}, optimizer='adam', metrics=['accuracy'])
    model.fit({'e0': Xtrain[0], 'e1': Xtrain[1]},
              {'classes': loader.sts_labels2categorical(ytrain)},
              batch_size=20, nb_epoch=100,
              validation_data=({'e0': Xtest[0], 'e1': Xtest[1]}, {'classes': loader.sts_labels2categorical(ytest)}))
    ev.eval_sts(model.predict({'e0': Xtrain[0], 'e1': Xtrain[1]}), ytrain, 'Train')
    ev.eval_sts(model.predict({'e0': Xtest[0], 'e1': Xtest[1]}), ytest, 'Test')


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
