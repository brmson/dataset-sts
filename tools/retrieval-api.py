#!/usr/bin/python3
#
# This script loads keras model and pre-trained weights for the para task
# and starts REST API for scoring questions against fixed set of sentence pairs.
#
# Usage:   ./tools/retrieval-api.py MODEL TASK VOCAB WEIGHTS S1SET [PARAM=VALUE]...
# Example: ./tools/retrieval-api.py rnn para data/para/msr/msr-para-train.tsv weights/weights-rnn-38a84de439de6337-bestval.h5 frontpage-faq.txt 'loss="binary_crossentropy"'
#
# The S1SET should be a text file with one sentence per line; it will be
# tokenized at load time.
#
# The script listens on given (as "port" config parameter) or default (5050)
# port and accepts JSON (on http://address:port/score) in the following format:
#
#    {"s0":"S0TEXT", "k":K}
#
# where K denotes the number of matches to retrieve.
# Example:
#
#    curl -H "Content-Type: application/json" -X POST -d '{"s0":"how much it pays","k":3}' http://localhost:5051/score
#
# The response is JSON object
#    {"matches":[{"s1":"S1TEXT", "score": SCORE}, ...]}
#
# This is a work in progress, the API may not be stable.  para-specific at the moment.

from __future__ import print_function
from __future__ import division

from flask import *
import importlib
from nltk.tokenize import word_tokenize
import numpy as np
import sys

import pysts.embedding as emb

from train import config
import models  # importlib python3 compatibility requirement
import tasks

# Unused imports for evaluating commandline params
from keras.layers.recurrent import SimpleRNN, GRU, LSTM
from pysts.kerasts.objectives import ranknet, ranksvm, cicerons_1504
import pysts.kerasts.blocks as B

# Support compiling rnn on CPU (XXX move to a better, more generic place)
sys.setrecursionlimit(10000)

app = Flask(__name__)

s1texts = []
s1toks = []


def load_s1texts(fname):
    with open(fname) as f:
        for l in f:
            s1texts.append(l.rstrip())
            s1toks.append(word_tokenize(l.rstrip()))


@app.route('/score', methods=['POST'])
def get_score():
    if not request.json['s0']:
        return jsonify({'matches': []}), 200
    s0toks = word_tokenize(request.json['s0'])

    gr, y, _ = task.load_set(None, ([s0toks for _ in s1toks], s1toks, np.array([0 for _ in s1toks])))

    # XXX: 'score' assumed
    res = task.predict(model, gr)
    print(s1texts)
    print(res)
    return jsonify({'matches': sorted([{'s1': s1, 'score': sc} for s1, sc in zip(s1texts, res)], key=lambda x: x['score'], reverse=True)[:int(request.json['k'])]}), 200


if __name__ == "__main__":
    modelname, taskname, vocabf, weightsf, s1f = sys.argv[1:6]
    params = sys.argv[6:]

    load_s1texts(s1f)

    model_module = importlib.import_module('.'+modelname, 'models')
    task_module = importlib.import_module('.'+taskname, 'tasks')
    task = task_module.task()
    conf, ps, h = config(model_module.config, task.config, params)
    task.set_conf(conf)
    print(ps)

    # TODO we should be able to get away with actually *not* loading
    # this at all!
    if conf['embdim'] is not None:
        print('GloVe')
        task.emb = emb.GloVe(N=conf['embdim'])
    else:
        task.emb = None

    print('Dataset')
    task.load_vocab(vocabf)

    print('Model')
    task.c['skip_oneclass'] = False  # load_samples() returns oneclass
    model = task.build_model(model_module.prep_model)

    print(weightsf)
    model.load_weights(weightsf)

    print("Running...")
    app.run(port=conf.get('port', 5051), host='::', debug=True, use_reloader=False)
