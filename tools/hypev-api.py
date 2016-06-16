#!/usr/bin/python3
#
# This script loads keras model and pre-trained weights for hypothesis
# evaluation and starts REST API for scoring hypotheses based on evidence
#
# Usage:   ./tools/hypev-api.py MODEL TASK VOCAB WEIGHTS [PARAM=VALUE]...
# Example: ./tools/hypev-api.py rnn hypev data/hypev/argus/argus_train.csv hypev-attn1511--9df6b60f36f5b00-05
#
# The script listens on given (as "port" config parameter) or default (5052)
# port and accepts JSON (on http://address:port/score) in the following format:
#
#    {"s0":"S0TEXT", "s1":[{"text":"S1TEXT", "#@Subject match": 1.0, ...}, ...]}
#
# S0TEXT should be pre-tokenized, with the tokens separated by spaces.
# The auxiliary feature labels must match column names of the aux file used
# for training.
# Example (FIXME):
#
#    curl -H "Content-Type: application/json" -X POST -d '{"s0":"how much it pays","k":3}' http://localhost:5051/score
#
# The response is JSON object
#    {"score":SCORE}
#
# This is a work in progress, the API may not be stable.

from __future__ import print_function
from __future__ import division

from flask import *
import importlib
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


@app.route('/score', methods=['POST'])
def get_score():
    if not request.json['s0']:
        return jsonify({'matches': []}), 200
    s0toks = request.json['s0'].split(' ')
    s1toks = [s1['text'].split(' ') for s1 in request.json['s1']]

    l_s0 = [s0toks for s1 in s1toks]
    l_s1 = s1toks
    l_y = [0.5 for s1 in s1toks]
    l_qids = [0 for s1 in s1toks]
    if len(request.json['s1'][0]) > 1:
        l_xtra = [loader.load_hypev_xtra(s1) for s1 in request.json['s1']]
    else:
        l_xtra = None
    l_types = None
    lists = l_s0, l_s1, l_y, l_qids, l_xtra, l_types

    s0, s1, y, qids, xtra, types = lists
    gr, y, _ = task.load_set(None, lists)

    sc = task.predict(model, gr)
    return jsonify({'score': sc}), 200


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
    app.run(port=conf.get('port', 5052), host='::', debug=True, use_reloader=False)
