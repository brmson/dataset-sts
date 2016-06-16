#!/usr/bin/python3
#
# This script loads keras model and pre-trained weights for hypothesis
# evaluation and starts REST API for scoring hypotheses based on evidence
#
# Usage:   ./tools/hypev-api.py MODEL TASK VOCAB WEIGHTS [PARAM=VALUE]...
# Example: ./tools/hypev-api.py attn1511 hypev data/hypev/argus/argus_train.csv weights-hypev-attn1511--2f4946b5cbace063-00-bestval.h5 "focus_act='sigmoid/maxnorm'" "cnnact='relu'" nb_runs=4 aux_c=True aux_r=True
# Example: ./tools/hypev-api.py rnn hypev data/anssel/ubuntu/v2-vocab.pickle weights-ubuntu-hypev-rnn-19a300a6532a4345-bestval.h5 "vocabt='ubuntu'" pdim=1 ptscorer=B.mlp_ptscorer aux_c=True aux_r=True
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
#    curl -H "Content-Type: application/json" -X POST -d '{"s0":"how much it pays","k":3}' http://localhost:5052/score
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
from pysts import loader

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


@app.route('/weights/<node_name>')
def get_weights(node_name):
    W = model.nodes[node_name].get_weights()[0]
    print(W)
    return jsonify({node_name: [float(x) for x in W[:, 0]]}), 200


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
        l_xtra = loader.load_hypev_xtra(request.json['s1'])
    else:
        l_xtra = None
    l_types = None
    lists = l_s0, l_s1, l_y, l_qids, l_xtra, l_types

    s0, s1, y, qids, xtra, types = lists
    gr, y, _ = task.load_set(None, lists=lists)

    cl, rel, sc = [], [], []
    for ogr in task.sample_pairs(gr, 16384, shuffle=False, once=True):
        cl += list(model.predict(ogr)['class'])
        rel += list(model.predict(ogr)['rel'])
        sc += list(model.predict(ogr)['score'])
    return jsonify({'score': sc[0].tolist()[0], 'class': [x[0] for x in cl[0].tolist()], 'rel': [x[0] for x in rel[0].tolist()]}), 200


if __name__ == "__main__":
    modelname, taskname, vocabf, weightsf = sys.argv[1:5]
    params = sys.argv[5:]

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
    if 'vocabt' in conf:
        taskv_module = importlib.import_module('.'+conf['vocabt'], 'tasks')
        taskv = taskv_module.task()
        taskv.load_vocab(vocabf)
        task.vocab = taskv.vocab
    else:
        task.load_vocab(vocabf)

    print('Model')
    task.c['skip_oneclass'] = False  # load_samples() returns oneclass
    model = task.build_model(model_module.prep_model, classrel_outputs=True)

    print(weightsf)
    model.load_weights(weightsf)

    print("Running...")
    app.run(port=conf.get('port', 5052), host='::', debug=True, use_reloader=False)
