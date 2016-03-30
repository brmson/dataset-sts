#!/usr/bin/python3
#
# This script loads keras model and pre-trained weights for sentence/property selection
# and starts REST API for scoring question - sentence pairs.
#
# Usage:   ./tools/scoring-api.py MODEL TASK VOCAB WEIGHTS [PORT] [PARAM=VALUE]...
# Example: ./tools/scoring-api.py rnn data/anssel/yodaqa/curatedv2-training.csv weights/weights-rnn-38a84de439de6337-bestval.h5 5001 'loss="binary_crossentropy"'
#
# The script listens on given or default (5000) port and accepts JSON (on http://address:port/score) in the following format:
#    {"qtext":"QUESTION_TEXT","atext":["SENTENCE1", "SENTENCE2", ...]}
# Example:
#    {"qtext:"what country is the grand bahama island in","atext":["contained by", "contained in", "contained"]}
#
# The response is JSON object with key "score" and value containing list of scores, one for each label given
#    {"score":[SCORE1, SCORE2, ...]}

from __future__ import print_function
from __future__ import division

import importlib
from nltk.tokenize import word_tokenize
import subprocess
import sys
import tempfile
from flask import *
import json
import csv

import pysts.embedding as emb
from pysts.kerasts.objectives import ranknet

from train import config
import models  # importlib python3 compatibility requirement
import tasks

# Unused imports for evaluating commandline params
from keras.layers.recurrent import SimpleRNN, GRU, LSTM
from pysts.kerasts.objectives import ranknet, ranksvm, cicerons_1504
import pysts.kerasts.blocks as B

modelname, taskname, vocabf, weightsf = sys.argv[1:5]
if (len(sys.argv) > 5):
	port = int(sys.argv[5])
else:
	port = 5000
params = sys.argv[6:]

app = Flask(__name__)

def load_samples(question, prop_labels):
    samples = []
    q = word_tokenize(question)
    for label in prop_labels:
        text = word_tokenize(label.lower())
        samples.append({'qtext': ' '.join(q), 'label': 0, 'atext': ' '.join(text)})    
    return samples

def write_csv(file, samples):
    outcsv = csv.DictWriter(file, fieldnames=['qtext', 'label', 'atext'])
    outcsv.writeheader()
    for s in samples:
        outcsv.writerow(s)

@app.route('/score', methods=['POST'])
def get_score():
    if (request.json['atext'] == []):
        return jsonify({'score': []}), 200  
    f = tempfile.NamedTemporaryFile(mode='w')
    # FIXME: Avoid temporary files!!!
    write_csv(f.file, load_samples(request.json['qtext'], request.json['atext']))
    f.file.close()
    gr, y, _ = task.load_set(f.name)
    # XXX: 'score' assumed
    res = model.predict(gr)['score'][:,0]
    return jsonify({'score': res.tolist()}), 200

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
app.run(port=port, host='::', debug=True, use_reloader=False)
