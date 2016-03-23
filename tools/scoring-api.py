#!/usr/bin/python3
#
# Usage:   ./tools/scoring-api.py MODEL_NAME DATASET.CSV WEIGHTSFILE.H5 [PORT] [PARAM=VALUE]...
# Example: ./tools/scoring-api.py rnn data/anssel/yodaqa/full-dataset.csv weights/weights-rnn-38a84de439de6337-bestval.h5 5001 loss=\"binary_crossentropy\"
# This script loads keras model and pre-trained weights for sentence/property selection
# and starts REST API for scoring question - sentence pairs.
# The script listens on given or default (5000) port and accepts JSON in following format:
#    {"question":"QUESTION_TEXT","labels":["SENTENCE1", "SENTENCE2", ...]}
# Example:
#    {"question":"what country is the grand bahama island in","labels":["contained by", "contained in", "contained"]}
# The response is JSON object with key "score" and value containing list of scores, one for each label given
#    {"score":[SCORE1, SCORE2, ...]}

from __future__ import print_function
from __future__ import division

import importlib
import subprocess
import sys
import tempfile
from flask import *
import json
import csv

from keras.layers.convolutional import MaxPooling1D, AveragePooling1D
from keras.layers.recurrent import SimpleRNN, GRU, LSTM
import pysts.embedding as emb
import pysts.eval as ev
import pysts.kerasts.blocks as B
from pysts.kerasts.objectives import ranknet

import anssel_train as anssel_train
import models  # importlib python3 compatibility requirement
from nltk.tokenize import word_tokenize

modelname = sys.argv[1]
trainf = sys.argv[2]
weightsfile = sys.argv[3]
if (len(sys.argv) > 4):
	port = int(sys.argv[4])
else:
	port = 5000
params = sys.argv[5:]

app = Flask(__name__)

def load_jacana(question, prop_labels):
    samples = []
    q = word_tokenize(question.replace("?",""))
    for label in prop_labels:
        text = word_tokenize(label.lower())
        samples.append({'qtext': ' '.join(q), 'label': 0, 'atext': ' '.join(text)})    
    return samples

def write_csv(file, samples):
    outcsv = csv.DictWriter(file, fieldnames=['qtext', 'label', 'atext'])
    outcsv.writeheader()
    for s in samples:
        outcsv.writerow(s)

def load_model(trainf, model_name, weights_file):
    params = []    
    module = importlib.import_module('.'+modelname, 'models')
    conf, ps, h = anssel_train.config(module.config, params)
    s0, s1, y, vocab, gr = anssel_train.load_set(trainf)
    model = anssel_train.build_model(glove, vocab, module.prep_model, conf)
    model.load_weights(weights_file)
    return model, vocab

@app.route('/score', methods=['POST'])
def get_score():
    # print(request.json)
    if (request.json['labels'] == []):
        return jsonify({'score': []}), 200  
    f = tempfile.NamedTemporaryFile(mode='w')
    write_csv(f.file, load_jacana(request.json['question'], request.json['labels']))
    f.file.close()
    s0t, s1t, yt, _, grt = anssel_train.load_set(f.name, vocab, False)
    res = model.predict(grt)['score'][:,0]
    # print(res)
    return jsonify({'score': res.tolist()}), 200

module = importlib.import_module('.'+modelname, 'models')
conf, ps, h = anssel_train.config(module.config, params)
print(conf)

print("Loading GloVe")
glove = emb.GloVe(N=conf['embdim'])

print("Loading model")
model, vocab = load_model(trainf, modelname, weightsfile)

print("Running...")
app.run(port=port, host='::', debug=True, use_reloader=False)
