#!/usr/bin/python3
#
# This script loads keras model and pre-trained weights for sentence/property selection
# and starts REST API for scoring question - sentence pairs.
#
# Usage:   ./tools/scoring-api.py MODEL TASK VOCAB WEIGHTS [PARAM=VALUE]...
# Example: ./tools/scoring-api.py rnn data/anssel/yodaqa/curatedv2-training.csv weights/weights-rnn-38a84de439de6337-bestval.h5 'loss="binary_crossentropy"'
#
# The script listens on given (as "port" config parameter) or default (5050)
# port and accepts JSON (on http://address:port/score) in the following format:
#
#    {"qtext":"QUESTION_TEXT","atext":["SENTENCE1", "SENTENCE2", ...]}
#
# Example:
#
#    curl -H "Content-Type: application/json" -X POST \
#       -d '{"qtext":"what country is the grand bahama island in","atext":["contained by", "contained in", "contained"]}' \
#       http://localhost:5001/score
#
# The response is JSON object with key "score" and value containing list of scores, one for each label given
#    {"score":[SCORE1, SCORE2, ...]}
#
# This is a work in progress, the API may not be stable.  anssel-specific at the moment.

from __future__ import print_function
from __future__ import division

from flask import *
import csv
import importlib
from nltk.tokenize import word_tokenize
import sys
import tempfile

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


def load_anssel_samples(qtext, atexts):
    samples = []
    qtext = word_tokenize(qtext)
    for atext in atexts:
        atext = word_tokenize(atext)
        samples.append({'qtext': ' '.join(qtext), 'label': 0, 'atext': ' '.join(atext)})
    return samples


def write_csv(file, samples):
    outcsv = csv.DictWriter(file, fieldnames=['qtext', 'label', 'atext'])
    outcsv.writeheader()
    for s in samples:
        s2 = ({k: v.encode("utf-8") for k, v in s.items() if k != 'label'})
        s2['label'] = s['label']
        outcsv.writerow(s2)



@app.route('/score', methods=['POST'])
def get_score():
    if (request.json['atext'] == []):
        return jsonify({'score': []}), 200

    print("%d atexts for <<%s>>" % (len(request.json['atext']), request.json['qtext']))

    f = tempfile.NamedTemporaryFile(mode='w')
    # FIXME: Avoid temporary files!!!
    write_csv(f.file, load_anssel_samples(request.json['qtext'], request.json['atext']))
    f.file.close()

    gr, y, _ = task.load_set(f.name)

    # XXX: 'score' assumed
    res = task.predict(model, gr)
    return jsonify({'score': res}), 200


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
    task.load_vocab(vocabf)

    print('Model')
    task.c['skip_oneclass'] = False  # load_samples() returns oneclass
    model = task.build_model(model_module.prep_model)

    print(weightsf)
    model.load_weights(weightsf)

    print("Running...")
    app.run(port=conf.get('port', 5050), host='::', debug=True, use_reloader=False)
