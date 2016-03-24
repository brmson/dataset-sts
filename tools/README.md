KeraSTS Training and Evaluation Tools
=====================================

Along with the datasets, we ship a variety of models that implement the *f_2*
text similarity functions.  These models should be task-independent, and they
can be run and applied on a specific task (family of datasets) using the tools
below.

**Getting Started:**
Basically, you pick a model from the **models/** directory, then pass its
name, dataset paths and possibly some parameter tweaks to the training tool
to train the model on that task/dataset and show validation set performance.

**Legacy:**
In the past, each task had a separate suite of scripts.  We are currently
converting the tasks to the generic interface, but meanwhile their scripts
are listed in sections below.

Training
--------

The basic thing you are going to need is to train a model on a task --- using
a training split of a dataset and evaluating it on the validation split.

Usage: ``tools/train.py MODEL TASK TRAINDATA VALDATA [PARAM=VALUE]...``

Example: ``tools/train.py cnn para data/para/msr/msr-para-train.tsv data/para/msr/msr-para-val.tsv inp_e_dropout=1/2 nb_epoch=64``

Evaluation
----------

The training script above creates a file with weights dump for a particular
trained model instance (or instances, with nb_runs parameter).  You can
use a set of instances to evaluate performance on datasets, statistically.

Usage: ``tools/eval.py MODEL TASK VOCABDATA TRAINDATA VALDATA TESTDATA WEIGHTFILES... [PARAM=VALUE]...``

Example: ``tools/eval.py cnn para data/para/msr/msr-para-train.tsv data/para/msr/msr-para-train.tsv data/para/msr/msr-para-val.tsv - weights-para-cnn--69489c8dc3b6ce11-*``

(Instead of -, you can pass a test set for the very final evaluation.)

A custom dataset for constructing the vocabulary can be passed as a config
argument like:

	"vocabf='data/para/msr/msr-para-train.tsv'"

(This is useful for evaluating a model on a different dataset than how it
was trained, or for datasets with external vocabulary, like ubuntu.)


Task: Answer Sentence Selection
-------------------------------

Please use the task-generic interface for basic tasks (training, eval).

  * **anssel_tuning.py** tunes the model parameters using random search

  * **anssel-visual.ipynb** showcases IPython/Jupyter notebook for model
    development and visualizing attention models using a token heatmap

The Ubuntu dialogue corpus is the same task, but uses dedicated scripts to be
able to process large datasets.
Please use the task-generic interface for basic tasks (training, eval).

  * **ubuntu_anssel_transfer.py** to adapt a model trained on an Ubuntu Dialog
    dataset towards a particular anssel task instance

  * **ubuntu_para_transfer.py** to adapt a model trained on Ubuntu Dialog
    dataset onto a Paraphrasing dataset

  * **ubuntu_sts_transfer.py** to adapt a model trained on Ubuntu Dialog
    dataset onto an STS dataset

  * **ubuntu-visual.ipynb** is a simple transposition of anssel-visual for
    attention visualization on the Ubuntu corpus


Task: Semantic Text Similarity
------------------------------

The main differences to the anssel task are that (i) this task is symmetric
and both sentences should be considered from the same viewpoint; (ii) the
output is a number between 0 and 5 and Pearson correlation is the metric.

  * **sts_train.py** to train a model on 2012-2014 and evaluate on 2015:

	tools/sts_train.py rnn data/sts/semeval-sts/all/201[-4].[^t]* -- data/sts/semeval-sts/all/2014.tweet-news.test.tsv

  * **sts_fineval.py** evaluates the model N times on all sets, producing
    statistical measurements suitable for publication


Task: Paraphrasing
------------------

This task is like the STS task, but rather than regressing a numerical score
on output, it is a binary classification task.

Please use the task-generic interface.


Task: Hypothesis Evaluation
---------------------------

This task is like "Answer Sentence Selection" in that we have many s1 for
a single s0, but the goal is to produce an aggregate judgement on s0 based
on the pairs.

  * **hypev_train.py** to train a model e.g. on the argus dataset:

	tools/hypev_train.py rnn data/hypev/argus/argus_train.csv data/hypev/argus/argus_test.csv
