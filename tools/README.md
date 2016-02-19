KeraSTS Training and Evaluation Tools
=====================================

Along with the datasets, we ship a variety of models that implement the *f_b*
text similarity functions.  These models should be task-independent, and they
can be run and applied on a specific task (family of datasets) using the tools
below.

**Getting Started:**
Basically, you pick a model from the **models/** directory, then pass its
name, dataset paths and possibly some parameter tweaks to the training tool
to train the model on that task/dataset and show validation set performance.


Answer Sentence Selection
-------------------------

  * **annssel_train.py** to train a model and evaluate on validation/dev set:

	tools/anssel_train.py attn1511 data/anssel/wang/train-all.csv data/anssel/wang/dev.csv

  * **anssel_treceval.py** to measure anssel performance using the official
    ``trec_eval`` tool

  * **anssel_tuning.py** tunes the model parameters using random search

  * **anssel-visual.ipynb** showcases IPython/Jupyter notebook for model
    development and visualizing attention models using a token heatmap

The Ubuntu dialogue corpus is the same task, but uses dedicated scripts to be
able to process large datasets:

  * **ubuntu_train.py** to train a model and evaluate on validation/dev set:

	tools/ubuntu_train.py rnn data/anssel/ubuntu/v1-vocab.pickle data/anssel/ubuntu/v1-trainset.pickle data/anssel/ubuntu/v1-valset.pickle

    (see instructions on top of the file re preprocessing csv to the pickle files)


Semantic Text Similarity Task
-----------------------------

The main differences to the anssel task are that (i) this task is symmetric
and both sentences should be considered from the same viewpoint; (ii) the
output is a number between 0 and 5 and Pearson correlation is the metric.

  * **sts_train.py** to train a model on 2012-2014 and evaluate on 2015:

	tools/sts_train.py rnn data/sts/semeval-sts/all/201[0-4]* -- data/sts/semeval-sts/all/2015*
