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

	tools/anssel_train.py attn1511 anssel-wang/train-all.csv anssel-wang/dev.csv

  * **anssel_treceval.py** to measure anssel performance using the official
    ``trec_eval`` tool

  * **anssel_tuning.py** tunes the model parameters using random search

  * **anssel-visual.ipynb** showcases IPython/Jupyter notebook for model
    development and visualizing attention models using a token heatmap


Semantic Text Similarity Task
-----------------------------

The main differences to the anssel task are that (i) this task is symmetric
and both sentences should be considered from the same viewpoint; (ii) the
output is a number between 0 and 5 and Pearson correlation is the metric.

  * **sts_train.py** to train a model on 2012-2014 and evaluate on 2015:

	tools/sts_train.py rnn sts/all/201[0-4]* -- sts/all/2015*
