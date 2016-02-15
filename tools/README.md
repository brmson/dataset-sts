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

