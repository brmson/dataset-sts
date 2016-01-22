Answer Sentence Selection - Classic Dataset
===========================================

In Question Answering systems on top of unstructured corpora, one task is
selecting the sentences in corpora that are most likely to carry an answer
to a given question.

The classic academic standard stems from the TREC-based dataset originally
by Wang et al., 2007, in the form by Yao et al., 2013 as downloaded from
https://code.google.com/p/jacana/ - however, it suffers from a variety of
ailments; it's hard to make sense of and process, it's not very high quality,
and most importantly the train/test split is very unbalanced, both in the
kind of sentence pairs and difficulty (train set is a lot harder!).

Therefore, we recommend to develop and test primarily against anssel-yodaqa/
and use this dataset just for comparison to past work.

Dataset
-------

This dataset has been imported from

	jacana/tree-edit-data/answerSelectionExperiments/data

pseudo-XML files to a much more trivial CSV format compatible with anssel-yodaqa/.
In harmony with the other papers, we use the filtered version that keeps
only samples with less than 40 tokens.  "train-all" is a larger dataset with
noisy labels (automatically rather than manually labelled).

Licence
-------

Derived from TREC questions, which are public domain.
