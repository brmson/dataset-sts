Answer Sentence Selection - Classic Dataset
===========================================

In Question Answering systems on top of unstructured corpora, one task is
selecting the sentences in corpora that are most likely to carry an answer
to a given question.

The classic academic standard stems from the TREC-based dataset originally
by Wang et al., 2007, in the form by Yao et al., 2013 as downloaded from
https://code.google.com/p/jacana/

Model Comparison
----------------

For third-party measurements, see

	http://aclweb.org/aclwiki/index.php?title=Question_Answering_(State_of_the_art)

All models are trained on train-all.  For randomized models, 95% confidence
intervals (t-distribution) are reported.

| Model                    | trainAllMRR | devMRR   | testMAP  | testMRR  | settings
|--------------------------|-------------|----------|----------|----------|---------
| termfreq TF-IDF #w       | 0.714169    | 0.725217 | 0.578200 | 0.708957 | ``freq_mode='tf'``
| termfreq BM25 #w         | 0.813992    | 0.829004 | 0.630100 | 0.765363 | (defaults)
|--------------------------|-------------|----------|----------|----------|---------
| avg                      | 0.786983    | 0.799939 | 0.607031 | 0.689948 | (defaults)
|                          |±0.019449    |±0.007218 |±0.005516 |±0.009912 |
| DAN                      | 0.838842    | 0.828035 | 0.643288 | 0.734727 | ``inp_e_dropout=0`` ``inp_w_dropout=1/3`` ``deep=2`` ``pact='relu'``
|                          |±0.013775    |±0.007839 |±0.009993 |±0.008747 |
|--------------------------|-------------|----------|----------|----------|---------

Dataset
-------

This dataset has been imported from

	jacana/tree-edit-data/answerSelectionExperiments/data

pseudo-XML files to a much more trivial CSV format compatible with anssel-yodaqa/.

In harmony with the other papers, we use the filtered version that keeps
only samples with less than 40 tokens.  "train-all" is a larger dataset with
noisy labels (automatically rather than manually labelled).

Note that all the papers use only questions that have at least one positive
and one negative sample for training and evaluation.  This is the default
mode of pysts load_anssel().

Licence
-------

Derived from TREC questions, which are public domain.
