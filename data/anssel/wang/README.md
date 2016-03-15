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
| termfreq TF-IDF #w       | 0.715266    | 0.725217 | 0.634500 | 0.708957 | ``freq_mode='tf'`` weighed count of common words
| termfreq TF-IDF cos      | 0.601831    | 0.696384 | 0.583100 | 0.634582 | ``freq_mode='tf' score_mode='cos'``
| termfreq BM25 #w         | 0.813992    | 0.829004 | 0.693800 | 0.765363 | (defaults)
| termfreq BM25 cos        | 0.602093    | 0.684234 | 0.641078 | 0.582000 | ``score_mode='cos'``
| avg mean+project (dot)   |             | 0.752555 | 0.630773 | 0.556700 | ``ptscorer=B.dot_ptscorer``
|                          |             |±0.024913 |          |          |
| avg mean+project (MLP)   |             | 0.815309 | 0.729056 | 0.635900 | (defaults)
|                          |             |±0.032678 |          |          |
| avg DAN (MLP)            |             | 0.815309 | 0.729056 | 0.635900 | DAN
|                          |             |          |          |          |
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
