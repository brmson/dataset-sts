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
| Tan (2015)               |             |          | 0.728    | 0.832    | QA-LSTM/CNN+attention; state-of-art 2015
| dos Santos (2016)        |             |          | 0.753    | 0.851    | Attentive Pooling CNN; state-of-art 2016
| Wang et al. (2016)       |             |          | 0.771    | 0.845    | Lexical Decomposition and Composition; state-of-art 2016
|--------------------------|-------------|----------|----------|----------|---------
| avg                      | 0.786983    | 0.799939 | 0.607031 | 0.689948 | (defaults)
|                          |±0.019449    |±0.007218 |±0.005516 |±0.009912 |
| DAN                      | 0.838842    | 0.828035 | 0.643288 | 0.734727 | ``inp_e_dropout=0`` ``inp_w_dropout=1/3`` ``deep=2`` ``pact='relu'``
|                          |±0.013775    |±0.007839 |±0.009993 |±0.008747 |
|--------------------------|-------------|----------|----------|----------|---------
| rnn                      | 0.791770    | 0.842155 | 0.648863 | 0.742747 | (defaults)
|                          |±0.017036    |±0.009447 |±0.010918 |±0.009896 |
| cnn                      | 0.845162    | 0.841343 | 0.690906 | 0.770042 | (defaults)
|                          |±0.015552    |±0.005409 |±0.006910 |±0.010381 |
| rnncnn                   | 0.922721    | 0.849363 | 0.716519 | 0.797826 | (defaults)
|                          |±0.019407    |±0.006259 |±0.007169 |±0.011460 |
| attn1511                 | 0.852364    | 0.851368 | 0.708163 | 0.789822 | (defaults)
|                          |±0.017280    |±0.005533 |±0.008958 |±0.013308 |
|--------------------------|-------------|----------|----------|----------|---------
| rnn                      | 0.895331    | 0.872205 | 0.731038 | 0.814410 | Ubuntu transfer learning (``ptscorer=B.dot_ptscorer`` ``pdim=1`` ``inp_e_dropout=0`` ``dropout=0`` ``balance_class=True`` ``adapt_ubuntu=True`` ``opt='rmsprop'``)
|                          |±0.006360    |±0.004435 |±0.007483 |±0.008340 |
|--------------------------|-------------|----------|----------|----------|---------
| avg + BM25               | 0.880256    | 0.857196 | 0.583656 | 0.809148 |
|                          |±0.006249    |±0.004471 |±0.003216 |±0.006647 |
| DAN + BM25               | 0.884725    | 0.854399 | 0.571438 | 0.790041 | ``inp_e_dropout=0`` ``inp_w_dropout=1/3`` ``deep=2`` ``pact='relu'``
|                          |±0.010711    |±0.006776 |±0.005435 |±0.009364 |
| rnn + BM25               | 0.855969    | 0.858265 | 0.574919 | 0.809619 |
|                          |±0.004530    |±0.004576 |±0.003262 |±0.006840 |
| cnn + BM25               | 0.957526    | 0.872277 | 0.578244 | 0.789422 | ``dropout=0`` ``inp_e_dropout=0``
|                          |±0.012634    |±0.006433 |±0.004377 |±0.009193 |
| rnncnn + BM25            | 0.924293    | 0.855948 | 0.576938 | 0.794583 |
|                          |±0.015503    |±0.006826 |±0.004248 |±0.007316 |
| attn1511 + BM25          | 0.909971    | 0.868992 | 0.592856 | 0.816968 |
|                          |±0.009855    |±0.004665 |±0.004913 |±0.011196 |


These results are obtained like this:

	tools/train.py avg anssel data/anssel/wang/train-all.csv data/anssel/wang/dev.csv nb_runs=16
	tools/eval.py avg anssel data/anssel/wang/train-all.csv data/anssel/wang/dev.csv data/anssel/wang/test.csv weights-anssel-avg--69489c8dc3b6ce11-*-bestval.h5

BM25 results are obtained with:

	"prescoring='termfreq'" "prescoring_weightsf='weights-anssel-termfreq-3368350fbcab42e4-bestval.h5'" "prescoring_input='bm25'" "f_add=['bm25']" prescoring_prune=20


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
