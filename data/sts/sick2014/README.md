http://alt.qcri.org/semeval2014/task1/

This dataset is different from the STS dataset in that it aims to capture
only similarities on purely language and common knowledgde level, without
relying on domain knowledge.  There are no named entities or multi-word
idioms, and the models should know that "a couple is formed by a bride
and a groom" but not that "the current president of the US is Barack Obama".

To evaluate using KeraSTS, refer to ``tools/sts_train.py``.
Otherwise, for a Python code example for evaluation and processing, see
https://github.com/ryankiros/skip-thoughts/blob/master/eval_sick.py

STS Model Comparison
--------------------

For randomized models, 95% confidence intervals (t-distribution) are reported.
t. mean is the same as test (artifact of sts-semeval oriented evaluation).

Also NOTE THAT THESE RESULTS ARE OBSOLETE because they predate the f/bigvocab port.

| Model                    | train    | trial    | test     | settings
|--------------------------|----------|----------|----------|---------
| termfreq TF-IDF #w       | 0.479906 | 0.456354 | 0.478802 | ``freq_mode='tf'``
| termfreq BM25 #w         | 0.476338 | 0.458441 | 0.474453 | (defaults)
| ECNU run1                |          |          | 0.8414   | STS2014 winner
| Kiros et al. (2015)      |          |          | 0.8655   | skip-thoughts
| Mueller et al. (2016)    |          |          | 0.8822   | MaLSTM; state-of-art
|--------------------------|----------|----------|----------|---------
| avg                      | 0.722639 | 0.625970 | 0.621415 | (defaults)
|                          |±0.035858 |±0.015800 |±0.017411 |
| DAN                      | 0.730100 | 0.648327 | 0.641699 | ``inp_e_dropout=0`` ``inp_w_dropout=1/3`` ``deep=2`` ``pact='relu'``
|                          |±0.023870 |±0.012275 |±0.015708 |
|--------------------------|----------|----------|----------|---------
| rnn                      | 0.732334 | 0.684798 | 0.663615 | (defaults)
|                          |±0.035202 |±0.016028 |±0.022356 |
| cnn                      | 0.923268 | 0.757300 | 0.762184 | (defaults)
|                          |±0.028907 |±0.010743 |±0.006005 |
| rnncnn                   | 0.940838 | 0.784702 | 0.790240 | (defaults)
|                          |±0.012955 |±0.002831 |±0.004763 |
| attn1511                 | 0.835484 | 0.732324 | 0.722736 | (defaults)
|                          |±0.023749 |±0.011295 |±0.009434 |
|--------------------------|----------|----------|----------|---------
| rnn                      | 0.946294 | 0.792281 | 0.799129 | Ubuntu transfer learning (``ptscorer=B.dot_ptscorer`` ``pdim=1`` ``inp_e_dropout=0`` ``dropout=0`` ``adapt_ubuntu=False``)
|                          |±0.018979 |±0.009483 |±0.009060 |
| rnn                      | 0.936499 | 0.787830 | 0.798314 | SNLI transfer learning (``dropout=0`` ``inp_e_dropout=0``)
|                          |±0.042884 |±0.007839 |±0.007330 |

These results are obtained like this:

	tools/train.py avg sts data/sts/sick2014/SICK_train.txt data/sts/sick2014/SICK_trial.txt nb_runs=16
	tools/eval.py avg sts data/sts/sick2014/SICK_train.txt data/sts/sick2014/SICK_trial.txt data/sts/sick2014/SICK_test_annotated.txt weights-sts-avg--69489c8dc3b6ce11-*-bestval.h5

RTE Model Comparison
--------------------

Reporting accuracy...

| Model                    | train    | trial    | test     | settings
|--------------------------|----------|----------|----------|---------
| Bowman et al.(2015) Lex  | 0.904    |          | 0.778    | (the "Lexicalized" Excitement-derived baseline)
| Bowman et al.(2015) LSTM | 1.000    |          | 0.713    | (the proposed 100d LSTM model)
| Bowman et al.(2015) SNLI | 0.999    |          | 0.808    | (the proposed 100d LSTM model transferred from SNLI)
| Lai et Hockenmaier (2014)|          | 0.842    | 0.845    | (the SemEval 2014 Task 1 winner)
|--------------------------|----------|----------|----------|---------
| avg                      | 0.770347 | 0.678750 | 0.652172 | (defaults)
|                          |±0.020479 |±0.018501 |±0.016917 |
| DAN                      | 0.714889 | 0.682625 | 0.662472 | ``inp_e_dropout=0`` ``inp_w_dropout=1/3`` ``deep=2`` ``pact='relu'``
|                          |±0.009707 |±0.003892 |±0.002700 |
| rnn                      | 0.759486 | 0.738875 | 0.731568 | (defaults)
|                          |±0.015704 |±0.014402 |±0.010432 |
| cnn                      | 0.926708 | 0.822375 | 0.799104 | (defaults)
|                          |±0.008168 |±0.002850 |±0.003824 |
| rnncnn                   | 0.765472 | 0.717750 | 0.709420 | (defaults)
|                          |±0.084148 |±0.060789 |±0.058780 |
| attn1511                 | 0.857792 | 0.783875 | 0.766757 | ``ptscorer='1'``
|                          |±0.010444 |±0.005104 |±0.004373 |
|--------------------------|----------|----------|----------|---------
| rnn                      | 0.930833 | 0.829750 | 0.812614 | Ubuntu transfer learning (``pdim=1`` ``ptscorer=B.mlp_ptscorer`` ``dropout=0`` ``inp_e_dropout=0`` ``adapt_ubuntu=True``)
|                          |±0.017211 |±0.007164 |±0.004619 |
| rnn                      | 0.926556 | 0.831625 | 0.830703 | SNLI transfer learning (``dropout=0`` ``inp_e_dropout=0``)
|                          |±0.005925 |±0.003019 |±0.001528 |


These results are obtained like this:

	tools/train.py avg rte data/sts/sick2014/SICK_train.txt data/sts/sick2014/SICK_trial.txt nb_runs=16
	tools/eval.py avg rte data/sts/sick2014/SICK_train.txt data/sts/sick2014/SICK_trial.txt data/sts/sick2014/SICK_test_annotated.txt weights-rte-avg--69489c8dc3b6ce11-*-bestval.h5

