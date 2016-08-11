Microsoft Research Paraphrase Corpus
====================================

See README.html for the original documentation.

Compared to the original documentation, the train and test sets are swapped,
just as in all the recent papers.  An extra validation split is added by
moving the last 500 pairs of the training set.

System Comparison:
http://aclweb.org/aclwiki/index.php?title=Paraphrase_Identification_(State_of_the_art)

Model Comparison
----------------

For randomized models, 95% confidence intervals (t-distribution) are reported.

| Model                    | train acc | train F1 | val acc  | val F1   | test acc | test F1  | settings
|--------------------------|-----------|----------|----------|----------|----------|----------|---------
| always y=1               | 0.673378  | 0.804748 | 0.694000 | 0.818935 | 0.665507 | 0.799025 | (defaults)
| termfreq TF-IDF #w       | 0.694911  | 0.810557 | 0.704000 | 0.817734 | 0.695652 | 0.811625 | ``freq_mode='tf'``
| termfreq BM25 #w         | 0.696588  | 0.812251 | 0.704000 | 0.820388 | 0.695072 | 0.811063 | (defaults)
| Ji and Eisenstein (2013) |           |          |          |          | 0.804    | 0.859    | Matrix factorization with supervised reweighting
| He et al. (2015)         |           |          |          |          | 0.786    | 0.847    | Multi-perspective Convolutional NNs and structured similarity layer
|--------------------------|-----------|----------|----------|----------|----------|----------|---------
| avg                      | 0.726702  | 0.817746 | 0.734750 | 0.828196 | 0.707319 | 0.804215 | (defaults)
|                          |±0.005720  |±0.005384 |±0.003011 |±0.002984 |±0.002614 |±0.003876 |
|--------------------------|-----------|----------|----------|----------|----------|----------|---------
| rnn                      | 0.713035  | 0.808908 | 0.720250 | 0.819535 | 0.703007 | 0.801361 | (defaults)
|                          |±0.004878  |±0.004163 |±0.003282 |±0.003952 |±0.004594 |±0.005121 |
| cnn                      | 0.848819  | 0.903752 | 0.732875 | 0.829464 | 0.698188 | 0.801707 | (defaults)
|                          |±0.067693  |±0.039620 |±0.015090 |±0.005002 |±0.012668 |±0.004526 |
| attn1511                 | 0.856946  | 0.899338 | 0.764750 | 0.841120 | 0.726993 | 0.810355 | (defaults)
|                          |±0.030108  |±0.020247 |±0.007894 |±0.004343 |±0.005860 |±0.004620 |

rnncnn does not converge in our experiments.

THE RESULTS BELOW ARE OBSOLETE because they predate the f/bigvocab port.

| DAN                      | 0.751258  | 0.830429 | 0.719750 | 0.815998 | 0.703478 | 0.799095 | ``inp_e_dropout=0`` ``inp_w_dropout=1/3`` ``deep=2`` ``pact='relu'``
|                          |±0.021642  |±0.014033 |±0.005811 |±0.006513 |±0.003917 |±0.006118 |

These results are obtained like this:

	tools/train.py avg para data/para/msr/msr-para-train.tsv data/para/msr/msr-para-val.tsv nb_runs=16
	tools/eval.py avg para data/para/msr/msr-para-train.tsv data/para/msr/msr-para-train.tsv data/para/msr/msr-para-val.tsv data/para/msr/msr-para-test.tsv weights-para-avg--69489c8dc3b6ce11-*-bestval.h5
