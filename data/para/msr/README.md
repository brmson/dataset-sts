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
| avg                      | 0.752237  | 0.835484 | 0.714875 | 0.817603 | 0.701775 | 0.803806 | (defaults)
|                          |±0.016065  |±0.008362 |±0.006655 |±0.006592 |±0.003495 |±0.005670 |
| DAN                      | 0.751258  | 0.830429 | 0.719750 | 0.815998 | 0.703478 | 0.799095 | ``inp_e_dropout=0`` ``inp_w_dropout=1/3`` ``deep=2`` ``pact='relu'``
|                          |±0.021642  |±0.014033 |±0.005811 |±0.006513 |±0.003917 |±0.006118 |
|--------------------------|-----------|----------|----------|----------|----------|----------|---------
| rnn                      | 0.717806  | 0.816834 | 0.705750 | 0.812771 | 0.691920 | 0.797981 | (defaults)
|                          |±0.016668  |±0.006131 |±0.004277 |±0.006366 |±0.007403 |±0.007051 |
| cnn                      | 0.746487  | 0.827205 | 0.704125 | 0.805049 | 0.702029 | 0.798059 | (defaults)
|                          |±0.015070  |±0.009841 |±0.006288 |±0.006411 |±0.003744 |±0.005246 |
| rnncnn                   | 0.782316  | 0.854726 | 0.712750 | 0.811653 | 0.704167 | 0.799246 | (defaults)
|                          |±0.028483  |±0.015017 |±0.006946 |±0.008944 |±0.006232 |±0.009749 |
| attn1511                 | 0.741401  | 0.821830 | 0.702250 | 0.801453 | 0.699891 | 0.791798 | (defaults)
|                          |±0.012435  |±0.005271 |±0.004882 |±0.007168 |±0.004946 |±0.008456 |
