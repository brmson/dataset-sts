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
|--------------------------|-----------|----------|----------|----------|----------|----------|---------
| avg                      | 0.739094  | 0.827013 | 0.715750 | 0.816781 | 0.704710 | 0.803187 | (defaults)
|                          |±0.017685  |±0.008274 |±0.006054 |±0.004489 |±0.004266 |±0.005541 | 8-wise
| DAN                      | 0.776496  | 0.851614 | 0.717250 | 0.816756 | 0.704928 | 0.802403 | ``inp_e_dropout=0`` ``inp_w_dropout=1/3`` ``deep=2`` ``pact='relu'``
|                          |±0.059883  |±0.035841 |±0.007840 |±0.009356 |±0.005841 |±0.010487 | 8-wise
|--------------------------|-----------|----------|----------|----------|----------|----------|---------
| rnn                      | 0.716793  | 0.809159 | 0.688000 | 0.793592 | 0.687971 | 0.786216 | (defaults)
|                          |±0.038007  |±0.006681 |±0.010064 |±0.027672 |±0.030303 |±0.030582 | 4-wise
| cnn                      | 0.754334  | 0.832572 | 0.708500 | 0.806630 | 0.695797 | 0.792677 | (defaults)
|                          |±0.069931  |±0.036550 |±0.018334 |±0.020151 |±0.016102 |±0.020247 | 4-wise
| rnncnn                   | 0.721616  | 0.825273 | 0.708000 | 0.820639 | 0.702029 | 0.812272 | (defaults)
|                          |±0.008794  |±0.001507 |±0.000000 |±0.000000 |±0.005208 |±0.001433 | 2-wise
| attn1511                 | 0.738255  | 0.822510 | 0.707000 | 0.808802 | 0.708696 | 0.804536 | (defaults)
|                          |           |          |          |          |          |          | 2-wise
