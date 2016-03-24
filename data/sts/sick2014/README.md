http://alt.qcri.org/semeval2014/task1/

This dataset is different from the STS dataset in that it aims to capture
only similarities on purely language and common knowledgde level, without
relying on domain knowledge.  There are no named entities or multi-word
idioms, and the models should know that "a couple is formed by a bride
and a groom" but not that "the current president of the US is Barack Obama".

To evaluate using KeraSTS, refer to ``tools/sts_train.py``.
Otherwise, for a Python code example for evaluation and processing, see
https://github.com/ryankiros/skip-thoughts/blob/master/eval_sick.py

Model Comparison
----------------

For randomized models, 95% confidence intervals (t-distribution) are reported.
t. mean is the same as test (artifact of sts-semeval oriented evaluation).

| Model                    | train    | trial    | test     | t. mean  | settings
|--------------------------|----------|----------|----------|----------|---------
| termfreq TF-IDF #w       | 0.479906 | 0.456354 | 0.478802 | 0.478802 | ``freq_mode='tf'``
| termfreq BM25 #w         | 0.476338 | 0.458441 | 0.474453 | 0.474453 | (defaults)
| ECNU run1                |          |          | 0.8414   |          | STS2014 winner
| Kiros et al. (2015)      |          |          | 0.8655   |          | skip-thoughts
| Tai et al. (2015)        |          |          | 0.8676   |          | TreeLSTM; state-of-art
|--------------------------|----------|----------|----------|----------|---------
| avg                      | 0.722639 | 0.625970 | 0.621415 | 0.621415 | (defaults)
|                          |±0.035858 |±0.015800 |±0.017411 |
| DAN                      | 0.730100 | 0.648327 | 0.641699 | 0.641699 | ``inp_e_dropout=0`` ``inp_w_dropout=1/3`` ``deep=2`` ``pact='relu'``
|                          |±0.023870 |±0.012275 |±0.015708 |
|--------------------------|----------|----------|----------|----------|---------
| rnn                      | 0.732334 | 0.684798 | 0.663615 | 0.663615 | (defaults)
|                          |±0.035202 |±0.016028 |±0.022356 |
| cnn                      | 0.893748 | 0.763518 | 0.757617 | 0.757617 | (defaults)
|                          |±0.011819 |±0.005309 |±0.005817 |
| rnncnn                   | 0.940838 | 0.784702 | 0.790240 | 0.790240 | (defaults)
|                          |±0.012955 |±0.002831 |±0.004763 |
| attn1511                 | 0.835484 | 0.732324 | 0.722736 | 0.722736 | (defaults)
|                          |±0.023749 |±0.011295 |±0.009434 |
