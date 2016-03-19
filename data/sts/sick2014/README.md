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
| avg                      | 0.713548 | 0.624316 | 0.621074 | 0.621074 | (defaults)
|                          |±0.044891 |±0.019678 |±0.022723 |          |
| DAN                      | 0.748372 | 0.654445 | 0.649951 | 0.649951 | ``inp_e_dropout=0`` ``inp_w_dropout=1/3`` ``deep=2`` ``pact='relu'``
|                          |±0.028542 |±0.017949 |±0.020164 |          |
|--------------------------|----------|----------|----------|----------|---------
| rnn                      | 0.739980 | 0.687428 | 0.686777 | 0.686777 | (defaults)
| cnn                      | 0.889760 | 0.770156 | 0.750282 | 0.750282 | (defaults)
