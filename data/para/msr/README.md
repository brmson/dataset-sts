Microsoft Research Paraphrase Corpus
====================================

See README.html for the original documentation.

Compared to the original documentation, the train and test sets are swapped,
just as in all the recent papers.

System Comparison:
http://aclweb.org/aclwiki/index.php?title=Paraphrase_Identification_(State_of_the_art)

Model Comparison
----------------

For randomized models, 95% confidence intervals (t-distribution) are reported.

| Model                    | train    | trial    | test     | t. mean  | settings
|--------------------------|----------|----------|----------|----------|---------

| termfreq TF-IDF #w       | 0.479906 | 0.456354 | 0.478802 | 0.478802 | ``freq_mode='tf'``
| termfreq BM25 #w         | 0.476338 | 0.458441 | 0.474453 | 0.474453 | (defaults)
| ECNU run1                |          |          | 0.8414   |          | STS2014 winner
| Kiros et al. (2015)      |          |          | 0.8655   |          | skip-thoughts
| Tai et al. (2015)        |          |          | 0.8676   |          | TreeLSTM; state-of-art
|--------------------------|----------|----------|----------|----------|---------
| avg                      | 0.758096 | 0.707246 |          |          | (defaults); once
| rnn d0                   | 0.709028 | 0.697971 |          |          | dropout=0 inp_e_dropout=0; once
| attn1511                 | 0.709028 | 0.697971 |          |          | (defaults); once
