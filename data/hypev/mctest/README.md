MCTest Machine Comprehension Dataset
====================================

This dataset comes from

	http://research.microsoft.com/en-us/um/redmond/projects/mctest/

and contains children stories with many sentences and 4 questions each.

The standard dataset contains of two subsets that differ by their precise
origin, but seem similar in character - mc160 and mc500.  We also just
concatenated them together to create a cummulative dataset mc660, to get
more data to train the models as the samples are pretty tiny otherwise.

Rather than Q/A pairs, we take advantage of the "Statements" dataset
provided along with the original dataset, which just contains statements.
These are our "htext" while "mtext" are story sentences.

The state-of-art is probably 1602.04341 Yin et al.'s HABCNN-TE
(http://arxiv.org/pdf/1602.04341v1.pdf).

## Evaluation

To compare results here, we train on MC660 (160+500 merged), all question types
(both one and multi), but for simplicity report performance just on **MC500-one**:

  * MC500 test set is obviously bigger than MC160.
  * The hypev task was never meant to account for surrounding evidence, i.e.
    composing multiple pieces of evidence together, so the interesting bit is
    how well it can answer "one"-type questions.

| Model                    | trn Acc  | val Acc  | val MRR  | tst Acc  | tst MRR   | settings
|--------------------------|----------|----------|----------|----------|-----------|----------
| avg                      | 0.632732 | 0.611192 | 0.768290 | 0.587086 | 0.748851  |
|                          |±0.012922 |±0.017245 |±0.009873 |±0.018124 |±0.010366  |
| DAN                      | 0.650221 | 0.603924 | 0.761446 | 0.636259 | 0.776597  | ``inp_e_dropout=0`` ``inp_w_dropout=1/3`` ``deep=2`` ``pact='relu'``
|                          |±0.014159 |±0.016462 |±0.008623 |±0.012985 |±0.007635  |
|--------------------------|----------|----------|----------|----------|-----------|----------
| rnn                      | 0.643409 | 0.545058 | 0.722626 | 0.539062 | 0.716912  | ``inp_e_dropout=1/3`` ``dropout=1/3``
|                          |±0.029702 |±0.021225 |±0.011171 |±0.016418 |±0.010984  |
| cnn                      | 0.706646 | 0.587936 | 0.752483 | 0.570542 | 0.738990  | ``inp_e_dropout=1/3`` ``dropout=1/3``
|                          |±0.023560 |±0.021122 |±0.013158 |±0.012510 |±0.006880  |
| rnncnn                   | 0.643778 | 0.575581 | 0.742127 | 0.553768 | 0.725107  | ``inp_e_dropout=1/3`` ``dropout=1/3``
|                          |±0.047369 |±0.031873 |±0.020092 |±0.023455 |±0.015103  |
| attn1511                 | 0.724411 | 0.523256 | 0.710938 | 0.571232 | 0.737956  | ``focus_act='sigmoid/maxnorm'`` ``cnnact='relu'``
|                          |±0.061083 |±0.035323 |±0.021406 |±0.035533 |±0.023051  |
| Ubu. RNN w/ MLP          | 0.749239 | 0.617054 | 0.768928 | 0.641422 | 0.783211  | ``vocabt='ubuntu'`` ``pdim=1`` ``ptscorer=B.mlp_ptscorer`` ``dropout=0`` ``inp_e_dropout=0`` ``task1_conf={'ptscorer':B.dot_ptscorer, 'f_add_kw':False}`` ``opt='rmsprop'``
|                          |±0.024649 |±0.024375 |±0.016828 |±0.016967 |±0.010961  |

The model was trained and evaluated like:

	python -u tools/train.py avg hypev data/hypev/mctest/mc660.train data/hypev/mctest/mc660.dev nb_runs=16
	python -u tools/transfer.py rnn ubuntu data/anssel/ubuntu/v2-vocab.pickle weights-ubuntu-rnn-37d0427cb1ad18c4-00-bestval.h5 hypev data/hypev/mctest/mc660.train data/hypev/mctest/mc660.dev pdim=1 ptscorer=B.mlp_ptscorer dropout=0 inp_e_dropout=0 "task1_conf={'ptscorer':B.dot_ptscorer, 'f_add_kw':False}" "opt='rmsprop'"  # alternatively
	python -u tools/eval.py avg hypev data/hypev/mctest/mc660.train data/hypev/mctest/mc500.dev data/hypev/mctest/mc500.test weights-hypev-cnn--1ec5e645570d9eeb-*-bestval.h5 "mcqtypes=['one']"
