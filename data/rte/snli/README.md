The Stanford Natural Language Inference (SNLI) Corpus
=====================================================

http://nlp.stanford.edu/projects/snli/

More on the dataset at http://nlp.stanford.edu/pubs/snli_paper.pdf.

Regarding software, use ``tools/snli_preprocess.py`` and ``tools/train.py``
(on ``snli`` task).

Model Comparison
----------------

For randomized models, 95% confidence intervals (t-distribution) are be reported.
Each model is run 4×.

Note that NO DROPOUT is applied for any of the models.

| Model                    | train    | dev      | test  | settings
|--------------------------|----------|----------|----------|----------
| Bowman et al.(2015) LSTM | 0.848    |          | 0.776    | The proposed 100d LSTM model with three tanh layers processing sentence embedding concatenation
| Bowman et al. '16        | 0.892    |   NA     | 0.832    | 300D SPINN-NP encoders (3.7m params)
| Cheng et al. '16         | 0.921    |   NA     | 0.890    | 300D LSTMN with deep attention fusion (1.4m params), state-of-art
|--------------------------|----------|----------|----------|----------
| avg                      | 0.735293 | 0.706106 | 0.710378 | ``dropout=0`` ``inp_e_dropout=0``
|                          |±0.014398 |±0.004129 |±0.007915 |
| DAN                      | 0.718436 | 0.705167 | 0.707629 | ``inp_e_dropout=0`` ``inp_w_dropout=1/3`` ``deep=2`` ``pact='relu'``
|                          |±0.008969 |±0.006060 |±0.001547 |
| avg Bowman               | 0.766992 | 0.727494 | 0.728166 | Approximating Bowmat et al. (2015) architecture (``dropout=0`` ``inp_e_dropout=0`` ``Ddim=[1,1]`` ``wproject=True`` ``wdim=1/2``)
|                          |±0.010214 |±0.005646 |±0.002769 |
| rnn                      | 0.784382 | 0.754344 | 0.748651 | ``dropout=0`` ``inp_e_dropout=0``
|                          |±0.019295 |±0.006224 |±0.009505 |
| rnncnn                   | 0.810825 | 0.761075 | 0.752748 | ``dropout=0`` ``inp_e_dropout=0``
|                          |±0.037068 |±0.012773 |±0.008269 |
| attn1511                 | 0.828611 | 0.781472 | 0.773921 | ``dropout=0`` ``inp_e_dropout=0``
|                          |±0.013870 |±0.002253 |±0.004134 |

These results are obtained like this:

	tools/train.py avg snli data/rte/snli/snli_1.0_train.pickle data/rte/snli/snli_1.0_dev.pickle "vocabf='data/rte/snli/v1-vocab.pickle'" nb_runs=4 inp_w_dropout=0 dropout=0 inp_e_dropout=0
	tools/eval.py avg snli data/rte/snli/snli_1.0_train.pickle data/rte/snli/snli_1.0_dev.pickle data/rte/snli/snli_1.0_test.pickle weights-snli-avg--69489c8dc3b6ce11-*-bestval.h5 "vocabf='data/rte/snli/v1-vocab.pickle'" nb_runs=4 inp_w_dropout=0 dropout=0 inp_e_dropout=0

HOWTO
-----

	cd data/rte/snli
	wget http://nlp.stanford.edu/projects/snli/snli_1.0.zip
	unzip snli_1.0.zip
	cd ../../..
	tools/snli_preprocess.py --revocab data/rte/snli/snli_1.0/snli_1.0_train.jsonl data/rte/snli/snli_1.0/snli_1.0_dev.jsonl data/rte/snli/snli_1.0/snli_1.0_test.jsonl data/rte/snli/snli_1.0_train.pickle data/rte/snli/snli_1.0_dev.pickle data/rte/snli/snli_1.0_test.pickle data/rte/snli/v1-vocab.pickle
	tools/train.py avg snli data/rte/snli/snli_1.0_train.pickle data/rte/snli/snli_1.0_dev.pickle "vocabf='data/rte/snli/v1-vocab.pickle'" inp_w_dropout=0 dropout=0 inp_e_dropout=0

NO DROPOUT applied (like for Ubuntu Dialogue).
