The Stanford Natural Language Inference (SNLI) Corpus
=====================================================

http://nlp.stanford.edu/projects/snli/

More on the dataset at http://nlp.stanford.edu/pubs/snli_paper.pdf.

Regarding software, use ``tools/snli_preprocess.py`` and ``tools/train.py``
(on ``snli`` task).

Model Comparison
----------------

For randomized models, 95% confidence intervals (t-distribution) are be reported.

Note that NO DROPOUT is applied for any of the models.

| Model                    | train    | dev      | test  | settings
|--------------------------|----------|----------|----------|----------
| Bowman et al. '16        | 0.892    |   NA     | 0.832    | 300D SPINN-NP encoders (3.7m params)
| Cheng et al. '16         | 0.921    |   NA     | 0.890    | 300D LSTMN with deep attention fusion (1.4m params), state-of-art
|--------------------------|----------|----------|----------|----------
| avg                      | 0.741874 | 0.708824 | 0.712490 | ``inp_w_dropout=0`` ``dropout=0`` ``inp_e_dropout=0``
|                          |±0.009237 |±0.003442 |±0.005930 |
| rnn                      | 0.770763 | 0.753759 | 0.747303 | ``inp_w_dropout=0`` ``dropout=0`` ``inp_e_dropout=0``
|                          |±0.025463 |±0.009667 |±0.007605 |


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
