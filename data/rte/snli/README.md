The Stanford Natural Language Inference (SNLI) Corpus
=====================================================


More on the dataset at http://nlp.stanford.edu/pubs/snli_paper.pdf.

Regarding software, use ``tools/snli_preprocess.py`` and ``tools/train.py``
(on ``snli`` task).

Model Comparison
----------------

For randomized models, 95% confidence intervals (t-distribution) will be reported.

Note that NO DROPOUT is applied for any of the models.

| Model                    | train    | test     | t. mean  | settings
|--------------------------|----------|----------|----------|----------
| Bowman et al. '16        | 0.892    |  0.832   | NA       | 300D SPINN-NP encoders (3.7m params)
| Cheng et al. '16         | 0.921    |  0.890   | NA       | 300D LSTMN with deep attention fusion (1.4m params), state-of-art
|--------------------------|----------|----------|----------|----------
| avg                      | 0.774939 | 0.726283 | NA       | (defaults)
| rnn                      | 0.818001 | 0.773208 | NA       | (defaults)


These results are obtained like this:

   ``tools/snli_preprocess.py --revocab data/rte/snli/snli_1.0_train.jsonl data/rte/snli/snli_1.0_dev.jsonl data/rte/snli/snli_1.0_test.jsonl data/rte/snli/snli_1.0_train.pickle data/rte/snli/snli_1.0_dev.pickle data/rte/snli/snli_1.0_test.pickle data/rte/snli/v1-vocab.pickle``


   ``tools/train.py avg snli  data/rte/snli/snli_1.0_train.pickle data/rte/snli/snli_1.0_dev.pickle vocabf="data/rte/snli/v1-vocab.pickle" testf="data/rte/snli/snli_1.0_test.pickle" inp_w_dropout=0 dropout=0 inp_e_dropout=0``

HOWTO
-----

Go to data/snli
    ``cd data/snli``

Download dataset
    ``wget http://nlp.stanford.edu/projects/snli/snli_1.0.zip``
    ``unzip snli_1.0.zip``

Go to datasets-sts/

Preprocess - create input files and vocabulary
   ``tools/snli_preprocess.py --revocab data/rte/snli/snli_1.0_train.jsonl data/rte/snli/snli_1.0_dev.jsonl data/rte/snli/snli_1.0_test.jsonl data/rte/snli/snli_1.0_train.pickle data/rte/snli/snli_1.0_dev.pickle data/rte/snli/snli_1.0_test.pickle data/rte/snli/v1-vocab.pickle``

Run selected model on the SNLI task - NO DROPOUT applied:
   ``tools/train.py avg snli  data/rte/snli/snli_1.0_train.pickle data/rte/snli/snli_1.0_dev.pickle vocabf="data/rte/snli/v1-vocab.pickle" testf="data/rte/snli/snli_1.0_test.pickle" inp_w_dropout=0 dropout=0 inp_e_dropout=0``
