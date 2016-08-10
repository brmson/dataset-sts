AI2 8th Grade Science Questions, Hypothesis Based Evaluation
============================================================

This dataset is meant to support solving science tests, using a reduced
(public version) dataset of the Aristo Kaggle challenge:

	http://aristo-public-data.s3.amazonaws.com/AI2-8thGr-NDMC-Feb2016.zip

Each question + answer is combined to a hypothesis sentence; "memory"
evidencing sentences are retrieved from either enwiki or a collection
of CK12 textbooks.  Extra token annotations are included, denoting matches
of named entities linked from the sentences.  This all is produced using
the Chios framework:

	https://github.com/brmson/aristo-chios

This is a very hard dataset, perhaps one of the ultimate tests for our models.
Moreover, while the lower bound of accuracy is 0.25, the upper bound is much
below 1.0 - a large chunk of hypotheses are probably unprovable with the
evidence we have (or simply unprovable in this task framework).

This is a v0; we expect better dataset versions to evolve over time,
with smoother hypothesis sentences and better evidence.  More public
questions were also promised.

| Model                    | trn Acc  | val Acc  | val MRR  | tst Acc  | tst MRR   | settings
|--------------------------|----------|----------|----------|----------|-----------|----------
| avg                      | 0.505379 | 0.442402 | 0.779080 | 0.400881 | 0.687528  | (defaults)
|                          |±0.024486 |±0.021844 |±0.018049 |±0.015593 |±0.013768  |
| DAN                      | 0.555840 | 0.491422 | 0.816840 | 0.390969 | 0.686856  | ``inp_e_dropout=0`` ``inp_w_dropout=1/3`` ``deep=2`` ``pact='relu'``
|                          |±0.038450 |±0.014991 |±0.013183 |±0.007895 |±0.006521  |
|--------------------------|----------|----------|----------|----------|-----------|----------
| rnn                      | 0.711967 | 0.381176 | 0.730000 | 0.360705 | 0.658262  | ``dropout=1/3`` ``inp_e_dropout=1/3``
|                          |±0.052589 |±0.015708 |±0.012071 |±0.012326 |±0.008562  |
| cnn                      | 0.675973 | 0.442402 | 0.780961 | 0.384086 | 0.686716  | ``dropout=1/3`` ``inp_e_dropout=1/3``
|                          |±0.056218 |±0.012234 |±0.012751 |±0.011142 |±0.009110  |
| rnncnn                   | 0.582480 | 0.438725 | 0.780527 | 0.375826 | 0.679659  | ``dropout=1/3`` ``inp_e_dropout=1/3``
|                          |±0.056659 |±0.024469 |±0.023307 |±0.014320 |±0.011600  |
| attn1511                 | 0.724898 | 0.383578 | 0.724826 | 0.357654 | 0.658294  | ``focus_act='sigmoid/maxnorm'`` ``cnnact='relu'``
|                          |±0.069447 |±0.011663 |±0.011245 |±0.015420 |±0.011558  |
| Ubu. RNN w/ MLP          | 0.569672 | 0.493873 | 0.827836 | 0.441355 | 0.728019  | ``vocabt='ubuntu'`` ``pdim=1`` ``ptscorer=B.mlp_ptscorer`` ``dropout=0`` ``inp_e_dropout=0`` ``task1_conf={'ptscorer':B.dot_ptscorer, 'f_add_kw':False}`` ``opt='rmsprop'``
|                          |±0.058833 |±0.012373 |±0.011304 |±0.010668 |±0.007045  |

Licencing
---------

Carissa Schoenick of the Allen Foundation stated:

>  The data set on allenai.org/data is assembled from public sources and is
> therefore open for redistribution. (To clarify - the data from this Kaggle
> competition is not redistributable, only the data we've posted on the AI2
> website.)
