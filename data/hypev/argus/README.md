Argus YES/NO questions dataset
==============================

This dataset is meant to support answering yes/no questions based on newspaper
snippets:

	https://github.com/AugurProject/argus

The newspaper snippets are extraced by Solr fulltext search on a custom-built
news articles database based on question keywords.  Thus, for each such
snippet, both relevancy and logic relation must be jointly determined.  There
is no direct supervision for relevancy right now; each tuple is labelled by
just whether the question is true, regardless of the supporting snippet
entailing it.

Dataset contains triplets of:
  * questions: dataset of yes/no questions, partially from mTurk,
    partially auto-generated
  * answers: gold standard answer for each question
  * sentences: (hopefully) relevant sentences from our Argus news-articles
    database (The Guardian, NYtimes, and a couple of archive.org-fetched RSS
    streams from ABCnews, Reuters, etc)

Data split to train-test sets, same as the split used for Argus evaluation.

| Model                    | trnQAcc  | testQAcc | settings
|--------------------------|----------|----------|--------------
| avg                      | 0.870466 | 0.784722 | ``dropout=0`` ``nb_runs=4``
|                          |±0.060866 |±0.004942 |

These results are obtained like this:

	tools/train.py avg hypev data/hypev/argus/argus_train.csv data/hypev/argus/argus_test.csv dropout=0 nb_runs=16 nb_epoch=16
	tools/eval.py avg hypev data/hypev/argus/argus_train.csv data/hypev/argus/argus_test.csv - weights-hypev-avg-69b40732de2a1d70-0* dropout=0
