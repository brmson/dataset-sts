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

Licencing
---------

Carissa Schoenick of the Allen Foundation stated:

>  The data set on allenai.org/data is assembled from public sources and is
> therefore open for redistribution. (To clarify - the data from this Kaggle
> competition is not redistributable, only the data we've posted on the AI2
> website.)
