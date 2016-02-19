The Ubuntu Dialogue Corpus
==========================

  * http://cs.mcgill.ca/~jpineau/datasets/ubuntu-corpus-1.0/

  * https://github.com/rkadlec/ubuntu-ranking-dataset-creator

This is not an "answer sentence selection" problem per se, but it is the same
kind of bipartite ranking task.

This corpus is obviously too big to be included as-is in this repository.
To get it, run these commands in this (data/anssel/ubuntu/) directory:

	for i in aa ab ac ad ae; do wget http://cs.mcgill.ca/~npow1/data/ubuntu_dataset.tgz.$i; done
	cat ubuntu_dataset.tgz.a* | tar xz
	mv ubuntu_csvfiles/trainset.csv v1-trainset.csv
	mv ubuntu_csvfiles/valset.csv v1-valset.csv
	mv ubuntu_csvfiles/testset.csv v1-testset.csv

TODO: v2 dataset instructions

Regarding software, use ``tools/ubuntu_preprocess.py`` and ``tools/ubuntu_train.py``.
See instructions on top of ``tools/ubuntu_train.py`` re preprocessing
the csv files (the dataset is too large to be fed directly to the
train tool).
