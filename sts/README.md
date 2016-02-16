SemEval STS Task
================

Primarily based on the references in

http://alt.qcri.org/semeval2016/task1/index.php?id=data-and-tools

Per-year directories contain the datasets for these years, while
the ``all/`` contains a lot of symlinks.

Let's say you want to compare your model to 2015 entrants. To get
all the datasets before 2015 for your training set, load the
``all/201[0-4]*`` glob.

Note that 2016 is not symlinked to all/ yet since there is no gold
standard available so far.

To evaluate using KeraSTS, refer to ``tools/sts_train.py``.

Otherwise: Use scipy.stats.pearson.  The evaluation code is also
the same as in ../sick2014 - refer e.g. to the python example from
skip-thoughts, or in our own examples/ directory.

Changes
-------

The original distribution puts gold standard and sentence pairs
in separate files.  To make ingestion easier, we paste them together
in .tsv files.

Not Included
------------

These datasets were not included:

  * STS2012-MSRvid (licence restriction)
  * Sample outputs
  * Raw STS2015 (at least for now; TODO?)
  * STS2015 sample baseline and per-forum stackoverflow data (aggregate
    data is included!)
  * correlation.pl scripts (incompatible with our tsv files)
