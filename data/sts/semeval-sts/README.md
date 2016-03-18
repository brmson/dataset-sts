SemEval STS Task
================

Primarily based on the references in

http://alt.qcri.org/semeval2016/task1/index.php?id=data-and-tools

Per-year directories contain the datasets for these years, while
the ``all/`` contains a lot of symlinks.

Let's say you want to compare your model to 2016 entrants. To get
all the datasets before 2016 for your training set, load the
``all/201[0-5]*`` glob.

To evaluate using KeraSTS, refer to ``tools/sts_train.py``.

Otherwise: Use scipy.stats.pearson.  The evaluation code is also
the same as in ../sick2014 - refer e.g. to the python example from
skip-thoughts, or in our own examples/ directory.

Model Comparison
----------------

For randomized models, 95% confidence intervals (t-distribution) are reported.

Because SemEval 2016 competition results weren't published at the test time,
we train on -2014 and test on 2015.  We use 2014.tweet-news as a validation
set.

| Model                    | train    | val      | ans.for. | ans.stud | belief   | headline | images   | t. mean  | settings
|--------------------------|----------|----------|----------|----------|----------|----------|----------|----------|---------
| termfreq TF-IDF #w       | 0.497085 | 0.651653 | 0.607226 | 0.676746 | 0.622920 | 0.725578 | 0.714331 | 0.669360 | ``freq_mode='tf'``
| termfreq BM25 #w         | 0.503736 | 0.656081 | 0.626950 | 0.690302 | 0.632223 | 0.725748 | 0.718185 | 0.678681 | (defaults)
| DLS@CU-S1                |          |          | 0.7390   | 0.7725   | 0.7491   | 0.8250   | 0.8644   | 0.8015   | STS2015 winner
|--------------------------|----------|----------|----------|----------|----------|----------|----------|----------|---------


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
