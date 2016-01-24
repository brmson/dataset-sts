Semantic Text Similarity Dataset Hub
====================================

A typical NLP machine learning task involves classifying a sequence of tokens
such as a sentence or a document, i.e. approximating a function

	f_a(s) ∈ [0,1]

(where *f_a* may determine a domain, sentiment, etc.).  But there is a large
class of problems that are often harder and involve classifying a *pair* of
sentences:

	f_b(s1, s2) ∈ [0,1]*c

(where s1, s2 are sequences of tokens and c is a rescaling factor like c=5).

Typically, the function *f_b* denotes some sort of **semantic similarity**,
that is whether (or how much) the two parameters "say the same thing".
(However, the function could do something else - like classify entailment
or contradiction or just topic relatedness.  We may include such datasets
as well.)

This repo aims to gather a variety of standard datasets for training and
evaluating such models in a single place, with the base belief that it should
be possible to build generic models for *f_b* that aren't tailored to particular
tasks (and even multitask learning should be possible).

Most of the datasets are pre-existing; text similarity datasets that may be
redistributed (at least for research purposes) are included.  Always check
the licence of a particular dataset.  Some datasets may be original though,
because we are working on many applied problems that pertain training such
a function...

**Pull requests welcome that extend the datasets, or add important comments,
references or attributions.  Please let us know if we misread some licence
terms and shouldn't be including something, we'll take that down right away!**

Pull request that include simple baselines for *f_b* models are also welcome.
(Simple == they fit in a couple of screenfuls of code and are batch-runnable.
Python is preferred, but not mandatory.)

Package Overview
----------------

### Datasets

  * [X] **sts/** SemEval STS Task - multiple years, each covers a bunch of
    topics that share the same precise similarity metric definition

  * [X] **sick2014/** SemEval SICK2014 Task

  * [ ] SemEval 2014 Cross-level Semantic Similarity Task

  * [ ] MSR Paraphrase Dataset (TODO)

  * [ ] RTE Datasets (TODO)

  * [X] **anssel-wang/** Answer Sentence Selection - original Wang dataset

  * [X] **anssel-yodaqa/** Answer Sentence Selection - YodaQA-based

  * [ ] Property Selection (based on WebQuestions + YodaQA; TODO)

  * [ ] Argus Dataset (Yes/No Question vs. News Headline; work in progress)

  * [ ] COCO image-sentence ranking experiments

So, this is for now as much a TODO list as an overview.

TODO: We should explain also the nature of the datasets - size, their exact
*f_b* definition, or whether contradictions occur.

### Software tools

Python:

  * **pysts/** Python module contains various tools for easily working with the dataset
  * **example_yu1412_sts.py** shows a simple embedding-based classifier hacked
    together on a few lines that wouldn't end up dead last in STS2015 competition.
  * **example_yu1412_anssel.py** shows the yu1412 classifier applied to its original
    "answer sentence selection" task

Other Datasets
--------------

Some datasets could not have been included for legal or size reasons, but you
might find them inspiring:

  * https://www.kaggle.com/c/the-allen-ai-science-challenge/ derived dataset
that combines question+answer in a single sentence and pairs it with relevant
sentences extracted from textbooks and Wikipedia.

  * https://archive.org/details/stackexchange could yield question paraphrases
after some datacrunching; post-processed dataset submissions welcome!

  * https://catalog.ldc.upenn.edu/LDC2013T18 payment required

  * http://research.microsoft.com/en-us/downloads/38cf15fd-b8df-477e-a4e4-a4680caa75af/
Microsoft Research Video Description Corpus

  * http://research.microsoft.com/en-US/downloads/4495da01-db8c-4041-a7f6-7984a4f6a905/default.aspx
Microsoft Research WikiQA Corpus


Algorithm References
--------------------

Here, we refer to some interesting models for sentence pair classification.
We focus mainly on papers that consider multiple datasets or are hard to find;
you can read e.g. about STS winners on STS wiki.

  * https://github.com/alvations/stasis contains several baselines and another
    view of the datasets (incl. the CLSS task)
  * https://github.com/ryankiros/skip-thoughts
  * Standard memory networks (MemNN, MemN2N) are in fact *f_b* models at their
    core; very similar to http://arxiv.org/abs/1412.1632

Licence and Attribution
-----------------------

Always check the licences of the respective datasets you are using!  Some of
them are plain CC-BY, others may be heavily restricted e.g. for non-commercial
use only.  Default licence for anything else in this repository is ASLv2 for
the code, CC-BY 4.0 for data.

There should be a paper on this conglomerate of datasets (and comparison of
*f_b* metrics) to cite soon!  (As of Jan 2016.)  Watch this space for the
reference when it's done.
