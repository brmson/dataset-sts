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

This repo aims to gather a variety of standard datasets and tools for training
and evaluating such models in a single place, with the base belief that it
should be possible to build generic models for *f_b* that aren't tailored to
particular tasks (and even multitask learning should be possible).

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

This is for now as much a TODO list as an overview.

### Software tools

To get started with simple classifiers that use task-specific code,
look at the **examples/** directory.
To get started with task-universal deep learning models, look at the
**tools/** and **models/** directory.

  * **pysts/** Python module contains various tools for easy loading,
    manipulation and evaluation of the dataset.

  * **pysts/kerasts** the KeraSTS allows easy prototyping of deep learning
    models for many of the included tasks using the Keras library.

  * **examples/** contains a couple of simple, self-contained baselines
    on various tasks.

  * **tools/** directory contains tools to run, tune and evaluate the
    KeraSTS models from the models/ directory

  * **models/** directory contains various strong baseline models using
    the KeraSTS toolkit, including state-of-art neural networks

### Full Datasets

These datasets are small enough, require significant postprocessing
to be easily usable, and/or are original.  Therefore, they are included
in full in this Git repository:

  * [X] **sts/** SemEval STS Task - multiple years, each covers a bunch of
    topics that share the same precise similarity metric definition

  * [X] **sick2014/** SemEval SICK2014 Task

  * [ ] [SemEval 2014 Cross-level Semantic Similarity Task](http://alt.qcri.org/semeval2014/task3/index.php?id=data-and-tools)
    (TODO; 500 paragraph-to-sentence training items)

  * [X] **msr/** MSR Paraphrase Dataset (TODO: pysts manipulation tools)

  * [ ] RTE Datasets up to RTE-3 http://nlp.stanford.edu/RTE3-pilot/ (TODO)

  * [X] **anssel-wang/** Answer Sentence Selection - original Wang dataset

  * [X] **anssel-yodaqa/** Answer Sentence Selection - YodaQA-based

  * [ ] Property Path Selection (based on WebQuestions + YodaQA; TODO)

  * [X] **argus/** Argus Dataset (Yes/No Question vs. News Headline)

TODO: We should explain also the nature of the datasets - size, their exact
*f_b* definition, or whether contradictions occur.

### Other Free Datasets

Other datasets (will) have their own directory and Python loading / evaluation
tools, but you need to get the data separately for size reasons.  Of course,
we highly encourage using these datasets if you can deal with their size.

  * [ ] [The Ubuntu Dialogue Corpus](http://cs.mcgill.ca/~jpineau/datasets/ubuntu-corpus-1.0/)
contains pairs of sequences where the second sentence is a candidate for being
a followup in a community techsupport chat dialog.  10M pairs make this
awesome.

  * [ ] [The Stanford Natural Language Inference (SNLI) Corpus](http://nlp.stanford.edu/projects/snli/)
(570k pairs dataset for an RTE-type task).

  * [ ] bAbI has a variety of datasets, especially re memory networks (memory
relevant to a given question), though with an extremely limited vocabulary.

### Other Restricted Datasets

Some datasets are (will be) also included, but have restrictions regarding
commercial usage or redistribution conditions.  Therefore, some scientists
may not be able to agree with the licence and download them, and/or may
decide not to use them for model development and research (if it is in
commercial setting), but only for some final benchmarks to benefit
cross-model comparisons.  We *discourage* using these datasets.

  * [ ] [Microsoft Research Video Description Corpus](http://research.microsoft.com/en-us/downloads/38cf15fd-b8df-477e-a4e4-a4680caa75af/)
(video annotation task, 120k sentences in 2k clusters)

  * [ ] [Microsoft Research WikiQA Corpus](http://research.microsoft.com/en-US/downloads/4495da01-db8c-4041-a7f6-7984a4f6a905/default.aspx)
(Answer Selection task, 3k questions and 29k answers with 1.5k correct)

  * [ ] [STS2013 Joint Student Response Analysis (RTE-8)](https://www.cs.york.ac.uk/semeval-2013/task7/index.php%3Fid=data.html)

  * [ ] [InsuranceQA Dataset](https://github.com/shuzi/insuranceQA)
(used in recent IBM papers, 25k question-answer pairs)

Even More Datasets
------------------

### Non-free Datasets

Some datasets are completely non-free and not available on the internet
and we *strongly discourage their usage*.
We are not enthusiastic about these as we strongly believe in reproducible
experiments.  Nevertheless, we accept contributions regarding Python tools
to load and process these datasets.

  * [ ] [STS2013 Machine Translation](https://catalog.ldc.upenn.edu/LDC2013T18)
pairs translated and translated-and-postedited newswire headlines.
Payment required.

  * [ ] [TAC tracks RTE-4 to RTE-7](http://www.nist.gov/tac/data/).
Printed user agreement required.

  * [ ] [The Allen AI Science Challenge](https://www.kaggle.com/c/the-allen-ai-science-challenge/)
derived dataset combines question+answer in a single sentence and pairs it with relevant
sentences extracted from textbooks and Wikipedia.  The resulting dataset makes
it possible for humans to answer many (not nearly all!) of the questions, but
is very hard for machine models as it often requires significant world modelling
and reasoning.  This dataset had been Kaggle-only, but a more limited few
hundreds questions dataset has been promised to be released publicly soon.

### Stack Exchange

https://archive.org/details/stackexchange could yield question paraphrases
after some datacrunching; post-processed dataset submissions welcome!

(dos Santos, 2015) http://www.aclweb.org/anthology/P15-2114 used two
communities (Ask Ubuntu and English) to benchmark some similarity models.


Algorithm References
--------------------

Here, we refer to some interesting models for sentence pair classification.
We focus mainly on papers that consider multiple datasets or are hard to find;
you can read e.g. about STS winners on the STS wiki or about anssel-wang models
on the ACL wiki.

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
