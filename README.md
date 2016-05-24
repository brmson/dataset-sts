Semantic Text Similarity Dataset Hub
====================================

A typical NLP machine learning task involves classifying a sequence of tokens
such as a sentence or a document, i.e. approximating a function

	f_1(s) ∈ [0,1]

(where *f_1* may determine a domain, sentiment, etc.).  But there is a large
class of problems that are often harder and involve classifying a *pair* of
sentences:

	f_2(s1, s2) ∈ [0,1]*c

(where s1, s2 are sequences of tokens and c is a rescaling factor like c=5).

Typically, the function *f_2* denotes some sort of **semantic similarity**,
that is whether (or how much) the two parameters "say the same thing".
(However, the function could do something else - like classify entailment
or contradiction or just topic relatedness.  We may include such datasets
as well.)

This repo aims to gather a variety of standard datasets and tools for training
and evaluating such models in a single place, with the base belief that it
should be possible to build generic models for *f_2* that aren't tailored to
particular tasks (and even multitask learning should be possible).

Most of the datasets are pre-existing; text similarity datasets that may be
redistributed (at least for research purposes) are included.  Always check
the licence of a particular dataset.  Some datasets may be original though,
because we are working on many applied problems that pertain training such
a function...

The contents of dataset-sts and baseline results are described in the
paper [Sentence Pair Scoring: Towards Unified Framework for Text Comprehension](http://arxiv.org/abs/1603.06127).

**Pull requests welcome that extend the datasets, or add important comments,
references or attributions.  Please let us know if we misread some licence
terms and shouldn't be including something, we'll take that down right away!**

Pull request that include simple baselines for *f_2* models are also welcome.
(Simple == they fit in a couple of screenfuls of code and are batch-runnable.
Python is preferred, but not mandatory.)


Software Tools
--------------

To get started with simple classifiers that use task-specific code,
look at the **examples/** directory.
To get started with task-universal deep learning models, look at the
**tools/**, **models/** and **tasks/** directory.

  * **pysts/** Python module contains various tools for easy loading,
    manipulation and evaluation of the dataset.

  * **pysts/kerasts** the KeraSTS allows easy prototyping of deep learning
    models for many of the included tasks using the Keras library.

  * **examples/** contains a couple of simple, self-contained baselines
    on various tasks.

  * **models/** directory contains various strong baseline models using
    the KeraSTS toolkit, including state-of-art neural networks

  * **tasks/** directory contains model-independent interfaces to datasets
    for various tasks (from Answer Sentence Selection to Paraphrasing)

  * **tools/** directory contains tools that put models and tasks together;
    training, evaluating, tuning and transferring models on tasks

Datasets
--------

This is for now as much a TODO list as an overview.

### "Paraphrasing" Task

These datasets are about binary classification of independent sentence
(or multi-sentence) pairs regarding whether they say the same thing;
for example if they describe the same event (with same data), ask the
same question, etc.

  * [X] **data/para/msr/** MSR Paraphrase Dataset (TODO: pysts manipulation tools)

  * [X] **data/para/askubuntu/** [AskUbuntu StackOverflow Similar Questions](https://github.com/taolei87/rcnn)

  * [ ] [PPDB: The Paraphrase Database](http://www.cis.upenn.edu/~ccb/ppdb/)
    contains only short phrase snippets, but tens of millions of pairs

  * [ ] More [Stack Exchange](https://archive.org/details/stackexchange) data?
    (some is also contained in the new STS datasets)

### "Semantic Text Similarity" Task

These datasets consider the semantic similarity of independent pairs of texts
(typically short sentences) and share a precise similarity metric definition
of assigning a number between 0 to 5 to each pair denoting the level of
similarity/entailment.

  * [X] **data/sts/semeval-sts/** SemEval STS Task - multiple years, each covers a bunch of
    topics that share the same precise similarity metric definition

  * [X] **data/sts/sick2014/** SemEval SICK2014 Task

  * [ ] [SemEval 2014 Cross-level Semantic Similarity Task](http://alt.qcri.org/semeval2014/task3/index.php?id=data-and-tools)
    (TODO; 500 paragraph-to-sentence training items)

### "Entailment" Task

These datasets classify independent pairs of "hypothesis" and "fact"
sentences as entailment, contradiction or unknown.

  * [X] **data/rte/sick2014/** SemEval SICK2014 Task also includes entailment data

  * [X] **data/rte/snli/** [The Stanford Natural Language Inference (SNLI) Corpus](http://nlp.stanford.edu/projects/snli/)
(570k pairs dataset for an RTE-type task).

  * [ ] RTE Datasets up to RTE-3 http://nlp.stanford.edu/RTE3-pilot/ (TODO)

### "Answer Sentence Selection" Task

These datasets concern a "bipartite ranking" task.  That is, each tuple
of sentences is binary classified, but there are many different S1 sentences
given the same S0 and the ultimate goal is to sort positive-labelled S1s
above negative-labelled S1s for each S0.

Typically, S0 is a question and S1 are potentially-answer-bearing passages
(in that case, identifying the actual answer might be an auxiliary task
to consider; see anssel-yodaqa).  However, other scenarios are possible, like
the Ubuntu Dialogue Corpus where S1 are dialogue followups to S0.

  * [X] **data/anssel/wang/** Answer Sentence Selection - original Wang dataset

  * [X] **data/anssel/yodaqa/** Answer Sentence Selection - YodaQA-based

  * [ ] [InsuranceQA Dataset](https://github.com/shuzi/insuranceQA)
(used in recent IBM papers, 25k question-answer pairs; unclear licencing)

  * [X] **data/anssel/wqmprop/** Property Path Selection (based on WebQuestions + YodaQA)

  * [X] **data/anssel/ubuntu/** The Ubuntu Dialogue Corpus
contains pairs of sequences where the second sentence is a candidate for being
a followup in a community techsupport chat dialog.  10M pairs make this
awesome.

### "Hypothesis Evidencing" Task

Similar to the "Answer Sentence Selection" task, these datasets need to
consider a variety of S1 given a fixed S0 - the desired output should be
however a judgement about S0 alone (typically true / false).

  * [X] **data/hypev/argus/** Argus Dataset (Yes/No Question vs. News Headline)

  * [X] **data/hypev/ai2-8grade/** [AI2 8th Grade Science Questions](http://allenai.org/data.html)
are 641 school Science quiz questions (A/B/C/D test format), stemming from
[The Allen AI Science Challenge](https://www.kaggle.com/c/the-allen-ai-science-challenge/)
We are going to produce a dataset that merges questions and answers in a single
sentence, and pairs each with potential-evidencing sentences from Wikipedia and
CK12 textbooks.  This will be probably the hardest dataset by far included in
this repo for some time.  (We may also want to include the Elementary dataset.)

  * [ ] bAbI has a variety of datasets, especially re memory networks (memory
relevant to a given question), though with an extremely limited vocabulary.

  * [X] **data/hypev/mctest/** [Machine Comprehension Test (MCTest)](http://research.microsoft.com/en-us/um/redmond/projects/mctest/)
contains 300 children stories with many sentences and 4 questions each.
A share-alike type licence.

  * [ ] More "Entrance Exam" tasks solving multiple-choice school tests.


Other Datasets
--------------

Some datasets are not universally available, but we may accept contributions
regarding code to load them.

### Non-Redistributable Datasets

Some datasets cannot be redistributed.  Therefore, some scientists
may not be able to agree with the licence and download them, and/or may
decide not to use them for model development and research (if it is in
commercial setting), but only for some final benchmarks to benefit
cross-model comparisons.  We *discourage* using these datasets.

  * [ ] [Microsoft Research Video Description Corpus](http://research.microsoft.com/en-us/downloads/38cf15fd-b8df-477e-a4e4-a4680caa75af/)
(video annotation task, 120k sentences in 2k clusters)

  * [ ] [Microsoft Research WikiQA Corpus](http://research.microsoft.com/en-US/downloads/4495da01-db8c-4041-a7f6-7984a4f6a905/default.aspx)
(Answer Selection task, 3k questions and 29k answers with 1.5k correct)

  * [ ] [STS2013 Joint Student Response Analysis (RTE-8)](https://www.cs.york.ac.uk/semeval-2013/task7/index.php%3Fid=data.html)

### Non-free Datasets

Some datasets are completely non-free and not available on the internet,
therefore as strong believers in reproducible experiments and open science,
we *strongly discourage their usage*.

  * [ ] [STS2013 Machine Translation](https://catalog.ldc.upenn.edu/LDC2013T18)
pairs translated and translated-and-postedited newswire headlines.
Payment required.

  * [ ] [TAC tracks RTE-4 to RTE-7](http://www.nist.gov/tac/data/).
Printed user agreement required.


Algorithm References
--------------------

Here, we refer to some interesting models for sentence pair classification.
We focus mainly on papers that consider multiple datasets or are hard to find;
you can read e.g. about STS winners on the STS wiki, about anssel/wang models
on the ACL wiki, about RTE models on the SNLI page.

  * https://github.com/alvations/stasis contains several baselines and another
    view of the datasets (incl. the CLSS task)
  * https://github.com/ryankiros/skip-thoughts
  * Standard memory networks (MemNN, MemN2N) are in fact *f_2* models at their
    core; very similar to http://arxiv.org/abs/1412.1632

Licence and Attribution
-----------------------

Always check the licences of the respective datasets you are using!  Some of
them are plain CC-BY, others may be heavily restricted e.g. for non-commercial
use only.  Default licence for anything else in this repository is ASLv2 for
the code, CC-BY 4.0 for data.

Work on this project has been in part kindly sponsored by the Medialab
foundation (http://medialab.cz/), a Czech Technical University incubator.
The rest of contributions by Petr Baudiš is licenced as open source via
Ailao (http://ailao.eu/).  (Ailao also provides commercial consulting,
customization, deployment and support services.)
