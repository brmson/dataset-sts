KeraSTS Tasks
=============

You can find Python interfaces for various tasks in this directory.
The tasks are model-invariant - given a task, you can train a TF-IDF
model as well as an attention-based deep neural network on it.
Pick a task, pick a model, and put them together by using one of
the scripts in the **tools/** directory.

In a sense, task is a class of datasets.  Find the datasets we include
(either as sources or as scripts that'll post-process external sources)
in **data/TASKNAME/** subdirectories.

  * **anssel.py** - Answer Sentence Selection Task.  Example:

	tools/train.py avg anssel data/anssel/yodaqa/curatedv2-training.csv data/anssel/yodaqa/curatedv2-val.csv nb_runs=4
	tools/eval.py avg anssel data/anssel/yodaqa/curatedv2-training.csv data/anssel/yodaqa/curatedv2-val.csv - weights-anssel-avg--731b5fca12808be0-*

  * **ubuntu.py** - Ubuntu Dialogue instance of anssel/next utterance task.
    The data is serialized efficiently and custom metrics reported.  Example:

        tools/train.py avg ubuntu data/anssel/ubuntu/v2-trainset.pickle data/anssel/ubuntu/v2-valset.pickle "vocabf='data/anssel/ubuntu/v2-vocab.pickle'" nb_runs=4
        tools/eval.py avg ubuntu data/anssel/ubuntu/v2-trainset.pickle data/anssel/ubuntu/v2-valset.pickle - weights-ubuntu-avg--731b5fca12808be0-* "vocabf='data/anssel/ubuntu/v2-vocab.pickle'"

  To set up the dataset from source files, refer to the data/ README and
  intro at the top of tasks/ubuntu.py.

  * **sts.py** - Semantic Text Similarity Task.
  The main differences to the anssel task are that (i) this task is symmetric
  and both sentences should be considered from the same viewpoint; (ii) the
  output is a number between 0 and 5 and Pearson correlation is the metric.
  Example:

	tools/train.py avg sts data/sts/semeval-sts/all/2015.train.tsv data/sts/semeval-sts/all/2015.val.tsv nb_runs=4
	tools/eval.py avg sts data/sts/semeval-sts/all/2015.train.tsv data/sts/semeval-sts/all/2015.val.tsv - weights-sts-avg--731b5fca12808be0-*

  * **para.py** - Paraphrasing Task.  This task is like the STS task,
    but rather than regressing a numerical score on output, it is
    a binary classification task.  Example:

        tools/train.py cnn para data/para/msr/msr-para-train.tsv data/para/msr/msr-para-val.tsv nb_runs=4
	tools/eval.py cnn para data/para/msr/msr-para-train.tsv data/para/msr/msr-para-val.tsv - weights-para-cnn--69489c8dc3b6ce11-*

  * **hypev.py** - Hypothesis Evidencing.  This task is like "Answer Sentence
    Selection" in that we have many s1 for a single s0, but the goal is
    to produce an aggregate judgement on s0 based on the pairs.  Example:

	tools/train.py avg hypev data/hypev/argus/argus_train.csv data/hypev/argus/argus_test.csv dropout=0 nb_runs=4 nb_epoch=16
