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

  * **para.py** - Paraphrasing Task.  This task is like the STS task,
    but rather than regressing a numerical score on output, it is
    a binary classification task.  Example:

        tools/train.py cnn para data/para/msr/msr-para-train.tsv data/para/msr/msr-para-val.tsv nb_runs=4
	tools/eval.py cnn para data/para/msr/msr-para-train.tsv data/para/msr/msr-para-val.tsv - weights-para-cnn--69489c8dc3b6ce11-*
