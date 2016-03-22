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
