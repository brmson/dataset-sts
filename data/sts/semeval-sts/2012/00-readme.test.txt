
			 SEMEVAL-2012 TASK 17

				 STS
		     Semantic Textual Similarity:
 
		     A Unified Framework for the
	      Evaluation of Modular Semantic Components



The test gold dataset contains the following:

  00-README.txt 		  this file

  STS.input.MSRpar.txt		  tab separated input file with sentence pairs
  STS.input.MSRvid.txt		   "
  STS.input.SMTeuroparl.txt	   "
  STS.input.surprise.OnWN.txt      "
  STS.input.surprise.SMTnews.txt   "

  STS.gs.MSRpar.txt		  tab separated gold standard file
  STS.gs.MSRvid.txt		   "
  STS.gs.SMTeuroparl.txt	   "
  STS.gs.surprise.OnWN.txt         "
  STS.gs.surprise.SMTnews.txt      "

  STS.gs.ALL.txt                  tab separated gold standard file (concatenation of the five above)
 

Introduction
------------

Given two sentences of text, s1 and s2, the systems participating in
this task should compute how similar s1 and s2 are, returning a
similarity score, and an optional confidence score.

The dataset comprises pairs of sentences drawn from the publicly
available datasets used in training:

- MSR-Paraphrase, Microsoft Research Paraphrase Corpus
  http://research.microsoft.com/en-us/downloads/607d14d9-20cd-47e3-85bc-a2f65cd28042/
  750 pairs of sentences.

- MSR-Video, Microsoft Research Video Description Corpus
  http://research.microsoft.com/en-us/downloads/38cf15fd-b8df-477e-a4e4-a4680caa75af/
  750 pairs of sentences.

- SMTeuroparl: WMT2008 develoment dataset (Europarl section)
  http://www.statmt.org/wmt08/shared-evaluation-task.html
  459 pairs of sentences.

In addition, it contains two surprise datasets comprising the
following collections:

- SMTnews: news conversation sentence pairs from WMT
  399 pairs of sentences.

- OnWN: pairs of sentences where the first comes from Ontonotes and
  the second from a WordNet definition.
  750 pairs of sentences.


NOTE: Participant systems should NOT use the following datasets to
develop or train their systems:

- test part of MSR-Paraphrase (development and train are fine)
- the text of the videos in MSR-Video
- the data from the evaluation tasks at any WMT (all years are forbidden)



License
-------

All participants need to agree with the license terms from Microsoft Research:

http://research.microsoft.com/en-us/downloads/607d14d9-20cd-47e3-85bc-a2f65cd28042/
http://research.microsoft.com/en-us/downloads/38cf15fd-b8df-477e-a4e4-a4680caa75af/




Input format
------------

The input file consist of two fields separated by tabs:

- first sentence (does not contain tabs)
- second sentence (does not contain tabs)

Please check any of STS.input.*.txt


Gold standard format
--------------------

Format: the gold standard file consist of one single field per line:

- a number between 0 and 5


Other
-----

Please check http://www.cs.york.ac.uk/semeval/task17/ for more details.

We recommend that potential participants join the task mailing list:

 http://groups.google.com/group/STS-semeval



STS Organizers

Eneko Agirre
Daniel Cer
Mona Diab
Bill Dolan 


