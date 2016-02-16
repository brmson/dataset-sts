
			 SEMEVAL-2012 TASK 17

				 STS
		     Semantic Textual Similarity:
 
		     A Unified Framework for the
	      Evaluation of Modular Semantic Components



The test dataset contains the following:

  00-README.txt 		  this file
  pearson.pl			  evaluation script
  STS.input.MSRpar.txt		  tab separated input file with ids and sentence pairs
  STS.input.MSRvid.txt		   "
  STS.input.SMTeuroparl.txt	   "
  STS.gs.MSRpar.txt	  	  tab separated gold standard
  STS.gs.MSRvid.txt	  	   "
  STS.gs.SMTeuroparl.txt  	   "
  STS.output.MSRpar.txt	  	  tab separated sample output

    

Introduction
------------

Given two sentences of text, s1 and s2, the systems participating in
this task should compute how similar s1 and s2 are, returning a
similarity score, and an optional confidence score.

The dataset comprises pairs of sentences drawn from publicly
available datasets:

- MSR-Paraphrase, Microsoft Research Paraphrase Corpus
  http://research.microsoft.com/en-us/downloads/607d14d9-20cd-47e3-85bc-a2f65cd28042/
  750 pairs of sentences.

- MSR-Video, Microsoft Research Video Description Corpus
  http://research.microsoft.com/en-us/downloads/38cf15fd-b8df-477e-a4e4-a4680caa75af/
  750 pairs of sentences.

- SMTeuroparl: WMT2008 develoment dataset (Europarl section)
  http://www.statmt.org/wmt08/shared-evaluation-task.html
  734 pairs of sentences.

The sentence pairs have been manually tagged with a number from 0 to
5, as defined below (cf. Gold Standard section). 

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

The input file consist of three fields separated by tabs:

- unique id of pair
- first sentence (does not contain tabs)
- second sentence (does not contain tabs)

Please check any of STS.input.*.txt



Gold Standard
-------------

The gold standard contains a score between 0 and 5 for each pair of
sentences, with the following interpretation:

(5) The two sentences are completely equivalent, as they mean the same
    thing.  

      The bird is bathing in the sink.  
      Birdie is washing itself in the water basin.

(4) The two sentences are mostly equivalent, but some unimportant
    details differ.

      In May 2010, the troops attempted to invade Kabul.
      The US army invaded Kabul on May 7th last year, 2010.

(3) The two sentences are roughly equivalent, but some important
    information differs/missing.

      John said he is considered a witness but not a suspect.
      "He is not a suspect anymore." John said.

(2) The two sentences are not equivalent, but share some details.

      They flew out of the nest in groups.
      They flew into the nest together.

(1) The two sentences are not equivalent, but are on the same topic.

      The woman is playing the violin.
      The young lady enjoys listening to the guitar.

(0) The two sentences are on different topics.

      John went horse back riding at dawn with a whole group of friends.
      Sunrise at dawn is a magnificent view to take in if you wake up
      early enough for it.

Format: the gold standard file consist of one single field per line:

- a number between 0 and 5

The gold standard was assembled using mechanical turk, gathering 5
scores per sentence pair. The gold standard score is the average of
those 5 scores.

Please check any of STS.*.gs.txt



Answer format
--------------

The answer format is similar to the gold standard format, but includes
an optional confidence score. Each line has two fields separated by a
tab:

- a number between 0 and 5 (the similarity score)
- a number between 0 and 100 (the confidence score)

The use of confidence scores is experimental, and it is not required
for the official score.

Please check STS.MSRpar.output.txt which always returns 2.5 with
confidence 100.



Scoring
-------

The oficial score is based on Pearson correlation. The use of
confidence scores will be experimental, and it is not required for the
official scores. 

For instance:

  $ ./correlation.pl STS.gs.MSRpar.txt  STS.output.MSRpar.txt
  Pearson: 0.00000

Please check correlation.pl


Participation in the task
-------------------------

Participant teams will be allowed to submit three runs at most.

NOTE: Participant systems should NOT use the following datasets to
develop or train their systems:

- test part of MSR-Paraphrase (development and train are fine)
- the text of the videos in MSR-Video
- the data from the evaluation tasks at any WMT (all years are forbidden)



Other
-----

Please check http://www.cs.york.ac.uk/semeval/task17/ for more details.

We recommend that potential participants join the task mailing list:

 http://groups.google.com/group/STS-semeval



