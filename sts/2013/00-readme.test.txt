
			*SEM 2013 SHARED TASK

				 STS
		     Semantic Textual Similarity:
 
		     A Unified Framework for the
	      Evaluation of Modular Semantic Components

			     CORE DATASET
				   
		      TEST DATA WITH GOLD LABELS
				   

IMPORTANT NOTE: Due to licensing issues, after the task was finished
   we are not allowed to redistribute the SMT.input.txt file, and the
   file is distributed by LDC. Please check the website for further
   instructions.

This set of files describes the CORE DATASET for the main track of the
*SEM 2013 SHARED TASK on Semantic Textual Similarity.

The test dataset contains the following:

  00-README.txt 		  this file
  correlation.pl		  evaluation script for a single dataset
  correlation-all.pl		  evaluation script for all datasets 
  correct-output.pl               data integrity script 

  STS.input.headlines.txt         tab separated sample input file with 
                                  sentence pairs
  STS.input.OnWN.txt              "
  STS.input.FNWN.txt              "

  STS.input.SMT.txt               PLEASE NOTE: Due to licensing issues, after the task was finished
                                  we are not allowed to redistribute the SMT.input.txt file, and the
                                  file is distributed by LDC. Please check the website for further
                                  instructions.

  STS.gs.headlines.txt            tab separated sample gold standard
  STS.gs.OnWN.txt                 "
  STS.gs.FNWN.txt                 "
  STS.gs.SMT.txt                  "

  STS.output.headlines.txt	  tab separated sample output
  STS.output.OnWN.txt	  	  "
  STS.output.FNWN.txt	  	  "
  STS.output.SMT.txt	  	  "

 

Introduction
------------

Given two sentences of text, s1 and s2, the systems participating in
this task should compute how similar s1 and s2 are, returning a
similarity score, and an optional confidence score.

We include text data for the core test datasets, coming from the
following:

1) news headlines (headlines)
2) mapping of lexical resources (OnWN and FNWN)
3) evaluation of machine translation (SMT)

Note that the OnWN and FNWN test sets are smaller than the other two
datasets (headlines and SMT).

The datasets has been derived as follows:

- STS.input.headlines.txt: we used headlines mined from several news
  sources by European Media Monitor using the RSS feed. 
  http://emm.newsexplorer.eu/NewsExplorer/home/en/latest.html

- STS.input.OnWN.txt: The sentences are sense definitions from WordNet
  and OntoNotes. 

- STS.input.FNWN.txt: The sentences are sense definitions from WordNet
  and FrameNet. Note that some FrameNet definitions involve more than
  one sentence.

- STS.input.SMT.txt: This SMT dataset comes from DARPA GALE HTER and
  HyTER, where one sentence is a MT output and the other is a
  reference translation where a reference is generated based on human
  post editing (provided by LDC) or an original human reference
  (provided by LDC) or a human generated reference based on FSM as
  described in (Dreyer and Marcu, NAACL 2012). The reference comes
  from post edited translations.

The sentence pairs have been manually tagged with a number from 0 to
5, as defined below (cf. Gold Standard section). 



Input format
------------

The input file consist of two fields separated by tabs:

- first sentence (does not contain tabs)
- second sentence (does not contain tabs)

Please check any of STS.input.*.txt



Gold Standard
-------------

The gold standard (to be distributed later) contains a score between 0
and 5 for each pair of sentences, with the following interpretation:

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

The gold standard has been assembled using mechanical turk, gathering
several scores per sentence pair. The gold standard score will the
average of those scores. 

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

The output file needs to conform to the above specifications. Files
which do not follow those will be automatically removed from
evaluation. Please check that your answer files are in the correct
format using the following script:
 
  $ ./correct-output.pl STS.output.SMT.txt
  Output file is OK!

In addition to printing errors and a final message on standard error,
the script returns 0 if correct, and another value if incorrect. 



Scoring
-------

The oficial score is based on the average of Pearson correlation. The use of
confidence scores will be experimental, and it is not required for the
official scores. 

For instance the following script returns the correlation for
individual pairs in the trial dataset:

  $ ./correlation.pl STS.gs.SMT.txt  STS.output.SMT.txt
  Pearson: -0.30979

The following script returns the weighted average for all datasets in
the current directory, where the weights depend on the number of pairs
of each dataset. This is the output on the trial dataset:

  $ ./correlation-all.pl .
  STS.output.headlines.txt Pearson: -0.30979
  STS.output.OnWN.txt Pearson: -0.30979
  STS.output.FNWN.txt Pearson: -0.30979
  STS.output.SMT.txt Pearson: -0.30979
  Mean: -0.30979



Participation in the task
-------------------------

Participant teams will be allowed to submit three runs at most.

NOTE: Participant systems should NOT use the following datasets to
develop or train their systems: 

NOTE: Participant systems should NOT use the following datasets to
develop or train their systems: 

- Ontonotes - Wordnet sense aligned definitions.
- FrameNet - Wordnet sense aligned definitions.
- DARPA GALE HTER and HyTER datasets.



Other
-----

Please check http://ixa2.si.ehu.es/sts for more details.

We recommend that potential participants join the task mailing list:

 http://groups.google.com/group/STS-semeval



Authors
-------

Eneko Agirre
Daniel Cer
Mona Diab
Aitor Gonzalez-Agirre
Weiwei Guo
German Rigau


Acknowledgements
----------------

The WordNet-FrameNet mappings are funded by NSF 11-536 CRI planning
award for LexLink, Christiane Fellbaum, Collin Baker, Martha Palmer
and Orin Hargraves, and by NSF award CRI:CI-ADDO-EN 0855271 for
Christiane Fellbaum and Collin Baker. We are grateful for their
sharing of the data.

We are grateful to the OntoNotes team for sharing OntoNotes to WordNet
mappings (Hovy et al. 2006).

We thank DARPA and LDC for providing the SMT data.



References
----------

Eduard Hovy, Mitchell Marcus, Martha Palmer, Lance Ramshaw, and Ralph
Weischedel. 2006. Ontonotes: The 90% solution. In Proceedings of the
Human Language Technology Conference of the North American Chapter
of the ACL.

Markus Dreyer and Daniel Marcu. 2012. HyTER: Meaning-Equivalent
Semantics for Translation Evaluation. In Proceedings of the 2012
Conference of the North American Chapter of the Association for
Computational Linguistics: Human Language Technologies.


