
			       STS 2014

				 STS

	       Semantic Textual Similarity for English

		      TEST DATA WITH GOLD LABELS		   


This set of files describes the DATASET for the Semantic Textual Similarity 2014. 

The test dataset contains the following:

  00-README.txt 		  this file
  correlation-noconfidence.pl	  evaluation script

  sts2012-train.tgz               train data for sts 2012
  sts2012-test.tgz                test data for sts 2012
  sts2013-test.tgz                test data for sts 2013

  STS.gs.image.txt             gold standard file
  STS.gs.OnWN.txt              "
  STS.gs.tweet-news.txt        "
  STS.gs.deft-news.txt         "
  STS.gs.deft-forum.txt        "
  STS.gs.headlines.txt         "

  STS.input.image.txt             tab separated input file with 
                                  sentence pairs
  STS.input.OnWN.txt              "
  STS.input.tweet-news.txt        "
  STS.input.deft-news.txt         "
  STS.input.deft-forum.txt        "
  STS.input.headlines.txt         "

  STS.output.headlines.en.txt     tab separated sample input file with 
                                  sentence pairs


Introduction
------------

Given two sentences of text, s1 and s2, the systems participating in
this task should compute how similar s1 and s2 are, returning a
similarity score, and an optional confidence score.

The test dataset comprises the 2012 and 2013 datasets, which can be used to
develop and train systems.

We include sample data for the test datasets, coming
from the following:

1) image description (image)
2) OntoNotes and WordNet sense definition mappings (OnWN)
3) news title and tweet comments (tweet-news)
4) deft discussion forum and news (deft-forum and deft-news)
5) news headlines (headlines)


The datasets has been derived as follows:

- STS.input.image.txt: The Image Descriptions data set is a subset of
  the PASCAL VOC-2008 data set (Rashtchian et al., 2010) . PASCAL
  VOC-2008 data set consists of 1,000 images and has been used by a
  number of image description systems. The image captions of the data
  set are released under a CreativeCommons Attribution-ShareAlike
  license, the descriptions itself are free.

- STS.input.OnWN.txt: The sentences are sense definitions from WordNet
  and OntoNotes. 5 pairs of sentences.

- STS.input.tweet-news.txt: The tweet-news data set is a subset of the
  Linking-Tweets-to-News data set (Guo et al., 2013), which consists
  of 34,888 tweets and 12,704 news articles.  The tweets are the
  comments on the news articles.  The news sentences are the titles of
  news articles.

- STS.input.deft-news.txt: A subset of news article data in the DEFT
  project.

- STS.input.deft-forum.txt: A subset of discussion forum data in the
  DEFT project.

- STS.input.headlines.txt: we used headlines mined from several news
  sources by European Media Monitor using the RSS feed.
  http://emm.newsexplorer.eu/NewsExplorer/home/en/latest.html 



NOTE: Participant systems should NOT use the following datasets to
develop or train their systems: 

- Ontonotes - Wordnet sense aligned definitions.
- Data released in (Guo et al., 2013).
- The train data for task 1 (seminal 2014)


Input format
------------

The input file consist of two fields separated by tabs:

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

The gold standard in the test data will be assembled using mechanical
turk, gathering 5 scores per sentence pair. The gold standard score
will the average of those 5 scores. In this test dataset, this is
just a dummy number which you can ignore.



Answer format
--------------

The answer format is similar to the gold standard format, but includes
an optional confidence score. Each line has two fields separated by a
tab:

- a number between 0 and 5 (the similarity score)
- a number between 0 and 100 (the confidence score)

The use of confidence scores is experimental, and it is not required
for the official score.

An example is file STS.output.headlines.en.txt
Because there are two tasks this year, please add an extension “.en”
at the end of the system output submission file for the English task.


Scoring
-------

The official score is based on the average of Pearson correlation. The use of
confidence scores will be experimental, and it is not required for the
official scores. 



Participation in the task
-------------------------

Participant teams will be allowed to submit three runs at most.



Other
-----

Please check http://alt.qcri.org/semeval2014/task10/ for more details.



Authors
-------

Eneko Agirre
Daniel Cer
Mona Diab
Aitor Gonzalez-Agirre
Weiwei Guo
German Rigau



References
----------

Eduard Hovy, Mitchell Marcus, Martha Palmer, Lance Ramshaw, and Ralph
Weischedel. 2006. Ontonotes: The 90% solution. In Proceedings of the
Human Language Technology Conference of the North American Chapter
of the ACL.

Cyrus Rashtchian, Peter Young, Micah Hodosh, and Julia Hockenmaier. 
Collecting Image Annotations Using Amazon's Mechanical Turk. 
In Proceedings of the NAACL HLT 2010 Workshop on Creating Speech and 
Language Data with Amazon's Mechanical Turk.

Weiwei Guo, Hao Li, Heng Ji and Mona Diab. 2013.
Linking Tweets to News: A Framework to Enrich Online Short Text Data 
in Social Media.  In Proceedings of the 51th Annual Meeting of the 
Association for Computational Linguistics

