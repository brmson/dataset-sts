MCTest: Machine comprehension test
http://research.microsoft.com/mct

---------------------------------------
Welcome
---------------------------------------

In this archive, you'll find the 660 reading comprehension 
tests (split into MC160 and MC500) as described in this paper:

Matthew Richardson, Christopher J.C. Burges, and Erin Renshaw (2013).  
"MCTest: A Challenge Dataset for the Open-Domain Machine Comprehension of
Text," in Proceedings of the 2013 Conference on Emprical Methods in Natural 
Language Processing (EMNLP 2013), pp. 193-203, Seattle, WA, USA, October 2013.
http://research.microsoft.com/en-us/um/redmond/projects/mctest/MCTest_EMNLP2013.pdf

If you use this data, we ask that you reference the above paper so that 
others may also easily find and use the data themselves. If you have any
questions or find the data useful, please don't hesitate to contact us:
 
  Matt Richardson: mattri@microsoft.com
  Chris Burges:    cburges@microsoft.com
  Erin Renshaw:    erinren@microsoft.com
  
---------------------------------------
Train, Development, and Test sets
---------------------------------------

There are two primary datasets: mc500 and mc160. See our paper (link above)
for a description of the differences between these two sets. MC500 contains
500 story sets (a story set is a story and its associated questions); MC160
contains 160.

Each dataset is split into train, development, and test sets, with each
story set assigned to one. As one would expect, the train set can be used 
to train your algorithm, or to examine closely to see what kind of phenomenon
exist in the stories, or any other purpose you would like to use it for. 
The development set is intended to be a set-aside set that you can use for
evaluating your algorithm. The test set is intended to be a final test set
that you use only once, to get your final results for publishing.

The answers for the test set are distributed in a separate archive. We 
encourage you to download this file only when you are ready to do your 
final evaluation. For both TSV and TXT format, downloading the answers
archive will allow you to use the same code as was used when evaluating
on the development set.

The files are named: mc[500|160].[train|dev|test].[txt|tsv|ans]
  
---------------------------------------
License
---------------------------------------

Please see the included license.pdf for the license

---------------------------------------
Data Format
---------------------------------------

We provide the data into two separate formats: .txt for ease of 
readability, and .tsv/.ans for ease of coding. The two formats contain 
the same information, just in different formats.

In both formats, questions are prefixed with "one:" or "multiple:", 
indicating whether the author marked that the question required one or 
multiple sentences from the story in order to find the right answer.

The text in the stories has been processed as little as possible to remain
true to the original author's submission. Special non-ASCII characters such
as curly quotation marks, em-dashes, and elipses have been converted to 
their ASCII equivalent. 

---- TXT Format ----

The .txt files present the stories and questions in a convenient text 
format, for ease of reading. The correct answer for each question is marked 
with a "*". The format is consistent, so can be programmatically read if 
desired, but may be harder to write a parser for than the TSV format.
Note that the test story sets do not indicate the correct answer (see 
not above on train/dev/test split). To get the test sets in text format
that do contain the answer, you must download the test answers archive.

---- TSV/ANS Format ----

These consist of tab-delimited files, with one story set per line. 
The .tsv file contains the story, questions, and answers.
The .ans file contains the correct answer for each question

The format of a line in the TSV file is:
  Id <tab> properties <tab> story <tab> q1 <tab> q2 <tab> q3 <tab> q4 
    where
  qN = questionText <tab> answerA <tab> answerB <tab> answerC <tab> answerD
    and
  properties is a semicolon-delimited list of property:value pairs, including
    Author (anonymized author id, consistent across all files)
    Work Time(s): Seconds between author accepting and submitting the task
    Qual. score: The author's grammar qualification test score (% correct)
    Creativity Words: Words the author was given to encourage creativity
    (there are no creativity words or qual score for mc160, see paper)

The format of a line in the ANS file is:
  answer1 <tab> answer2 <tab> answer3 <tab> answer4
    where
  answerN is the correct answer (A, B, C, or D) for question N

Finally, because some authors used newlines and/or tabs to indicate
paragraph separation, and this would break the TSV format, we have
replaced any newline or tab with "\newline" or "\tab", respectively.
No questions or answers required this escaping.

---------------------------------------
An Ongoing Resource
---------------------------------------

As we stated in our paper, we will maintain the website with links
to the latest published results using this data. If you publish a paper
using this data, let us know and we will add a link to it.

Also, if you publish a paper and are willing to share your scoring
files with others, send them to us and we will post them as well. We
hope that by providing the scoring files from previous work, we will
enable more rapid progress on this problem -- by enabling each new
algorithm to build on top of previous algorithmic results, by 
allowing pairwise statistical significance testing, and by allowing 
anyone to investigate what kind of errors are being made by previous
work.

---------------------------------------
Score Files
---------------------------------------

To send us your scores, please format them in the following tab-delimited
format, one line per story (similar to the ANS file):

  scores1 <tab> scores2 <tab> scores3 <tab> scores4 <tab>
    where
  scoresN are the scores for question N, and have the format:
  scoresN = scoreN_A, scoreN_B, scoreN_C, scoreN_D
  where scoreN_A is the score your algo assigns to answer A for question N.

The scores may be probabilities, or may simply be unnormalized real values. 
It is assumed that the highest score is the one your algorithm would select,
and the higher the score, the more confident it is in that selection.

example:
  3.2, 1.1, 0.9, 3.1 <tab> -0.3, 1.1, -4.3, 0.4 <tab> ...
would mean the algorithm selects "A" for question 1, and "B" for question 2.
  
We will provide these score files for public download, attributed to you and
with a link to your publication if possible. 

