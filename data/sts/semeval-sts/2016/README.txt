                       SemEval 2016 Shared Task 1
                    Semantic Textual Similarity (STS)
                               English 

This package contains test sets with gold standard labels for the 2016
Semantic Textual Similarity (STS) shared task. Each evaluation set input file
has the following tab-separated format:

  * One STS pair per line.
  * Each line contains the following fields: STS Sent1, STS Sent2,
    Sent 1 Source Notes, Sent 2 Source Notes

Each evaluation set gold standard label file has the following format:

  * One numeric STS label between 0 and 5 per line.
  * Blank lines indicate that the pair was not included in the official scoring.

Input files are provided in both UTF-8 and ASCII encodings.

Example:

Should I drink water during my workout?	How can I get my toddler to drink more water?	StackExchange Network: http://fitness.stackexchange.com/questions/1902 Author: Rogach (http://fitness.stackexchange.com/users/132) Last Editor: Nathan Wheeler (http://fitness.stackexchange.com/users/21)	StackExchange Network: http://parenting.stackexchange.com/questions/11704 Author: I Like to Code (http://parenting.stackexchange.com/users/4718) 

-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*
Important: As part of the official evaluation, STS systems were not allowed
to inspect or otherwise make use of the source notes field while participating
in the shared task.
-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*

Input Files (UTF-8):
--------------------
STS2016.input.answer-answer.txt
STS2016.input.headlines.txt
STS2016.input.plagiarism.txt
STS2016.input.postediting.txt
STS2016.input.question-question.txt

Input Files (ASCII):
--------------------
STS2016.input.answer-answer.ascii
STS2016.input.headlines.ascii
STS2016.input.plagiarism.ascii
STS2016.input.postediting.ascii
STS2016.input.question-question.ascii

Gold Standard Files:
--------------------
STS2016.gs.answer-answer.txt
STS2016.gs.headlines.txt
STS2016.gs.plagiarism.txt
STS2016.gs.postediting.txt
STS2016.gs.question-question.txt

Evaluation Script:
------------------
correlation-noconfidence.pl

Output Files: 
-------------
For each evaluation set, systems generate a plain text output file having one
line per STS pair. Each line provides the score assigned by an STS system for
that pair as a floating point number:

0.1
4.9
3.5
2.0
5.1

Evaluation Script:
------------------
The output files can be evaluated using the correlation-noconfidence.pl script
included in this package:

   ./correlation-noconfidence.pl STS2016.gs.<dataset>.txt \
                                 SYSTEM_OUT.<dataset>.txt 

Contact:
--------
STS 2016 Website: http://alt.qcri.org/semeval2016/task1/
Mailing List: sts-semeval@googlegroups.com

Organizers:
-----------
Eneko Agirre, Carmen Banea, Daniel Cer, Mona Diab, and 
Aitor Gonzalez-Agirre (STS Core)

Carmen Banea, Daniel Cer, Rada Mihalcea, and
Janyce Wiebe (Cross-lingual STS)
