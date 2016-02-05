                       SemEval 2016 Shared Task 1
                    Semantic Textual Similarity (STS)
                               English 

This package contains the test sets for the 2016 Semantic Textual Similarity
(STS) shared task. Each evaluation set has the following tab-separated format:

  * One STS pair per line.
  * Each line contains the following fields: STS Sent1, STS Sent2,
    Sent 1 Source Notes, Sent 2 Source Notes

The official input files are provided in UTF-8. As a convenience, alternative
ASCII encodings of the STS pairs are also provided.

Example:

Should I drink water during my workout?	How can I get my toddler to drink more water?	StackExchange Network: http://fitness.stackexchange.com/questions/1902 Author: Rogach (http://fitness.stackexchange.com/users/132) Last Editor: Nathan Wheeler (http://fitness.stackexchange.com/users/21)	StackExchange Network: http://parenting.stackexchange.com/questions/11704 Author: I Like to Code (http://parenting.stackexchange.com/users/4718) 

-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*
Important: STS systems must *NOT* inspect or otherwise make use of the source
notes field while participating in the shared task.
-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*

Input Files (UTF-8):
--------------------
STS2016.input.answer-answer.txt
STS2016.input.headlines.txt
STS2016.input.plagiarism.txt
STS2016.input.postediting.txt
STS2016.input.question-question.txt

Alternative Input Files (ASCII):
--------------------------------
STS2016.input.answer-answer.ascii
STS2016.input.headlines.ascii
STS2016.input.plagiarism.ascii
STS2016.input.postediting.ascii
STS2016.input.question-question.ascii

Output Files: 
-------------
For each evaluation set, please generate a plain text output file of the form
STS2016.OUTPUT.$TEAM_NAME.$RunName.$DATA_SET.txt

Example: STS2016.OUTPUT.MyTeam.LSTM-1.$Run.answers-answer.txt

The evaluation file should have one line for each STS pair that provides the 
score assigned by your system as a floating point number:

0.1
4.9
3.5
2.0
5.1

By default, each team can submit up to three runs. Teams that feel they have a 
good reason for submitting more than three runs should contact the organizers.
Such requests will be handled on a case by case basis.

Good Luck!
