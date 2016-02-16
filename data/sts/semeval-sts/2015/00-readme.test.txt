
			       STS 2015

                  English Semantic Textual Similarity

                      Test - Evaluation Package


This set of files comprises:

- the test input files

  STS.input.answers-forums.txt
  STS.input.answers-students.txt
  STS.input.belief.txt
  STS.input.headlines.txt
  STS.input.images.txt

- license related files

  STS.input.answers-forums.LICENSE
  STS.answers-forums.zip

- the gold standard files

  STS.gs.answers-forums.txt
  STS.gs.answers-students.txt
  STS.gs.belief.txt
  STS.gs.headlines.txt
  STS.gs.images.txt


- the script for evaluation

  correlation-noconfidence.pl	

  For example:
  $ perl correlation-noconfidence.pl gs sys


- the code for the baseline

  $ corebaseline-tokencos.tar.gz


See train data release for more details.


Authors
-------

Eneko Agirre
Dan Cer
Mona Diab
Aitor Gonzalez-Agirre
Weiwei Guo
German Rigau

