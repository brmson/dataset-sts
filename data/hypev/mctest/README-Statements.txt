MCTest: Machine Comprehension Test
http://research.microsoft.com/mct

----------------------------------------
Statements Data
----------------------------------------

This directory contains the MCTest data, but with questions and answers converted
into statements, using the rules employed in a web-based question answering
system [1]. We ignored the target type (person, time, etc) output by the system, 
and only selected the first sentence (the system produced multiple sentences, roughly
ordered by likelihood to be a valid sentence). These are the statements which were
used to evaluate an off-the-shelf RTE system for the MCTest data. See the MCTest 
paper [2] for more details. Please also see the MCTest release readme.txt for
more details on the dataset and file naming and file formats, downloaded
from
http://research.microsoft.com/en-us/um/redmond/projects/mctest/data.html
or the README, contained at
http://research.microsoft.com/en-us/um/redmond/projects/mctest/readme.txt

In this archive, we provide the data in to file formats: .tsv and .pairs.

The TSV format is the same as the MCTest TSV format, except each possible answer
has been replaced with a sentence that is meant to be the sentential version 
of that particular question and particular answer.

The pairs format is the same as the standard RTE-6 XML data format, where 
one pair is made per answer. The correct answer is marked as ENTAILMENT, and
the incorrect answer is marked as "UNKNOWN". We also experimented with 
marking incorrect answers as CONTRADICTION, but it made no difference in the results.
The order of the pairs is: 
  foreach story{ foreach question { foreach answer { output pair}}}

---------------------------------------
License
---------------------------------------

Please see the included license.pdf for the license

----------------------------------------
References
----------------------------------------

[1] S. Cucerzan and E. Agichtein. 2005. Factoid Question Answering over Unstructured 
    and Structured Content on the Web. In Proceedings of the Fourteenth Text 
    Retrieval Conference (TREC).

[2] M. Richardson, C.J.C. Burges, and E. Renshaw. 2013. MCTest: A Challenge Dataset
    for the Open-Domain Machine Comprehension of Text. In Proceedings of the 
    Conference on Empirical Methods in Natural Language Processing (EMNLP).


