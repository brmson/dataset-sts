WebQuestions-Movies Property Selection QA Dataset
=================================================

This task tries to find property path which leads from question entity to answer entity.
For example, in the question "Who directed Harry Potter" the correct property(relation) is "Directed by".

The dataset consists of pairs question - property path. For each question, there are such paths of maximal length 2
which are connected to each enitity linked during entity linking. If the question has two or more entities, and they can be connected
by path with the same first property as the path from question entity to answer entity, the second ("witness") property is concatenated to path
as third property. The properties on the path are separated by # symbol.

There are two types of data sets:

  * trainmodel.csv, test.csv, val.csv contain original question representation (val.csv also contains devtest split of the source QA dataset)
  * Files with prefix enttok- have named entities replaced by ENT_TOK. Only entities, which participated on given path creation were replaced.

These files can be generated using scipt propsel-dataset-refresh.sh contained in https://github.com/brmson/dataset-factoid-webquestions.
