::::::::::::::::::::::: University of Trento - Italy ::::::::::::::::::::::::::::::::

:::::::::::: SICK (Sentences Involving Compositional Knowledge) data set ::::::::::::


The SICK data set consists of 10,000 English sentence pairs, built starting from two existing 
paraphrase sets: the 8K ImageFlickr data set (http://nlp.cs.illinois.edu/HockenmaierGroup/data.html) 
and the SEMEVAL-2012 Semantic Textual Similarity Video Descriptions data set 
(http://www.cs.york.ac.uk/semeval-2012/task6/index.php?id=data). Each sentence pair is annotated 
for relatedness in meaning and for the entailment relation between the two elements.


The SICK data set is released under a Creative Commons Attribution-NonCommercial-ShareAlike 3.0 
Unported License (http://creativecommons.org/licenses/by-nc-sa/3.0/deed.en_US)


The SICK data set is used in SemEval 2014 - Task 1: Evaluation of compositional distributional 
semantic models on full sentences through semantic relatedness and textual entailment


The current release is a subset of the data set representing Task 1 Train data (4500 sentence pairs)


File Structure: tab-separated text file


Fields:

- sentence pair ID

- sentence A

- sentence B

- semantic relatedness gold label (on a 1-5 continuous scale)

- textual entailment gold label (NEUTRAL, ENTAILMENT, or CONTRADICTION)
