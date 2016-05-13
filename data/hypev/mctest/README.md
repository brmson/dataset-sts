MCTest Machine Comprehension Dataset
====================================

This dataset comes from

	http://research.microsoft.com/en-us/um/redmond/projects/mctest/

and contains children stories with many sentences and 4 questions each.

The standard dataset contains of two subsets that differ by their precise
origin, but seem similar in character - mc160 and mc500.  We also just
concatenated them together to create a cummulative dataset mc660, to get
more data to train the models as the samples are pretty tiny otherwise.

Rather than Q/A pairs, we take advantage of the "Statements" dataset
provided along with the original dataset, which just contains statements.
These are our "htext" while "mtext" are story sentences.

The state-of-art is probably 1602.04341 Yin et al.'s HABCNN-TE
(http://arxiv.org/pdf/1602.04341v1.pdf).
