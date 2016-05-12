MCTest Machine Comprehension Dataset
====================================

This dataset comes from

	http://research.microsoft.com/en-us/um/redmond/projects/mctest/

and contains 300 children stories with many sentences and 4 questions each.

Rather than Q/A pairs, we take advantage of the "Statements" dataset
provided along with the original dataset, which just contains statements.
These are our "htext" while "mtext" are story sentences.

The state-of-art is probably 1602.04341 Yin et al.'s HABCNN-TE
(http://arxiv.org/pdf/1602.04341v1.pdf).
