Example Models
==============

Yu et al., 1412 baseline - mean embeddings, projection matrix and dot-product
distance measure:

  * **anssel_yu1412.py** shows the yu1412 classifier applied to its original
    "answer sentence selection" task

  * **sts_yu1412.py** shows a simple embedding-based classifier hacked
    together on a few lines that wouldn't end up dead last in STS2015
    competition

Tai et al., 1503.00075 baseline - mean embeddings, elementwise comparison
features and hidden layer:

  * **sts_kst1503.py** applies this to the STS and SICK tasks, reproducing
    the "mean vector" baseline in the paper

  * **anssel_kst1503.py** applies this to the answer sentence selection task

Full-fledged Keras architecture using the KeraSTS toolkit (but as
a standalone script instead of the models+tools abstraction architecture):

  * **anssel_cnn.py** generates summary sentence embeddings using a CNN
