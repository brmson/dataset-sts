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

A full-fledged but simple Keras architectures:

  * **anssel_rnn.py** uses GRU hidden states to approach 2015-state-of-art
  * **anssel_rnn_eval.py** to measure anssel performance using the official tool
  * **anssel_cnn.py** generates summary sentence embeddings using a CNN
