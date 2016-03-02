#!/bin/bash

#python argus_tests.py cnn data/hypev/argus/argus_train.csv data/hypev/argus/argus_test.csv inp_e_dropout=0.0 dropout=0.0 nb_epoch=3

models=(rnn cnn cnnrnn attn1511)

dropouts=(0.0 0.2 0.4 0.6 0.75)

embd=(50 100 300)

for model in ${models[@]}; do
	for dropout in ${dropouts[@]}; do
		for num in ${embd[@]}; do
			echo $model $dropout $num
			python argus_tests.py $model data/hypev/argus/argus_train.csv data/hypev/argus/argus_test.csv inp_e_dropout=0.0 dropout=$dropout nb_epoch=3 embdim=$num
		done
	done
done
