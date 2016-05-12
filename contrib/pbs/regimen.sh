#!/bin/bash
# Run a basic regimen of models on a given dataset.
#
# Usage: regimen.sh NB_RUNS BASE_TIME DATAID [IDSUFFIX [CONFIG_PARAM]...]
#
# Example: regimen.sh 4 1 al
#
# BASE_TIME is a top estimate of how long a single RNN run takes on the dataset
# (in hours).

nb_runs="$1"; shift
base_time="$1"; shift
dataid="$1"; shift
jobid="$1"; shift

prio=0
mem=16gb

walltime() {
	coef=$1
	h=$(awk -v nb_runs=$nb_runs -v base_time=$base_time -v coef=$coef 'END{h=int(base_time*nb_runs*coef + 0.99); print(h>24 ? 24 : (h<1 ? 1 : h))}' </dev/null)
	echo "$h:00:00"
}

case $dataid in
	ay)
		task=anssel
		data="data/anssel/yodaqa/curatedv2-training.csv data/anssel/yodaqa/curatedv2-val.csv"
		;;
	al)
		task=anssel
		data="data/anssel/yodaqa/large2470-training.csv data/anssel/yodaqa/large2470-val.csv"
		;;
	aw)
		task=anssel
		data="data/anssel/wang/train-all.csv data/anssel/wang/dev.csv"
		;;
	au)
		task=asku
		data="data/para/askubuntu/train_random.txt data/para/askubuntu/dev.txt"
		;;
	rg)
		task=hypev
		data="data/hypev/argus/argus_train.csv data/hypev/argus/argus_val.csv"
		;;
	r8c)
		task=hypev
		data="data/hypev/ai2-8grade/ck12v0-train.csv data/hypev/ai2-8grade/ck12v0-dev.csv"
		;;
	r8e)
		task=hypev
		data="data/hypev/ai2-8grade/enwv0-train.csv data/hypev/ai2-8grade/enwv0-dev.csv"
		;;
	r8)
		task=hypev
		data="data/hypev/ai2-8grade/v0-train.csv data/hypev/ai2-8grade/v0-dev.csv"
		;;
	esi)
		task=rte
		data="data/rte/sick2014/SICK_train.txt data/rte/sick2014/SICK_trial.txt"
		;;
esac

qsubopts="-p $prio -q gpu -l nodes=1:gpu=1:ppn=1:^cl_gram:^cl_konos -l mem=16gb"

~/ssub "$qsubopts -l walltime=$(walltime 0.5) -N R_${dataid}_2avg$jobid" tools/train.py avg $task $data nb_runs=$nb_runs "$@"
~/ssub "$qsubopts -l walltime=$(walltime 0.5) -N R_${dataid}_2dan$jobid" tools/train.py avg $task $data inp_e_dropout=0 inp_w_dropout=1/3 deep=2 "pact='relu'" nb_runs=$nb_runs "$@"  # DAN
~/ssub "$qsubopts -l walltime=$(walltime 1) -N R_${dataid}_2rnn$jobid" tools/train.py rnn $task $data nb_runs=$nb_runs "$@"
~/ssub "$qsubopts -l walltime=$(walltime 1) -N R_${dataid}_2cnn$jobid" tools/train.py cnn $task $data nb_runs=$nb_runs "$@"
~/ssub "$qsubopts -l walltime=$(walltime 1.5) -N R_${dataid}_2rnncnn$jobid" tools/train.py rnncnn $task $data nb_runs=$nb_runs "$@"
~/ssub "$qsubopts -l walltime=$(walltime 1.5) -N R_${dataid}_2a51$jobid" tools/train.py attn1511 $task $data nb_runs=$nb_runs "$@"
