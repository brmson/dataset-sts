#!/bin/bash
@CMD@
me=$(readlink -f "$0")
export HOME=/storage/ostrava1/home/$USER
#if [[ $HOSTNAME == zubat* ]]; then
#	export THEANO_FLAGS=base_compiledir=/storage/brno2/home/$USER/theanoc
#fi
export THEANO_FLAGS=base_compiledir=/tmp/pasky-theanoc
cd $HOME
. init_script.sh
cd dataset-sts
mkdir -p ../joblogs

{
echo python -u "${cmd[@]}"
python -u "${cmd[@]}"
} | tee ../joblogs/$PBS_JOBID.$PBS_JOBNAME
