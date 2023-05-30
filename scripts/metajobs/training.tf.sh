#!/usr/bin/env bash

#PBS -N ntcnet-tf.py
#PBS -q gpu
#PBS -l select=1:ncpus=6:ngpus=1:mem=16gb:scratch_local=12gb
#PBS -l walltime=10:00:00
#PBS -m ae

set -e

nvidia-smi

# define a DATADIR variable: directory where the input files are taken from and where output will be copied to
user=$(whoami)

if [ $user = "exyi" ]; then
  DATADIR=/storage/praha1/home/exyi/ntcnet
elif [ $user = "TODO" ]; then
  DATADIR=TODO
fi

cd "$DATADIR"

source scripts/setup_tf.sh "$SCRATCH"

if test -z "$TRAINING_SET"; then
	TRAINING_SET=/storage/brno12-cerit/home/exyi/rna-csvs/train_set.tfrecord.gz
fi
if test -z "$VAL_SET"; then
	VAL_SET=/storage/brno12-cerit/home/exyi/rna-csvs/rna-puzzles-test.tfrecord.gz
fi
if test -z "$TBLOG"; then
	TBLOG=tt
fi

cd "$DATADIR"
function train {
	current_logdir="$DATADIR/metac-logs/$TBLOG-`date --utc +%y%m%d-%H%M`"
	mkdir -p "$current_logdir"
	hostname >> "$current_logdir/hostinfo.txt"
	nvidia-smi >> "$current_logdir/hostinfo.txt"
	$py "$DATADIR/model/training_tf.py" --train_set "$TRAINING_SET" --val_set "$VAL_SET" --logdir "$current_logdir" $@ 2>&1 | tee "$current_logdir/stdouterr.txt"
}

if test -n "$CMD"; then
    eval "$CMD"
else
    train $ARGS
fi
