#!/usr/bin/env bash

#PBS -N ntcnet-interactive
#PBS -q gpu
#PBS -l select=1:ncpus=3:ngpus=1:mem=16gb:scratch_local=8gb
#PBS -l walltime=2:00:00
#PBS -m ae

## paste this into an interactive job

nvidia-smi
DATADIR=/storage/praha1/home/exyi/ntcnet
RNA_CSVS=/storage/brno12-cerit/home/exyi/rna-csvs
BASEPAIRING_DIR=/storage/plzen1/home/exyi/fr3d-python/output-dir/
cd "$DATADIR"
source scripts/setup_tf.sh "$SCRATCH"

function train {
	$py "$DATADIR/model/training_tf.py" --train_set "$TRAINING_SET" --val_set "$VAL_SET" --logdir "$DATADIR/metac-logs/$TBLOG-`date --utc +%y%m%d-%H%M`" $@
}


# cd $RNA_CSVS
# $py $DATADIR/model/dataset_tf.py --input ./train_set --pairing_input $BASEPAIRING_DIR --output train_set_nodna.tfrecord.gz --dna_handling ignore --verbose
