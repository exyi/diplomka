#!/bin/bash
#PBS -N ntcnet.py
#PBS -q gpu
#PBS -l select=1:ncpus=4:ngpus=1:mem=16gb:scratch_local=12gb
#PBS -l walltime=10:00:00
#PBS -m ae

set -e

module add zstd/1.5.2-gcc-10.2.1-jj6bl5j
module add python/3.9.12-gcc-10.2.1-rg2lpmk
module add cuda/cuda-11.2.0-intel-19.0.4-tn4edsz
# module add cudnn/cudnn-8.1.0.77-11.2-linux-x64-intel-19.0.4-wx22b5t

cd "$SCRATCH"
tar -I zstd -xf /storage/brno2/home/exyi/torch-venv.tar.zst

# venv python is just a symlink to local python installation, we need to fix it to point to the correct python on the specific machine
real_python="$(which python3.9)"
rm torch-venv/bin/python3.9
ln -s $real_python torch-venv/bin/python3.9

if test -z "$TRAINING_SET"; then
	TRAINING_SET=/mnt/storage-brno12-cerit/nfs4/home/exyi/rna-csvs/train_set
fi
if test -z "$VAL_SET"; then
	VAL_SET=/mnt/storage-brno12-cerit/nfs4/home/exyi/rna-csvs/rna-puzzles-test
fi
if test -z "$LOGNAME"; then
	LOGNAME=tt
fi

# define a DATADIR variable: directory where the input files are taken from and where output will be copied to
user=$(whoami)

if test $user = "exyi"; then
#   DATADIR=/auto/pruhonice1-ibot/home/exyi/ntcnet
  DATADIR=/storage/praha1/home/exyi/ntcnet
elif test $user = "TODO"; then
  DATADIR=TODO
fi

cd "$DATADIR"
py="$SCRATCH/torch-venv/bin/python"
function train {
	$py "$DATADIR/model/training.py" --train_set "$TRAINING_SET" --val_set "$VAL_SET" --logdir "$DATADIR/metac-logs/$LOGNAME-`date --utc +%y%m%d-%H%M`" $@
}

echo "init done, running the python script"


if test -n "$CMD"; then
    eval "$CMD"
else
    train $ARGS
fi
