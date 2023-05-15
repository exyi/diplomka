#!/usr/bin/env bash

#PBS -N ntcnet-tf.py
#PBS -q gpu
#PBS -l select=1:ncpus=6:ngpus=1:mem=16gb:scratch_local=12gb
#PBS -l walltime=10:00:00
#PBS -m ae

## paste this into an interactive job

nvidia-smi

module add zstd/1.5.2-gcc-10.2.1-jj6bl5j
module add python/3.10.4-gcc-8.3.0-ovkjwzd
module add cuda/cuda-11.2.0-intel-19.0.4-tn4edsz 
# module add cudnn/cudnn-8.1.0.77-11.2-linux-x64-intel-19.0.4-wx22b5t

export TMPDIR=$SCRATCHDIR/tmp
mkdir $TMPDIR
cd "$SCRATCH"
tar -I zstd -xf /storage/brno2/home/exyi/tf-venv.tar.zst
tar -I xz -xf /storage/praha1/home/exyi/cudnnlibs/cudnn-linux-x86_64-8.9.1.23_cuda11-archive.tar.xz
VENV_DIR="$SCRATCH/tf-venv"
NVIDIA_DIR="$(dirname "$(dirname "$(which nvcc)")")"
export XLA_FLAGS="--xla_gpu_cuda_data_dir=$NVIDIA_DIR"
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$SCRATCH/cudnn-linux-x86_64-8.9.1.23_cuda11-archive/lib
export CPATH=$CPATH:$SCRATCH/cudnn-linux-x86_64-8.9.1.23_cuda11-archive/include

# venv python is just a symlink to local python installation, we need to fix it to point to the correct python on the specific machine
real_python="$(which python3.10)"
rm "$VENV_DIR/bin/python3.10"
ln -s "$real_python" "$VENV_DIR/bin/python3.10"

if test -z "$TRAINING_SET"; then
	TRAINING_SET=/mnt/storage-brno12-cerit/nfs4/home/exyi/rna-csvs/train_set.tfrecord.gz
fi
if test -z "$VAL_SET"; then
	VAL_SET=/mnt/storage-brno12-cerit/nfs4/home/exyi/rna-csvs/rna-puzzles-test.tfrecord.gz
fi
if test -z "$TBLOG"; then
	TBLOG=i
fi

# define a DATADIR variable: directory where the input files are taken from and where output will be copied to
user=$(whoami)

if [ $user = "exyi" ]; then
  DATADIR=/storage/praha1/home/exyi/ntcnet
elif [ $user = "TODO" ]; then
  DATADIR=TODO
fi

cd "$DATADIR"
py="$VENV_DIR/bin/python"
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
