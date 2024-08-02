#!/usr/bin/env bash

#PBS -N batch-basepairs
#PBS -l select=1:ncpus=10:mem=30gb:scratch_local=12gb
#PBS -l walltime=14:00:00
#PBS -m ae

# BASEPAIRS_PARTITION=1/40

echo "Running on $(hostname)"

module add zstd/1.5.2-gcc-10.2.1-jj6bl5j
module add python/3.10.4-gcc-8.3.0-ovkjwzd

DATADIR=/storage/praha1/home/exyi/basepairs

# cd "$SCRATCH"
# python -m venv new-venv
# new-venv/bin/pip install --no-cache-dir --no-deps -r /storage/praha1/home/exyi/basepairs-git/pairclusters/requirements.txt

cd "$SCRATCH"
tar -I zstd -xf /storage/praha1/home/exyi/basepairs-venv.tar.zst
py="$SCRATCH/basepairs-venv/bin/python"

cd /storage/praha1/home/exyi/basepairs-git/pairclusters

if [ -z "$BASEPAIRS_PAIR_TYPE" ]; then
    BASEPAIRS_PAIR_TYPE="all"
fi

if [ -z "$BASEPAIRS_PARTITION" ]; then
    partition_settings="--partition-input=1"
else
    partition_settings="--partition-input-select=$BASEPAIRS_PARTITION"
fi

export PYTHONUNBUFFERED=1

$py ./pairs.py --override-pair-family=$BASEPAIRS_PAIR_TYPE\
    --filter --dedupe --postfilter-hb=4.2 --postfilter-shift=2.5\
    --reference-basepairs="$DATADIR/reference-basepairs-KDEyawpitchroll.json"\
    --pdbcache="$DATADIR/pdbcache"\
    --threads="$PBS_NCPUS"\
    --output=$DATADIR/out/all-$BASEPAIRS_PAIR_TYPE.parquet\
    "$DATADIR/fr3d-exportonly-ID.parquet" "$DATADIR/vsecky-paryIDinv.parquet" "$DATADIR/vsecky-paryID.parquet"\
    $partition_settings 2>&1 | tee $DATADIR/out/all-$BASEPAIRS_PAIR_TYPE-p${BASEPAIRS_PARTITION/\//of}.log
