#!/usr/bin/env bash

#PBS -N ntcnet-interactive
#PBS -l select=1:ncpus=20:mem=64gb:scratch_local=12gb
#PBS -l walltime=2:00:00
#PBS -m ae

## paste this into an interactive job


module add zstd/1.5.2-gcc-10.2.1-jj6bl5j
module add python/3.10.4-gcc-8.3.0-ovkjwzd

DATADIR=/storage/praha1/home/exyi/basepairs

# cd "$SCRATCH"
# python -m venv new-venv
# new-venv/bin/pip install --no-cache-dir --no-deps -r /storage/praha1/home/exyi/basepairs-git/pairclusters/requirements.txt



cd "$SCRATCH"
tar -I zstd -xf /storage/praha1/home/exyi/basepairs-venv.tar.zst
py="$SCRATCH/basepairs-venv/bin/python"

pair_type=cWW

cd /storage/praha1/home/exyi/basepairs-git/pairclusters

$py ./pairs.py --override-pair-family=$pair_type\
    --filter --dedupe --postfilter-hb=4.2 --postfilter-shift=2.5\
    --reference-basepairs="$DATADIR/reference-basepairs-KDEyawpitchroll.json"\
    --pdbcache="$DATADIR/pdbcache"\
    --threads="$PBS_NCPUS"\
    --output=$DATADIR/out/all-$pair_type.parquet\
    $DATADIR/fr3d-exportonly-ID.parquet\
    --partition_input=1 | tee $DATADIR/out/all-$pair_type.log
