#!/usr/bin/env bash

BASEPAIRS_PARTITION=0-32/32

echo "Running on $(hostname)"

DATADIR=~/basepairs/data

py=~/basepairs/venv/bin/python

cd ~/basepairs

if [ -z "$BASEPAIRS_PAIR_TYPE" ]; then
    BASEPAIRS_PAIR_TYPE="all"
fi

if [ -z "$BASEPAIRS_PARTITION" ]; then
    partition_settings="--partition-input=1"
else
    partition_settings="--partition-input-select=$BASEPAIRS_PARTITION"
fi

runname=all4

export PYTHONUNBUFFERED=1
pdbcache=~/tmp/pdbcache

$py ~/ntcnet/pairclusters/pairs.py --override-pair-family=$BASEPAIRS_PAIR_TYPE\
    --filter --dedupe --postfilter-hb=4.2 --postfilter-shift=2.5\
    --reference-basepairs="$DATADIR/reference-basepairs-KDEyawpitchroll.json"\
    --pdbcache="$pdbcache"\
    --threads=70\
    --output=$DATADIR/out/$runname.parquet\
    "$DATADIR/fr3d-exportonly-ID.parquet" "$DATADIR/vsecky-paryIDinv.parquet" "$DATADIR/vsecky-paryID.parquet"\
    $partition_settings 2>&1 | tee $DATADIR/out/$runname-p${BASEPAIRS_PARTITION/\//of}.log
