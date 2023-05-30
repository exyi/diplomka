#!/usr/bin/env bash

#PBS -N rnacentral-prep
#PBS -l select=1:ncpus=1:mem=8gb:scratch_local=12gb
#PBS -l walltime=60:00:00
#PBS -m ae

set -e

# define a DATADIR variable: directory where the input files are taken from and where output will be copied to
user=$(whoami)

DATADIR=/storage/praha1/home/exyi/ntcnet

cd "$DATADIR"

source scripts/setup_tf.sh "$SCRATCH"

cd /storage/brno12-cerit/home/exyi/rnacentral

mkdir -p tf-all
mkdir -p tf-rRNA
mkdir -p tf-norRNA
mkdir -p tf-norRNA-notRNA

function run {
	output=$1
	shift

	$py $DATADIR/rnacentral_pretrain/rnacentral_dataset.py --input rnacentral_active.fasta.gz rnacentral_inactive.fasta.gz --verbose --output "$output"/rnacentral.json "$@" 2>&1 | tee "$output"/job.stdout
}

run tf-all
run tf-rRNA --only_types rRNA
run tf-norRNA --exclude_types rRNA
run tf-norRNA-notRNA --exclude_types rRNA tRNA
