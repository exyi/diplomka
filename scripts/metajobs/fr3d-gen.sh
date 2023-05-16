#!/usr/bin/env bash

#PBS -N fr3d-basepairing
#PBS -l select=1:ncpus=1:mem=3gb
#PBS -l walltime=60:00:00
#PBS -m ae

module add zstd/1.5.2-gcc-10.2.1-jj6bl5j
module add python/3.9.12-gcc-10.2.1-rg2lpmk

export TMPDIR=$SCRATCHDIR
# cd "$SCRATCH"

# python -m venv fr3d-venv
py="$(pwd)/fr3d-venv/bin/python"
# $py -m pip install mmcif-pdbx scipy numpy~=1.21.0
# git clone https://github.com/BGSU-RNA/fr3d-python

echo "Working directory: $(pwd)"

cd fr3d-python
ls /mnt/storage-brno12-cerit/nfs4/home/exyi/rna-csvs/csv | awk -F_ '{print $1}' > rna-structures.txt

$py << EOF
from fr3d.classifiers.NA_pairwise_interactions import generatePairwiseAnnotation
with open('rna-structures.txt') as f:
	for l in f:
		generatePairwiseAnnotation(l.strip(), None, "", "./output-dir", "basepair", "txt")
EOF

zstd -r -14 output-dir
