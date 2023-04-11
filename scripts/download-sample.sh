#!/usr/bin/bash

# Downloads about 70 structures for local experiments

cd "$(dirname "$0")"
cd ..
mkdir -p data/sample
mkdir -p data/sample2
rsync 'blackbox:/scratch/non_redundant/csv/2d*' data/sample/
rsync 'blackbox:/scratch/non_redundant/csv/3a*' data/sample2
