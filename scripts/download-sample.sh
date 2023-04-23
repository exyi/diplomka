#!/usr/bin/bash

# Downloads about 70 structures for local experiments

cd "$(dirname "$0")"
cd ..
mkdir -p data/sample
mkdir -p data/sample2
mkdir -p data/rna-puzzles-test
rsync 'blackbox:/scratch/non_redundant/csv/2d*' data/sample/
rsync 'blackbox:/scratch/non_redundant/csv/3a*' data/sample2


for puzzle in $(cat rna-puzzles.txt); do
	rsync "blackbox:/scratch/non_redundant/csv/$puzzle*.csv*" data/rna-puzzles-test
done
