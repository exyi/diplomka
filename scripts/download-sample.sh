#!/usr/bin/bash

# Downloads about 70 structures for local experiments

cd "$(dirname "$0")"
cd ..
rsync 'blackbox:/scratch/non_redundant/csv/2d*' data/sample/
