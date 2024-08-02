#!/usr/bin/env bash

rm -rf att.zip

7z a att.zip data/refset/ 7z data/parameter-boundaries.csv diplomka/ .gitignore pipeline-refset.sh scripts/ webapp webapp-build/ '-x!node_modules/*' '-x!*.pse' '-x!*.js.map'

7z d att.zip webapp/node_modules/
7z d att.zip webapp/docs
7z d att.zip webapp/.svelte-kit
7z d att.zip scripts/__pycache__
7z d att.zip diplomka/html/vendor
7z d att.zip diplomka/out
7z d att.zip diplomka/tmp
7z d att.zip notes
7z d att.zip zip.sh
