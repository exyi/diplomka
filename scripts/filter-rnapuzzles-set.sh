#!/usr/bin/env bash

cd "$(dirname "$0")"
cd ..

from="$1"
to="$2"

mkdir -p "$to"

for puzzle in $(cat rna-puzzles.txt); do
	echo "$puzzle"
	cp "$from/$puzzle"*.csv* "$to"
done
