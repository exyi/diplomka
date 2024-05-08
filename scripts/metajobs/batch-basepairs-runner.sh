#!/usr/bin/env bash

partitions=15

for partition in $(seq 0 $((partitions-1))); do
    echo qsub -v "BASEPAIRS_PARTITION=${partition}0-$((partition+1))0/${partitions}0" -N "basepairs-part$partition-$partitions" ./batch-basepairs.sh
done
