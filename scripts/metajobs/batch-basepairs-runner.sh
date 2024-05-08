#!/usr/bin/env bash

partitions=40

for partition in $(seq 0 $((partitions-1))); do
    echo qsub -v "BASEPAIRS_PARTITION=$partition/$partitions" -N "basepairs-part$partition-$partitions" ./batch-basepairs.sh
done
