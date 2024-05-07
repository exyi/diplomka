#!/usr/bin/env bash

partitions=20

for partition in $(seq 0 $((partitions-1))); do
    qsub -v "BASEPAIRS_PARTITION=$partition/$partitions" -N "basepairs-part$partition-$partitions" ./batch-basepairs.sh
done
