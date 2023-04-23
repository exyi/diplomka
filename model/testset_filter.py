#!/usr/bin/env python3

import sys, os, random, argparse, math
from typing import List
import csv_loader
import numpy as np
import Bio.Align

def filter(
    iterator,
    test_sequences: List[str],
    identity_threshold = 0.8
):
    """
    Filter out sequences from the iterator that are in the test set
    """
    aligner = Bio.Align.PairwiseAligner(open_gap_score=-5, extend_gap_score=0.1, match_score=1, mismatch_score=-1, mode="local")
    total_count = 0
    filtered_count = 0
    for file, data in iterator:
        total_count += 1
        matches = False
        for train_chain in data[1].values():
            for test_seq in test_sequences:
                if len(train_chain["sequence"]) <= 15:
                    continue
                alignment = aligner.align(train_chain["sequence"], test_seq)
                if len(alignment) == 0:
                    continue

                counts = alignment[0].counts()
                fragments = np.diff(alignment[0].coordinates[1, :])
                # gap counts as 2 mismatches
                identity = (counts.identities - len(fragments) * 2) / len(test_seq)
                if alignment.score > 0 and identity >= identity_threshold:
                    matches = True
                    print(train_chain["sequence"], "is too similar to", test_seq)
                    break

            if matches:
                break

        if not matches:
            filtered_count += 1
            print("OK  ", file)
            yield file, data
        else:
            print("SKIP", file)

    print(f"Passed {filtered_count} out of {total_count} sequences")

def filter_directory(
    directory,
    output_dir,
    test_sequences: List[str],
    identity_threshold,
    score_threshold
):
    files = [ x for x in os.listdir(directory) if csv_loader.csv_extensions.search(x) ]
    if os.path.isdir(output_dir) and len(os.listdir(output_dir)) > 0:
        print(f"WARNING: output directory already exists and is non-empty")
    os.makedirs(output_dir, exist_ok=True)

    existing_files = set(x for x in os.listdir(output_dir) if csv_loader.csv_extensions.search(x))

    loaded_files = ((x, csv_loader.load_csv_file(os.path.join(directory, x))) for x in files if x not in existing_files)

    filtered = filter(loaded_files, test_sequences, identity_threshold)
    for file, seq in filtered:
        os.symlink(os.path.join(directory, file), os.path.join(output_dir, file))

def load_sequences(path: str):
    for file in os.listdir(path):
        if csv_loader.csv_extensions.search(file):
            _, chains = csv_loader.load_csv_file(os.path.join(path, file))
            for chain in chains.values():
                yield chain["sequence"]

if __name__ == "__main__":
    parser = argparse.ArgumentParser("Filter for test set")
    parser.add_argument("--testset", type=str, required=True)
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--output_dir", type=str)
    parser.add_argument("--identity_threshold", type=float, default=0.7)
    parser.add_argument("--score_threshold", type=float, default=0.5)

    args = parser.parse_args()

    test_sequences = [ s for s in load_sequences(args.testset) if len(s) >= 20 ]
    filter_directory(args.dataset, args.output_dir, test_sequences, args.identity_threshold, args.score_threshold)



