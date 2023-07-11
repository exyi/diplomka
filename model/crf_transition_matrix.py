#!/usr/bin/env python3
from typing import List
import numpy as np
import os, sys, math

# def _hist_overlap(a, b):
#     result = np.zeros(len(a)//2, np.bool_)
#     # bT = np.expand_dims(b, axis=1)
#     end = len(a) - 1
#     for i in range(len(result)):
#         result[i] = a[i] < b[end-i] and a[end-i] > b[i]
#     return result

def _hist_overlap(a, b) -> float:
    b_rev = b[::-1]
    half = len(a)//2
    odd = len(a) % 2
    return np.sum(np.logical_and(a[0:half] < b_rev[0:half], (a[half+odd:] > b_rev[half+odd:])[::-1])) / half

def overlap_mapping(x):
    return min(1, max(0, x * 2))

def get_transition_matrix(ntc_list: List[str]):
    import pandas as pd
    df = pd.read_parquet(os.path.join(os.path.dirname(__file__), "../ntc_quantiles.parquet"))
    df.set_index("NtC", inplace=True)

    matrix = np.ones((len(ntc_list), len(ntc_list)))
    for i in range(len(ntc_list)):
        for j in range(len(ntc_list)):
            matrix[i, j] *= overlap_mapping(_hist_overlap(df.d2_hist[ntc_list[i]], df.d1_hist[ntc_list[j]]))
            matrix[i, j] *= overlap_mapping(0.1 + _hist_overlap(df.ch2_hist[ntc_list[i]], df.ch1_hist[ntc_list[j]]))
    # nant_i = list(ntc_list).index("NANT")
    # matrix[nant_i, :] = 1
    # matrix[:, nant_i] = 1
    return matrix

def load_saved_matrix(ntcs: List[str] | np.ndarray, file = os.path.join(os.path.dirname(__file__), "../data/crf_ntc_matrix.npz")):
    data = np.load(file)
    m = data["matrix"]
    if ntcs is not None:
        mntcs = data["ntcs"]
        ntcs = np.array(ntcs)
        index = { ntc: i for i, ntc in enumerate(mntcs) }
        permutation = np.array([ index.get(ntc, index["NANT"]) for ntc in ntcs ])
        m = m[permutation, :][:, permutation]
    return m

if __name__ == "__main__":
    import argparse
    import csv_loader
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="transition_matrix.csv")
    args = parser.parse_args()
    ntcs = list(sorted(csv_loader.ntcs))
    matrix = get_transition_matrix(ntcs)
    np.savetxt(args.output, matrix, delimiter=",", header=",".join(ntcs), comments="")
    np.savez_compressed(args.output + ".npy", matrix=matrix, ntcs=ntcs)

