#!/usr/bin/env python3
import Bio.SeqIO
from typing import Any, Dict, List, Optional, Set
import tensorflow as tf
import sys, os, math, json, re, itertools
import numpy as np


def parse_fasta_core(file):
    for record in Bio.SeqIO.parse(file, "fasta"):
        yield str(record.id), str(record.description), str(record.seq)

def parse_fasta(path: str):
    if path.endswith(".gz"):
        import gzip
        with gzip.open(path, "rt") as f:
            yield from parse_fasta_core(f)
    elif path.endswith(".zst"):
        import zstandard
        with zstandard.open(path, "rt") as f:
            yield from parse_fasta_core(f)
    else:
        with open(path, "rt") as f:
            yield from parse_fasta_core(f)


def _bytes_feature(value):
  """Returns a bytes_list from a string / byte."""
  if isinstance(value, type(tf.constant(0))):
    value = value.numpy() # BytesList won't unpack a string from an EagerTensor.

  if isinstance(value, np.ndarray):
      if value.dtype.kind == 'U':
          value = np.char.encode(value, 'utf-8')
  else:
    value = [ v.encode('utf-8') if isinstance(v, str) else v for v in value ]
  return tf.train.Feature(bytes_list=tf.train.BytesList(value=value))

def _float_feature(value):
  """Returns a float_list from a float / double."""
  return tf.train.Feature(float_list=tf.train.FloatList(value=value))

def _int64_feature(value):
  """Returns an int64_list from a bool / enum / int / uint."""
  return tf.train.Feature(int64_list=tf.train.Int64List(value=value))

def write_sharded_dataset(
    main_file: str,
    num_shards: int,
    data_iterator,
    write_fn,
    get_stats_fn = None
):
    directory = os.path.dirname(main_file)
    main_file_base = os.path.basename(main_file).split(".", 1)[0]

    shard_files = [ f"{main_file_base}.{str(i).zfill(math.ceil(math.log10(num_shards)))}.tfrecord.gz" for i in range(num_shards) ]
    shard_writers = [ tf.io.TFRecordWriter(os.path.join(directory, f), tf.io.TFRecordOptions(compression_type="GZIP")) for f in shard_files ]
    shard_counts = [ 0 for _ in shard_files ]
    count_total = 0
    def write_sample(example):
        nonlocal count_total
        shard = count_total % num_shards
        shard_writers[shard].write(example)
        shard_counts[shard] += 1
        count_total += 1
    try:
        for i, data in enumerate(data_iterator):
            try:
                write_fn(write_sample, i, data)
            except Exception as e:
                print(f"Error while writing #{i}: ", data)
                raise e
    finally:
        for w in shard_writers:
            w.close()

    result = {
        "shards": [ { "file": f, "count": c } for f, c in zip(shard_files, shard_counts) ],
        "stats": get_stats_fn() if get_stats_fn else None,
        "count": count_total
    }
    with open(main_file, "w") as f:
        json.dump(result, f, indent=4)

def write_tfrecord_dataset(
    inputs: List[str],
    output_file: str,
    num_shards: int = 64,
    exclude_types: Optional[Set[str]] = None,
    only_types: Optional[Set[str]] = None,
    verbose = False
):
    max_len = 0
    rna_type_counts = {}
    length_histogram = []

    def mark_histogram(length):
        length = length // 100
        while length >= len(length_histogram):
            length_histogram.append(0)
        length_histogram[length] += 1

    def get_stats():
        return {
            "rna_type_counts": rna_type_counts,
            "length_histogram": length_histogram,
            "max_len": max_len
        }

    def write_sample(write, i, data):
        if verbose and i % 1000 == 0:
            print(f"Writing #{i} {data[0]}")
        nonlocal max_len
        id, fasta_label, seq = data
        _, rna_type, comment = fasta_label.split(" ", 2)

        if exclude_types and rna_type in exclude_types:
            return
        if only_types and rna_type not in only_types:
            return

        if rna_type not in rna_type_counts:
            rna_type_counts[rna_type] = 0
        rna_type_counts[rna_type] += 1

        mark_histogram(len(seq))

        max_len = max(max_len, len(seq))

        features = {
            "id": _bytes_feature([id.encode('utf-8')]),
            "type": _bytes_feature([rna_type.encode('utf-8')]),
            "comment": _bytes_feature([comment.encode('utf-8')]),
            "sequence": _bytes_feature([seq.encode('utf-8')]),
        }
        example = tf.train.Example(features=tf.train.Features(feature=features))
        write(example.SerializeToString())

    data_iterator = itertools.chain.from_iterable(parse_fasta(f) for f in inputs)

    write_sharded_dataset(
        output_file,
        num_shards,
        data_iterator,
        write_sample,
        get_stats
    )

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="""
        Convert rnacentral fasta file(s) to TFRecord
        
        Usage:
          * --input <directory> --output <file>
            Converts csv files to one TFRecord file
          TODO * --testload <file>
            Prints first 10 items from a TFRecord file
          TODO * --list <file>
            Prints list of items from a TFRecord file
            Format: PDBID    LENGTH    #CHAINS
    """)
    parser.add_argument('--testload', type=str, help='TFRecord file to load and print')
    parser.add_argument('--list', type=str, help='TFRecord file to load and print')
    parser.add_argument('--output', type=str, help='Output file')
    parser.add_argument('--num_shards', type=int, help='Number of TFRecord shards', default=64)
    parser.add_argument('--verbose', action='store_true', help='Verbose output', default=False)
    parser.add_argument('--exclude_types', type=str, nargs="*", help='Exclude RNA types')
    parser.add_argument('--only_types', type=str, nargs="*", help='Include only specified RNA types')
    parser.add_argument('--input', type=str, nargs="*", help='Input directory with CSV files')
    args = parser.parse_args()

    if args.testload:
        raise Exception("Not implemented")
        # loader = NtcDatasetLoader(args.testload)
        # for i, example in enumerate(loader.dataset):
        #     print(example)
        #     if i > 10:
        #         break
    elif args.list:
        raise Exception("Not implemented")

    elif args.input:
        output = args.output
        if not output:
            if args.only_types:
                label = "_" + "+".join(args.only_types)
            elif args.exclude_types:
                label = "_" + "+".join([ "no" + x for x in args.exclude_types])
            else:
                label = ""
            output = os.path.join(os.path.dirname(args.input[0]), f"rnacentral{label}.json")
        write_tfrecord_dataset(args.input, output, args.num_shards, args.exclude_types, args.only_types, args.verbose)
    
    else:
        print("Invalid parameter combination")
        parser.print_help()
        sys.exit(1)
