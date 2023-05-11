#!/usr/bin/env python3
from typing import Any, Dict, List
import tensorflow as tf
import sys, os, math, json
import csv_loader



class NtcDatasetLoader:
    LETTERS = csv_loader.basic_nucleotides
    NTCS = csv_loader.ntcs
    def parse(self, example) -> Dict[str, tf.Tensor]:
        example = tf.io.parse_single_example(example, {
            "pdbid": tf.io.FixedLenFeature([], tf.string),
            "sequence": tf.io.FixedLenFeature([], tf.string),
            "sequence_full": tf.io.VarLenFeature(tf.string),
            "is_dna": tf.io.VarLenFeature(tf.int64),
            "NtC": tf.io.VarLenFeature(tf.string),
            "nearest_NtC": tf.io.VarLenFeature(tf.string),
            # "CANA": tf.io.VarLenFeature(tf.string),
        })
        example["sequence"] = tf.strings.unicode_split(example["sequence"], "UTF-8")
        if self.convert_to_numbers:
            example["sequence"] = self.letters_mapping(example["sequence"])
            # example["sequence_full"] = self.letters_mapping(example["sequence_full"])
            example["NtC"] = self.ntc_mapping(example["NtC"])
            example["nearest_NtC"] = self.ntc_mapping(example["nearest_NtC"])
            # example["CANA"] = self.cana_mapping(example["CANA"])

        example["is_dna"] = tf.sparse.to_dense(example["is_dna"])
        example["sequence_full"] = tf.sparse.to_dense(example["sequence_full"])
        example["NtC"] = tf.sparse.to_dense(example["NtC"])
        example["nearest_NtC"] = tf.sparse.to_dense(example["nearest_NtC"])
        # example["CANA"] = tf.sparse.to_dense(example["CANA"])
        # example["nearest_NtC"] = tf.sparse.to_dense(example["nearest_NtC"])

        return example
    
    @staticmethod
    def try_load_metadata(file_name: str) -> Dict[str, Any]:
        meta_file = file_name + ".meta.json"
        if os.path.exists(meta_file):
            with open(meta_file, "r") as f:
                return json.load(f)
        else:
            return dict()

    def __init__(self, files, convert_to_numbers = True) -> None:
        if isinstance(files, str):
            files = [ files ]

        for f in files:
            if not os.path.exists(f):
                raise FileNotFoundError(f)
            if not os.path.isfile(f):
                raise ValueError(f"{f} is not a file")

        self.convert_to_numbers = convert_to_numbers
        self.file_name = files
        self.metadata = [ self.try_load_metadata(f) for f in files ]
        self.letters_mapping = tf.keras.layers.StringLookup(vocabulary=self.LETTERS)
        self.ntc_mapping = tf.keras.layers.StringLookup(vocabulary=self.NTCS)
        self.cana_mapping = tf.keras.layers.StringLookup(vocabulary=self.NTCS)

        self.cardinality = sum([ m.get("count", float('nan')) for m in self.metadata ])
        if math.isnan(self.cardinality):
            self.cardinality = None

        if files[0].endswith(".gz"):
            compression_type = "GZIP"
        else:
            compression_type = None
        self.dataset = tf.data.TFRecordDataset(files, compression_type=compression_type).map(self.parse)

        if self.cardinality:
            self.dataset = self.dataset.apply(tf.data.experimental.assert_cardinality(self.cardinality))














def _bytes_feature(value):
  """Returns a bytes_list from a string / byte."""
  if isinstance(value, type(tf.constant(0))):
    value = value.numpy() # BytesList won't unpack a string from an EagerTensor.
  return tf.train.Feature(bytes_list=tf.train.BytesList(value=value))

def _float_feature(value):
  """Returns a float_list from a float / double."""
  return tf.train.Feature(float_list=tf.train.FloatList(value=value))

def _int64_feature(value):
  """Returns an int64_list from a bool / enum / int / uint."""
  return tf.train.Feature(int64_list=tf.train.Int64List(value=value))

def write_tfrecord_dataset(
    files: List[str],
    output_file: str,
    verbose = False):
    import csv_loader

    if output_file.endswith(".gz"):
        compression_type = "GZIP"
    else:
        compression_type = None
    count = 0
    max_len = 0
    with tf.io.TFRecordWriter(output_file, tf.io.TFRecordOptions(compression_type=compression_type)) as writer:
        for i, f in enumerate(files):
            count += 1
            df, chains = csv_loader.load_csv_file(f)
            joined = csv_loader.get_joined_arrays(chains)
            pdbid = list(chains.keys())[0][0]

            if max_len < len(joined['sequence']):
                max_len = len(joined['sequence'])

            if verbose:
                print(f"Processing {i+1:5d}/{len(files)}: {f} ({len(joined['sequence_full'])} nt)")
            features = {
                "pdbid": _bytes_feature([pdbid.encode('utf-8')]),
                "sequence": _bytes_feature([joined['sequence'].encode('utf-8')]),
                "sequence_full": _bytes_feature([x.encode('utf-8') for x in joined['sequence_full']]),
                "is_dna": _int64_feature(joined['is_dna'].astype('int64')),
                "NtC": _bytes_feature([x.encode('utf-8') for x in joined['NtC']]),
                "nearest_NtC": _bytes_feature([x.encode('utf-8') for x in joined['nearest_NtC']]),
                "CANA": _bytes_feature([x.encode('utf-8') for x in  joined['CANA']]),
                "rmsd": _float_feature(joined['rmsd']),
                "confalA": _float_feature(joined['confalA']),
                "confalG": _float_feature(joined['confalG']),
                "confalH": _float_feature(joined['confalH']),
            }
            example = tf.train.Example(features=tf.train.Features(feature=features))
            serialized = example.SerializeToString()
            writer.write(serialized)

    try:
        metadata = {
            "count": count,
            "max_len": max_len,
        }
        with open(output_file + ".meta.json", "w") as f:
            json.dump(metadata, f)
    except Exception as e:
        print(f"Error writing metadata: {e}")

if __name__ == "__main__":
    import argparse, csv_loader
    parser = argparse.ArgumentParser(description="""
        Convert CSV files to TFRecord
        
        Usage:
          * --input <directory> --output <file>
            Converts csv files to one TFRecord file
          * --testload <file>
            Prints first 10 items from a TFRecord file
          * --list <file>
            Prints list of items from a TFRecord file
            Format: PDBID    LENGTH    #CHAINS
    """)
    parser.add_argument('--testload', type=str, help='TFRecord file to load and print')
    parser.add_argument('--list', type=str, help='TFRecord file to load and print')
    parser.add_argument('--input', type=str, help='Input directory with CSV files')
    parser.add_argument('--output', type=str, help='Output TFRecord file')
    parser.add_argument('--verbose', action='store_true', help='Verbose output', default=False)
    args = parser.parse_args()

    if args.testload:
        loader = NtcDatasetLoader(args.testload)
        for i, example in enumerate(loader.dataset):
            print(example)
            if i > 10:
                break
    elif args.list:
        loader = NtcDatasetLoader(args.list)
        if args.verbose:
            print(f"# Cardinality = {loader.dataset.cardinality()}")
            print(f"# count =       {sum(1 for _ in loader.dataset)}")
            print(f"PDBID   LENGTH #CHAINS")
        for i, example in enumerate(loader.dataset):
            seq: str = bytes(example['sequence'].numpy()).decode('utf-8')
            seqs = seq.split(' ')
            print(f"{bytes(example['pdbid'].numpy()).decode('utf-8'):<8} {sum(len(x) for x in seqs):5d} {len(seqs):2d}")

    elif args.input and args.output:
        files = [os.path.join(args.input, f) for f in os.listdir(args.input) if csv_loader.csv_extensions.search(f)]
        write_tfrecord_dataset(files, args.output, args.verbose)
    
    else:
        print("Invalid parameter combination")
        parser.print_help()
        sys.exit(1)
