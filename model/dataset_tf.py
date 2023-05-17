#!/usr/bin/env python3
from typing import Any, Dict, List
import tensorflow as tf
import sys, os, math, json
import numpy as np
import csv_loader



class NtcDatasetLoader:
    LETTERS = csv_loader.basic_nucleotides
    NTCS = csv_loader.ntcs
    letters_mapping = tf.keras.layers.StringLookup(vocabulary=csv_loader.basic_nucleotides, oov_token="X")
    ntc_mapping = tf.keras.layers.StringLookup(vocabulary=csv_loader.ntcs)
    cana_mapping = tf.keras.layers.StringLookup(vocabulary=csv_loader.ntcs)

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

    def get_data(self, max_len = None, trim_prob: float = 0, shuffle = None, batch = None):
        data = self.dataset

        def mapping(x):
            length = tf.shape(x["sequence"])[0]
            s_slice = 0
            separator = self.letters_mapping(" ") if self.convert_to_numbers else " "
            # sequences = tf.strings.split(tf.shape(x["sequence"]), sep=separator)
            if trim_prob > 0 and tf.random.uniform(shape=[], minval=0, maxval=1) < trim_prob:
                new_len = tf.random.uniform(shape=[], minval=0, maxval=(max_len or length), dtype=tf.int32)
                s_slice = tf.random.uniform(shape=[], minval=0, maxval=length - new_len, dtype=tf.int32)
                length = new_len

            elif max_len and length > max_len:
                s_slice = tf.random.uniform(shape=[], minval=0, maxval=length - max_len, dtype=tf.int32)
                length = max_len
            
            return {
                "sequence": x["sequence"][s_slice:s_slice+length],
                "sequence_full": x["sequence_full"][s_slice:s_slice+length],
                "is_dna": x["is_dna"][s_slice:s_slice+length],
                "NtC": x["NtC"][s_slice:s_slice+length-1],
                "nearest_NtC": x["nearest_NtC"][s_slice:s_slice+length-1],
                # "CANA": x["CANA"][s_slice:s_slice+length-1],
            }


        data = data.map(mapping)
        def filter_dict(d: Dict[str, Any], keys: List[str]) -> Dict[str, Any]:
            return { k: v for k, v in d.items() if k in keys }
        data = data.map(lambda x: (
            filter_dict(x, ["is_dna", "sequence"]),
            filter_dict(x, ["NtC"]),
        ))

        if shuffle:
            data = data.shuffle(shuffle)

        if batch:
            data = data.ragged_batch(batch)
        return data.prefetch(tf.data.AUTOTUNE)















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

def sequence_np(x):
    if isinstance(x, str):
        return np.array(list(x), dtype='U1')
    else:
        return np.array(list(x['sequence']), dtype='U1')
def write_tfrecord_dataset(
    files: List[str],
    output_file: str,
    verbose = False,
    dna_handling = None):
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
            orig_chains = chains
            if dna_handling == "ignore":
                chains = { k: v for k, v in chains.items() if np.mean(v["is_dna"] & (sequence_np(v) != 'X') & (sequence_np(v) != ' ')) < 0.1 }

            if len(chains) == 0:
                print("Skipping (all chains filtered out)", f)
                continue

            joined = csv_loader.get_joined_arrays(chains)

            if dna_handling == "ignorepure":
                # print(sequence_np(joined))
                # print(joined['is_dna'])
                # print( (sequence_np(joined) == 'X') | (sequence_np(joined) == ' '))
                if np.mean(joined['is_dna'] | (sequence_np(joined) == 'X') | (sequence_np(joined) == ' ')) > 0.8:
                    print("Skipping (no RNA)", f)
                    continue
            pdbid = list(chains.keys())[0][0]

            if max_len < len(joined['sequence']):
                max_len = len(joined['sequence'])

            if verbose:
                print(f"Processing {i+1:5d}/{len(files)}: {f} ({len(joined['sequence_full']): 5} nt, {len(chains)}/{len(orig_chains)} chains)")
            features = {
                "pdbid": _bytes_feature([pdbid.encode('utf-8')]),
                "sequence": _bytes_feature([joined['sequence'].encode('utf-8')]),
                "sequence_full": _bytes_feature([x.encode('utf-8') for x in joined['sequence_full']]),
                "is_dna": _int64_feature(joined['is_dna'].astype('int64')),
                "NtC": _bytes_feature([x.encode('utf-8') for x in joined['NtC']]),
                "nearest_NtC": _bytes_feature([x.encode('utf-8') for x in joined['nearest_NtC']]),
                "CANA": _bytes_feature([x.encode('utf-8') for x in  joined['CANA']]),
                # "rmsd": _float_feature(joined['rmsd']),
                # "confalA": _float_feature(joined['confalA']),
                # "confalG": _float_feature(joined['confalG']),
                # "confalH": _float_feature(joined['confalH']),
                # "angle_d1": _float_feature(joined['d1']),
                # "angle_e1": _float_feature(joined['e1']),
                # "angle_z1": _float_feature(joined['z1']),
                # "angle_a2": _float_feature(joined['a2']),
                # "angle_b2": _float_feature(joined['b2']),
                # "angle_g2": _float_feature(joined['g2']),
                # "angle_d2": _float_feature(joined['d2']),
                # "angle_ch1": _float_feature(joined['ch1']),
                # "angle_ch2": _float_feature(joined['ch2']),
                # "angle_mu": _float_feature(joined['mu']),
                # "dist_NN": _float_feature(joined['NN']),
                # "dist_CC": _float_feature(joined['CC']),
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
    parser.add_argument('--dna_handling', type=str, help="How to handle DNA chains. Ignore removes any chains with at least 10% DNA. Ignorepure removes the structure if it is more than 80% DNA", choices=["ignore", "ignorepure", "keep"], default="keep")
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
            print(f"PDBID   LENGTH #CHAINS  SEQUENCE")
        for i, example in enumerate(loader.dataset):
            seq: str = "".join(csv_loader.basic_nucleotides[example['sequence'].numpy() - 1])
            print(seq)
            seqs = seq.split(' ')
            print_seq = (" " + seq) if args.verbose else ""
            print(f"{bytes(example['pdbid'].numpy()).decode('utf-8'):<8} {sum(len(x) for x in seqs):5d} {len(seqs):2d}{print_seq}")

    elif args.input and args.output:
        files = [os.path.join(args.input, f) for f in os.listdir(args.input) if csv_loader.csv_extensions.search(f)]
        write_tfrecord_dataset(files, args.output, args.verbose, args.dna_handling)
    
    else:
        print("Invalid parameter combination")
        parser.print_help()
        sys.exit(1)
