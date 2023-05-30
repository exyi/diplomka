#!/usr/bin/env python3
from typing import Any, Dict, List, Optional, Tuple
import tensorflow as tf
import sys, os, math, json, re
import numpy as np
import csv_loader



class NtcDatasetLoader:
    LETTERS = csv_loader.basic_nucleotides
    NTCS = csv_loader.ntcs
    letters_mapping = tf.keras.layers.StringLookup(vocabulary=csv_loader.basic_nucleotides, oov_token="X")
    ntc_mapping = tf.keras.layers.StringLookup(vocabulary=csv_loader.ntcs)
    cana_mapping = tf.keras.layers.StringLookup(vocabulary=csv_loader.CANAs[1:], oov_token="NAN")
    parsing_features = {
        "pdbid": tf.io.FixedLenFeature([], tf.string),
        "sequence": tf.io.FixedLenFeature([], tf.string),
        "sequence_full": tf.io.VarLenFeature(tf.string),
        "is_dna": tf.io.VarLenFeature(tf.int64),
        "NtC": tf.io.VarLenFeature(tf.string),
        "nearest_NtC": tf.io.VarLenFeature(tf.string),
        "CANA": tf.io.VarLenFeature(tf.string),
        "pairing_type": tf.io.VarLenFeature(tf.string),
        "pairing_is_canonical": tf.io.VarLenFeature(tf.int64),
        "pairing_nt1_ix": tf.io.VarLenFeature(tf.int64),
        "pairing_nt2_ix": tf.io.VarLenFeature(tf.int64),
        "dist_NN": tf.io.VarLenFeature(tf.float32),
    }

    def parse(self, example) -> Dict[str, tf.Tensor]:
        def to_dense(name, number_mapping = None):
            if name not in example:
                return
            if self.convert_to_numbers and number_mapping is not None:
                example[name] = number_mapping(example[name])
            example[name] = tf.sparse.to_dense(example[name])

        example = tf.io.parse_single_example(example, self.parsing_features)
        example["sequence"] = tf.strings.unicode_split(example["sequence"], "UTF-8")
        if self.convert_to_numbers:
            example["sequence"] = self.letters_mapping(example["sequence"])
            # example["sequence_full"] = self.letters_mapping(example["sequence_full"])
            # example["NtC"] = self.ntc_mapping(example["NtC"])
            # example["nearest_NtC"] = self.ntc_mapping(example["nearest_NtC"])

        to_dense('NtC', self.ntc_mapping)
        to_dense('nearest_NtC', self.ntc_mapping)
        to_dense('is_dna')
        to_dense('sequence_full')
        to_dense('CANA', self.cana_mapping)
        to_dense('pairing_type') # TODO mapping to numbers
        to_dense('pairing_is_canonical')
        to_dense('pairing_nt1_ix')
        to_dense('pairing_nt2_ix')
        to_dense('dist_NN')
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

    def get_data(self, max_len = None, trim_prob: float = 0, shuffle = None, batch = None, shuffle_chains = True, max_chains = None, pairing_seq = True, pairing_matrix = False):
        data = self.dataset

        # def remap_chains(tensor, separators, )

        def mapping(x):
            x_sequence = x["sequence"]

            length = tf.shape(x["sequence"])[0]
            s_slice: Any = 0
            separator = self.letters_mapping(" ") if self.convert_to_numbers else " "
            separator_idx = tf.where(tf.equal(x["sequence"], separator))

            pairs_with = None
            if pairing_seq:
                pairs_with = tf.zeros(shape=[length], dtype=tf.int32)
                pairs_with = tf.tensor_scatter_nd_update(pairs_with, tf.expand_dims(x["pairing_nt1_ix"], axis=1), tf.cast(x["pairing_nt2_ix"], tf.int32))

                tf.print(pairs_with, x["pairing_nt1_ix"], x["pairing_nt2_ix"])

            if len(separator_idx) > 0 and (shuffle_chains or (max_chains and max_chains > len(separator_idx))):
                # TODO: implement chain shuffling
                pass

            # sequences = tf.strings.split(tf.shape(x["sequence"]), sep=separator)
            if trim_prob > 0 and tf.random.uniform(shape=[], minval=0, maxval=1) < trim_prob:
                new_len = tf.random.uniform(shape=[], minval=0, maxval=(max_len or length), dtype=tf.int32)
                s_slice = tf.random.uniform(shape=[], minval=0, maxval=length - new_len, dtype=tf.int32)
                length = new_len

            elif max_len and length > max_len:
                s_slice = tf.random.uniform(shape=[], minval=0, maxval=length - max_len, dtype=tf.int32)
                length = max_len


            return {
                "pdbid": x["pdbid"],
                "sequence": x["sequence"][s_slice:s_slice+length],
                "sequence_full": x["sequence_full"][s_slice:s_slice+length],
                "is_dna": x["is_dna"][s_slice:s_slice+length],
                "NtC": x["NtC"][s_slice:s_slice+length-1],
                "nearest_NtC": x["nearest_NtC"][s_slice:s_slice+length-1],
                "pairs_with": pairs_with[s_slice:s_slice+length],
                # "CANA": x["CANA"][s_slice:s_slice+length-1],
            }


        data = data.map(mapping)
        def filter_dict(d: Dict[str, Any], keys: List[str]) -> Dict[str, Any]:
            return { k: v for k, v in d.items() if k in keys }
        data = data.map(lambda x: (
            filter_dict(x, ["is_dna", "sequence", "pdbid"]),
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

def sequence_np(x):
    if isinstance(x, str):
        return np.array(list(x), dtype='U1')
    else:
        return np.array(list(x['sequence']), dtype='U1')
    
def filter_models(chains):
    if any([ pdb.endswith("-m1") for (pdb, _, _) in chains.keys() ]):
        return {
            (pdb[0:-3], chain, chainslice): v for (pdb, chain, chainslice), v in chains.items()
                if pdb.endswith("-m1")
        }
    else:
        return chains

def create_chain_mapping(chains) -> Dict[str, Tuple[int, int]]:
    chain_index: Dict[str, Tuple[int, int]] = dict()
    x = 0
    for k, v in chains.items():
        if isinstance(k, tuple):
            k = k[1]
        if k in chain_index:
            if chain_index[k][1] == x:
                chain_index[k] = (chain_index[k][0], x + len(v['sequence']))
            else:
                raise ValueError(f"Chains {k} are not contiguous: {chain_index[k]} and {x}")
        else:
            chain_index[k] = (x, x + len(v['sequence']))
        x += len(v['sequence'])
    return chain_index

def map_chain_index(chain_index: Dict[str, Tuple[int, int]], chain: str, index: int) -> int:
    """
    Maps index in a chain to the index in the full concatenated sequence
    """
    assert index >= 0
    if chain not in chain_index:
        raise KeyError(f"Chain {chain} not found in chain index ({chain_index})")
    main_index = chain_index[chain][0] + index
    if main_index >= chain_index[chain][1]:
        raise IndexError(f"Index {index} in chain {chain} is out of bounds ({chain_index[chain]})")
        main_index = chain_index[chain][1] - 1
    return main_index

def map_index_to_chain(chain_index, index) -> tuple[str, int]:
    """
    Maps index in the full concatenated sequence to the chain and index in the chain
    """
    assert index >= 0, f"{index}"
    for chain, (start, end) in chain_index.items():
        if index >= start and index < end:
            return chain, index - start
    raise IndexError(f"Index {index} is not in any chain ({chain_index})")

def get_duplicates(xs, key=lambda x: x):
    seen = set()
    for x in xs:
        if key(x) in seen:
            yield x
        else:
            seen.add(key(x))

def write_tfrecord_dataset(
    files: List[str],
    output_file: str,
    pairing_files: Optional[Dict[str, str]] = None,
    verbose = False,
    dna_handling = None,
    angles = False
):
    if output_file.endswith(".gz"):
        compression_type = "GZIP"
    else:
        compression_type = None
    count = 0
    max_len = 0
    with tf.io.TFRecordWriter(output_file, tf.io.TFRecordOptions(compression_type=compression_type)) as writer:
        for i, f in enumerate(files):
            try:
                df, chains = csv_loader.load_csv_file(f)
                del df
                chains = filter_models(chains)
                orig_chains = chains
                if dna_handling == "ignore":
                    chains = { k: v for k, v in chains.items() if np.mean(v["is_dna"] & (sequence_np(v) != 'X') & (sequence_np(v) != ' ')) < 0.1 }

                if len(chains) == 0:
                    print("Skipping (all chains filtered out)", f)
                    continue

                joined = csv_loader.get_joined_arrays(chains)

                # print("Chains: ", { k: v['sequence'] for k, v in chains.items() })
                # print("Joined: ", joined['sequence'])

                chain_mapping = create_chain_mapping(chains)
                pdbid = list(chains.keys())[0][0]

                assert len(joined['chain_names']) == len(joined['indices'])
                chain_index_pairs = list(zip(joined['chain_names'], joined['indices']))
                index_index = { (chain, pdb_index): py_index for py_index, (chain, pdb_index) in enumerate(chain_index_pairs) if pdb_index != '' }
                if len(index_index) != np.sum(joined['indices'] != ''):
                    raise Exception(f"Duplicate indices in {pdbid}:\n{list(get_duplicates(chain_index_pairs))}\n{np.array(chain_index_pairs)}")

                if dna_handling == "ignorepure":
                    # print(sequence_np(joined))
                    # print(joined['is_dna'])
                    # print( (sequence_np(joined) == 'X') | (sequence_np(joined) == ' '))
                    if np.mean(joined['is_dna'] | (sequence_np(joined) == 'X') | (sequence_np(joined) == ' ')) > 0.8:
                        print("Skipping (no RNA)", f)
                        continue

                pairing = None
                if pairing_files:
                    if pdbid in pairing_files:
                        pairing = csv_loader.read_fr3d_basepairing(pairing_files[pdbid], pdbid, filter_model=1, filter_chains=set(chain_mapping.keys()))
                    else:
                        print(f"WARNING: No pairing for {pdbid}.")

                if max_len < len(joined['sequence']):
                    max_len = len(joined['sequence'])

                if verbose:
                    bp_print = f", {len(pairing['pairing'])} bp" if pairing else ""
                    print(f"Processing {i+1:5d}/{len(files)}: {f} ({len(joined['sequence_full']): 5} nt{bp_print}, {len(chains)}/{len(orig_chains)} chains)")

                features = {
                    "pdbid": _bytes_feature([pdbid.encode('utf-8')]),
                    "chains": _bytes_feature(chain_mapping.keys()),
                    "chain_index": _int64_feature([ start for start, end in chain_mapping.values() ]),
                    "sequence": _bytes_feature([joined['sequence'].encode('utf-8')]),
                    "sequence_full": _bytes_feature(joined['sequence_full']),
                    "is_dna": _int64_feature(joined['is_dna'].astype('int64')),
                    "NtC": _bytes_feature(joined['NtC']),
                    "nearest_NtC": _bytes_feature(joined['nearest_NtC']),
                    "CANA": _bytes_feature(joined['CANA']),
                }
                if angles:
                    features["rmsd"] = _float_feature(joined['rmsd'])
                    features["confalA"] = _float_feature(joined['confalA'])
                    features["confalG"] = _float_feature(joined['confalG'])
                    features["confalH"] = _float_feature(joined['confalH'])
                    features["angle_d1"] = _float_feature(joined['d1'])
                    features["angle_e1"] = _float_feature(joined['e1'])
                    features["angle_z1"] = _float_feature(joined['z1'])
                    features["angle_a2"] = _float_feature(joined['a2'])
                    features["angle_b2"] = _float_feature(joined['b2'])
                    features["angle_g2"] = _float_feature(joined['g2'])
                    features["angle_d2"] = _float_feature(joined['d2'])
                    features["angle_ch1"] = _float_feature(joined['ch1'])
                    features["angle_ch2"] = _float_feature(joined['ch2'])
                    features["angle_mu"] = _float_feature(joined['mu'])
                    features["dist_NN"] = _float_feature(joined['NN'])
                    features["dist_CC"] = _float_feature(joined['CC'])

                if pairing is not None:
                    nt1_ix = np.array([
                        index_index.get((chain, ix), -1)
                        for chain, ix in zip(pairing['nt1_chain'], pairing['nt1_ix'])
                    ])
                    nt2_ix = np.array([
                        index_index.get((chain, ix), -1)
                        for chain, ix in zip(pairing['nt2_chain'], pairing['nt2_ix'])
                    ])
                    valid_indices = np.logical_and(nt1_ix >= 0, nt2_ix >= 0)
                    # CSVs are missing some weird nucleotides, so we allow then to be missing
                    allowed_invalid_indices = np.array(
                        [ not (a in csv_loader.basic_nucleotides and b in csv_loader.basic_nucleotides) for a, b in zip(pairing['nt1_base'], pairing['nt2_base']) ],
                        dtype=np.bool_)
                    # the CSV are also missing nucleotide if the index is == 0 (wonder why :D)
                    allowed_invalid_indices = np.logical_or(allowed_invalid_indices, np.logical_or(pairing['nt1_ix'] == '0', pairing['nt2_ix'] == '0'))
                    if np.any(~valid_indices & ~allowed_invalid_indices):
                        print("inddices = ", joined['indices'])
                        print("chain_names = ", joined['chain_names'])
                        print("index_index = ", index_index)
                        print("invalid_indices = ", pairing['nt1_ix'][~valid_indices], pairing['nt2_ix'][~valid_indices])
                        print("invalid bases = ", pairing['nt1_base'][~valid_indices], pairing['nt2_base'][~valid_indices])
                        invalid_pairs = [
                            f"{chain1}-{ix1} {chain2}-{ix2}"
                            for chain1, ix1, chain2, ix2 in zip(pairing['nt1_chain'][~valid_indices], pairing['nt1_ix'][~valid_indices], pairing['nt2_chain'][~valid_indices], pairing['nt2_ix'][~valid_indices])
                        ]
                        raise Exception(f"Invalid indices in {pdbid}: [{', '.join(invalid_pairs)}], allowed: {allowed_invalid_indices[~valid_indices]}")
                    features["pairing_type"] = _bytes_feature(pairing['pairing'][valid_indices])
                    features["pairing_nt1_ix_orig"] = _bytes_feature(pairing['nt1_ix'][valid_indices])
                    features["pairing_nt1_ix"] = _int64_feature(nt1_ix[valid_indices])
                    features["pairing_nt2_ix_orig"] = _bytes_feature(pairing['nt2_ix'][valid_indices])
                    features["pairing_nt2_ix"] = _int64_feature(nt2_ix[valid_indices])
                    features["pairing_nt1_chain"] = _bytes_feature(pairing['nt1_chain'][valid_indices])
                    features["pairing_nt2_chain"] = _bytes_feature(pairing['nt2_chain'][valid_indices])
                    basepairs = np.array([
                        csv_loader.map_nucleotide(a) + csv_loader.map_nucleotide(b)
                        for a, b in zip(pairing['nt1_base'][valid_indices], pairing['nt2_base'][valid_indices])
                    ])
                    features["pairing_basepair"] = _bytes_feature([
                        csv_loader.map_nucleotide(a) + csv_loader.map_nucleotide(b)
                        for a, b in zip(pairing['nt1_base'][valid_indices], pairing['nt2_base'][valid_indices])
                    ])
                    features["pairing_is_canonical"] = _int64_feature(np.array([
                        t == 'cwW' and bp in ['GC', 'CG', 'AU', 'UA', 'AT', 'TA']
                        for bp, t in zip(basepairs, pairing['pairing'][valid_indices])
                    ]).astype('int64'))

                example = tf.train.Example(features=tf.train.Features(feature=features))
                serialized = example.SerializeToString()
                writer.write(serialized)
            except Exception as e:
                print(f"Error processing file {f}: {e}")
                raise e
            count += 1
    try:
        metadata = {
            "count": count,
            "max_len": max_len,
            "has_pairing": pairing_files is not None,
            "dna_handling": dna_handling or "keep",
            "has_angles": angles
        }
        with open(output_file + ".meta.json", "w") as f:
            json.dump(metadata, f)
    except Exception as e:
        print(f"Error writing metadata: {e}")


def find_pairing_files(directory):
    if directory is None:
        return None

    result = dict()
    for f in os.listdir(directory):
        if re.search(r"^[a-zA-Z0-9]{4}_basepair", f):
            pdbid = f.split("_")[0]
            if pdbid in result:
                print("WARNING: duplicate basepairing PDBID", pdbid, ":", f, "and", result[pdbid])
            result[pdbid] = os.path.join(directory, f)
    return result


if __name__ == "__main__":
    import argparse
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
    parser.add_argument('--pairing_input', type=str, help='Input directory with FR3D pairing files')
    parser.add_argument('--torsion_angles', action='store_true', help='Include torsion angles', default=False)
    parser.add_argument('--output', type=str, help='Output TFRecord file')
    parser.add_argument('--verbose', action='store_true', help='Verbose output', default=False)
    parser.add_argument('--dna_handling', type=str, help="How to handle DNA chains. Ignore removes any chains with at least 10 percent DNA. Ignorepure removes the structure if it is more than 80 percent DNA", choices=["ignore", "ignorepure", "keep"], default="keep")
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
            print(f"PDBID   LENGTH #CHAINS PAIRS SEQUENCE")
        for i, example in enumerate(loader.dataset):
            seq: str = "".join(csv_loader.basic_nucleotides[example['sequence'].numpy() - 1])
            seqs = seq.split(' ')
            print_seq = (" " + seq) if args.verbose else ""
            pairs_num = str(len(example["pairing_type"].numpy())) if 'pairing_type' in example else "-"
            print(f"{bytes(example['pdbid'].numpy()).decode('utf-8'):<8} {sum(len(x) for x in seqs):8d} {len(seqs):2d} {pairs_num:>7}{print_seq}")

    elif args.input and args.output:
        files = [os.path.join(args.input, f) for f in os.listdir(args.input) if csv_loader.csv_extensions.search(f)]
        write_tfrecord_dataset(files, args.output,
            pairing_files=find_pairing_files(args.pairing_input),
            verbose=args.verbose, dna_handling=args.dna_handling, angles=args.torsion_angles)
    
    else:
        print("Invalid parameter combination")
        parser.print_help()
        sys.exit(1)
