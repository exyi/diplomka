from typing import Dict, Tuple
import pandas as pd, numpy as np
import sys
import os
from collections import defaultdict

# maps nucleotide codes from the CSV to a single letter
nucleotide_mapping = defaultdict(lambda: 'X', {
    'A': 'A', 'C': 'C', 'G': 'G', 'T': 'T', 'U': 'U',
    'DA': 'A', 'DC': 'C', 'DG': 'G', 'DT': 'T', 'DU': 'U',

    '5MC': 'C', '2MG': 'G'
})

basic_nucleotides = [ "A", "T", "U", "G", "C", "X", " " ]
ntcs = [
    "NANT", # 2738272
    "AA00", # 2880801
    "AA08", # 1253456
    "AA01", # 290652
    "BB00", # 248604
    "AA04", # 196233
    "AB05", # 129296
    "AA12", # 100281
    "AA06", # 87112
    "AA03", # 67976
    "AA10", # 64719
    "AA05", # 61008
    "BB01", # 59435
    "BB04", # 55360
    "AA11", # 47708
    "AA09", # 42467
    "ZZ01", # 39024
    "BB07", # 37621
    "OP03", # 34898
    "OP11", # 33172
    "OP04", # 31279
    "AA02", # 25176
    "AB01", # 25119
    "BA05", # 24962
    "IC01", # 24854
    "OP12", # 24755
    "AA07", # 22712
    "OP15", # 22564
    "AB04", # 20022
    "BB10", # 16235
    "AA13", # 15698
    "AAS1", # 15057
    "OP08", # 13925
    "BB02", # 13591
    "OP24", # 13215
    "BA01", # 12341
    "BA08", # 11688
    "IC02", # 9702
    "BB16", # 9412
    "OP20", # 9392
    "OP10", # 9345
    "OP07", # 8782
    "OP09", # 8708
    "OPS1", # 8596
    "AB2S", # 8011
    "BB15", # 7969
    "OP13", # 7736
    "OP01", # 7657
    "OP21", # 7338
    "AB03", # 7028
    "OP23", # 6720
    "OP26", # 6712
    "OP18", # 6086
    "OP06", # 6070
    "OP29", # 6069
    "OP14", # 5621
    "BA13", # 5086
    "BA16", # 5065
    "OP02", # 4809
    "OP22", # 4800
    "OP31", # 4765
    "OP05", # 4695
    "BB03", # 4664
    "IC03", # 4161
    "IC04", # 3978
    "OP28", # 3876
    "BA10", # 3750
    "BBS1", # 3479
    "OP16", # 3256
    "BB12", # 3239
    "ZZ1S", # 2881
    "BB13", # 2582
    "OP25", # 2512
    "BB08", # 2484
    "OP17", # 2274
    "OP30", # 2111
    "BA17", # 2067
    "BB11", # 1910
    "BB17", # 1904
    "AB02", # 1793
    "BA09", # 1739
    "OP27", # 1582
    "ZZ2S", # 1453
    "IC07", # 1444
    "IC06", # 1325
    "BB14", # 1305
    "BB05", # 1105
    "ZZ02", # 1024
    "ZZS1", # 480
    "IC05", # 326
    "BB2S", # 326
    "ZZS2", # 147
    "OP19", # 108
    "BB20", # 81
    "OP1S", # 81
    "BB1S", # 80
    "AB1S", # 33
]
# mapping NTC-string -> integer index into the ntcs array
ntc_index = { word: index for index, word in enumerate(ntcs) }
# mapping nucleotide -> integer index into the basic_nucleotides array
nucleotide_index = { nucleotide: index for index, nucleotide in enumerate(basic_nucleotides) }

def encode_ntcs(ntcs: np.ndarray) -> np.ndarray:
    """
    Encodes a string array of NtCs into an array of integers.
    """
    return np.vectorize(lambda ntc: ntc_index[ntc])(ntcs)

def encode_nucleotides(nucleotides: str) -> np.ndarray:
    """
    Encodes a string of nucleotides into an array of integers.
    """
    return np.array([ nucleotide_index[nucleotide] for nucleotide in nucleotides ])

def map_nucleotide(nucleotide: str) -> str:
    """
    Maps a nucleotide from the CSVs to a single letter nucleotide.
    """
    if '.' in nucleotide:
        nucleotide = nucleotide.split('.')[0]
    return nucleotide_mapping[nucleotide]

def load_csv_file(file) -> Tuple[pd.DataFrame, Dict[Tuple[str, str], dict]]:
    """
    Loads the CSV into
    * pandas dataframe including all columns from the CSV, plus
        - pdbid: string
        - chain: string
        - nt_1: string - first nucleotide of the step
        - nt_2: string - second nucleotide
        - ntix_1: int - second nucleotide index in the chain
        - ntix_2: int - second nucleotide index
    * dictionary of chains: pdbid, chain -> dict
        - steps: subdataframe
        - sequence: string - single letter nucleotides
        - sequence_full: string array
        - is_dna: bool array
    """
    table = pd.read_csv(file, sep=',', header='infer')

    # split stepID into parts
    stepID: pd.Series[str] = table['step_ID']
    table['pdbid'] = [ s.split('_')[0] for s in stepID ]
    table['chain'] = [ s.split('_')[1] for s in stepID ]
    table['nt_1'] = [ s.split('_')[2] for s in stepID ]
    table['nt_2'] = [ s.split('_')[4] for s in stepID ]
    table['ntix_1'] = [ s.split('_')[3] for s in stepID ]
    table['ntix_2'] = [ s.split('_')[5] for s in stepID ]

    # create array columns for each column in table, grouped by pdbid and chain
    # identity = lambda x: x
    groups = dict(list(table.groupby(['pdbid', 'chain'])))
    def f(v):
        sequence = np.array([ v['nt_1'].array[0] ] + list(v['nt_2']))
        is_dna = np.array([ 'd' == s[0].lower() for s in sequence ], dtype=np.bool_)
        return {
            'steps': v,
            'sequence': ''.join([ map_nucleotide(n) for n in sequence ]),
            'sequence_full': sequence,
            'is_dna': is_dna,
        }

    groups2 = { k: f(v) for k, v in groups.items() }
    return table, groups2

def get_joined_arrays(chains: Dict[Tuple[str, str], dict]):
    """
    Joins the separated chains from load_csv_file into single arrays:
    * sequence: string - joined sequence separated by spaces
    * sequence_full: string array - joined sequence separated by spaces
    * is_dna: bool array - on separators it's False
    * NtC: ntc name array - on separators it's 'NANT'
    * nearest_NtC: ntc name array - on separators it's 'NANT'
    * CANA: name array - on separators it's 'NAN'
    * d1, e1, ... rmsd, confalA, ... - float arrays, on separators it's 0.0
    """
    def insert_spacers(arr, spacer):
        result = []
        for i in range(len(arr)):
            result.append(arr[i])
            if i < len(arr) - 1:
                result.append(spacer)
        return result

    cs = list(chains.values())
    sequence = " ".join([ c['sequence'] for c in cs ])
    sequence_full = np.concatenate(insert_spacers([ c['sequence_full'] for c in cs ], [ ' ' ]))
    is_dna = np.concatenate(insert_spacers([ c['is_dna'] for c in cs ], [ False ]))

    def join_arrays(arrays, zero_element, dtype=None):
        arrays = insert_spacers(arrays, np.array([ zero_element, zero_element ], dtype=dtype))
        return np.concatenate(arrays, axis=0, dtype=dtype)

    print(sequence)
    result = {
        "sequence": sequence,
        "sequence_full": sequence_full,
        "is_dna": is_dna,
        "NtC": join_arrays([ c['steps']['NtC'].to_numpy(np.str_) for c in cs ], "NANT", dtype=np.str_),
        "nearest_NtC": join_arrays([ c['steps']['nearest_NtC'].to_numpy(np.str_) for c in cs ], "NANT", dtype=np.str_),
        "CANA": join_arrays([ c['steps']['CANA'].to_numpy(np.str_) for c in cs ], "NAN"),
    }
    for k in [ "d1", "e1", "z1", "a2", "b2", "g2", "d2", "ch1", "ch2", "NN", "CC", "mu", "rmsd", "confalA", "confalG", "confalH" ]:
        result[k] = join_arrays([ c['steps'][k].to_numpy(np.float32) for c in cs ], 0.0, dtype=np.float32)
    return result


def save_parquet(structs, file):
    frame = pd.DataFrame(pd.concat(structs).groupby("name").apply(lambda x: x.to_dict(orient='records')), columns=['chains'])
    frame.to_parquet(file)

def load_csvs(path):
    """
    Dumps the csvs into parquet files for DuckDB querying
    """
    structs = []
    dirs = os.listdir(path)
    tmp_counter = 0
    for ix, file in enumerate(dirs):
        if file.endswith('.csv'):
            print(f'Loading {ix+1:<8}/{len(dirs)}: {file}                                 ', end='')
            structs.append(load_csv_file(os.path.join(path, file)))
            print('\r', end='')

        if len(structs) > 500:
            save_parquet(structs, f'./tmp_{tmp_counter}.parquet')
            tmp_counter += 1
            structs.clear()

    save_parquet(structs, f'./tmp_{tmp_counter}.parquet')

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Parse CSVs')
    parser.add_argument('--path', type=str, help='directory containing CSVs to load', required=True)
    args = parser.parse_args()
    load_csvs(args.path)
