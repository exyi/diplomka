from typing import Any, Dict, List, Optional, Set, TextIO, Tuple, Union
import pandas as pd, numpy as np
import sys, os, re
from collections import defaultdict

from utils import retry_on_error

csv_extensions = re.compile(r"\.csv(\.(zst|gz|xz|bz2))?$")

# maps nucleotide codes from the CSV to a single letter
nucleotide_mapping = defaultdict(lambda: 'X', {
    'A': 'A', 'C': 'C', 'G': 'G', 'T': 'T', 'U': 'U',
    'DA': 'A', 'DC': 'C', 'DG': 'G', 'DT': 'T', 'DU': 'U',

    '5MC': 'C', '2MG': 'G'
})

basic_nucleotides = np.array([ "A", "T", "U", "G", "C", "X", " " ])
ntc_frequencies = {
    "NANT": 2738272,
    "AA00": 2880801,
    "AA08": 1253456,
    "AA01": 290652,
    "BB00": 248604,
    "AA04": 196233,
    "AB05": 129296,
    "AA12": 100281,
    "AA06": 87112,
    "AA03": 67976,
    "AA10": 64719,
    "AA05": 61008,
    "BB01": 59435,
    "BB04": 55360,
    "AA11": 47708,
    "AA09": 42467,
    "ZZ01": 39024,
    "BB07": 37621,
    "OP03": 34898,
    "OP11": 33172,
    "OP04": 31279,
    "AA02": 25176,
    "AB01": 25119,
    "BA05": 24962,
    "IC01": 24854,
    "OP12": 24755,
    "AA07": 22712,
    "OP15": 22564,
    "AB04": 20022,
    "BB10": 16235,
    "AA13": 15698,
    "AAS1": 15057,
    "OP08": 13925,
    "BB02": 13591,
    "OP24": 13215,
    "BA01": 12341,
    "BA08": 11688,
    "IC02": 9702,
    "BB16": 9412,
    "OP20": 9392,
    "OP10": 9345,
    "OP07": 8782,
    "OP09": 8708,
    "OPS1": 8596,
    "AB2S": 8011,
    "BB15": 7969,
    "OP13": 7736,
    "OP01": 7657,
    "OP21": 7338,
    "AB03": 7028,
    "OP23": 6720,
    "OP26": 6712,
    "OP18": 6086,
    "OP06": 6070,
    "OP29": 6069,
    "OP14": 5621,
    "BA13": 5086,
    "BA16": 5065,
    "OP02": 4809,
    "OP22": 4800,
    "OP31": 4765,
    "OP05": 4695,
    "BB03": 4664,
    "IC03": 4161,
    "IC04": 3978,
    "OP28": 3876,
    "BA10": 3750,
    "BBS1": 3479,
    "OP16": 3256,
    "BB12": 3239,
    "ZZ1S": 2881,
    "BB13": 2582,
    "OP25": 2512,
    "BB08": 2484,
    "OP17": 2274,
    "OP30": 2111,
    "BA17": 2067,
    "BB11": 1910,
    "BB17": 1904,
    "AB02": 1793,
    "BA09": 1739,
    "OP27": 1582,
    "ZZ2S": 1453,
    "IC07": 1444,
    "IC06": 1325,
    "BB14": 1305,
    "BB05": 1105,
    "ZZ02": 1024,
    "ZZS1": 480,
    "IC05": 326,
    "BB2S": 326,
    "ZZS2": 147,
    "OP19": 108,
    "BB20": 81,
    "OP1S": 81,
    "BB1S": 80,
    "AB1S": 33,
}
ntcs = list(ntc_frequencies.keys())
CANAs = [ "NAN", "AAA", "AAw", "AAu", "A-B", "B-A", "BBB", "BBw", "B12", "BB2", "miB", "ICL", "OPN", "SYN", "ZZZ" ]
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

def try_parse_int(s: str, default: Any) -> Any:
    """
    Tries to parse a string as an integer, returning a default value if it fails.
    """
    try:
        return int(s)
    except ValueError:
        return default

def _separate_subchains(v: pd.DataFrame) -> List[Dict[str, Any]]:
    """
    Separates 
    """

    steps = dict()
    for _, row in v.iterrows():
        pdbid = row['pdbid']
        chain = row['chain']
        nt1 = row['ntix_1']
        nt2 = row['ntix_2']
        if (pdbid, chain, nt1) in steps:
            raise ValueError(f"Duplicate step: {pdbid} {chain} {nt1}")
        steps[(pdbid, chain, nt1)] = (nt2, row)

    roots = set(steps.keys())
    for (pdbid, chain, _), (nt2, _) in steps.values():
        if (pdbid, chain, nt2) in roots:
            roots.remove((pdbid, chain, nt2))

    subchains = []
    for root in roots:
        subchain = []
        while root in steps:
            nt2, row = steps[root]
            del steps[root]
            subchain.append(row)
            root = (*root[:2], nt2)
        subchains.append(subchain)

    if len(steps) > 0:
        print("WARNING: loop or something weird detected in ", list(sorted(steps.keys())), v['step_ID'])

    subchains.sort(key=lambda subchain: (subchain[0]['pdbid'], subchain[0]['chain'], try_parse_int(subchain[0]['ntix_1'], subchain[0]['ntix_1'])))
    return subchains

def _process_subchain(steps):
    """
    Processes a list of steps (CSV rows) into a dictionary of
    * steps - original steps
    * sequence - single letter nucleotides
    * sequence_full - string array of nucleotides
    * is_dna - bool array
    """
    sequence = [ steps[0]['nt_1'] ]
    indices = [ steps[0]['ntix_1'] ]
    for s in steps:
        assert s['ntix_1'] == indices[-1]
        assert s['nt_1'] == sequence[-1]
        indices.append(s['ntix_2'])
        sequence.append(s['nt_2'])
    
    return {
        'steps': steps,
        'sequence': ''.join([ map_nucleotide(n) for n in sequence ]),
        'sequence_full': sequence,
        'indices': indices,
        'is_dna': np.array([ 'd' == s[0].lower() for s in sequence ], dtype=np.bool_),
    }

def load_csv_file(file) -> Tuple[pd.DataFrame, Dict[Tuple[str, str, Optional[int]], Dict[str, Any]]]:
    """
    Loads the CSV into
    * pandas dataframe including all columns from the CSV, plus
        - pdbid: string
        - chain: string
        - nt_1: string - first nucleotide of the step
        - nt_2: string - second nucleotide
        - ntix_1: int - second nucleotide index (index is 1-based in the whole sequence, sometimes it's a string like '20.A')
        - ntix_2: int - second nucleotide index
    * dictionary of chains: pdbid, chain -> dict
        - steps: subdataframe
        - sequence: string - single letter nucleotides
        - sequence_full: string array
        - is_dna: bool array
    """
    table = retry_on_error(lambda: pd.read_csv(file, sep=',', header='infer'))

    # split stepID into parts
    stepID: pd.Series[str] = table['step_ID']
    table['pdbid'] = [ s.split('_')[0] for s in stepID ]
    table['chain'] = [ s.split('_')[1] for s in stepID ]
    table['nt_1'] = [ s.split('_')[2] for s in stepID ]
    table['nt_2'] = [ s.split('_')[4] for s in stepID ]
    table['ntix_1'] = [ s.split('_')[3] for s in stepID ]
    table['ntix_2'] = [ s.split('_')[5] for s in stepID ]

    # some structures contain multiple variants of one nucleotide, the CSV then contain one step multiple times - for all possible conformations
    table.drop_duplicates(subset=['pdbid', 'chain', 'ntix_1', 'ntix_2'], inplace=True, ignore_index=True)

    # create array columns for each column in table, grouped by pdbid and chain
    # identity = lambda x: x
    groups = dict(list(table.groupby(['pdbid', 'chain'])))

    groups_dict: Dict[Tuple[str, str, Optional[int]], Dict[str, Any]] = {}
    for (k_pdb, k_chain), v in groups.items():
        # split the chain into subchains if there are any breaks in the sequence
        start_i = 0
        end_i = 0
        subchain_index = 0
        all_indices = set() # detect loops and weird things

        sequence = [ v['nt_1'].array[0] ]
        indices = [ v['ntix_1'].array[0] ]

        def yield_subchain():
            assert start_i < end_i, f"Can't yield empty subchain with {start_i=} and {end_i=}"
            assert end_i - start_i + 1 == len(sequence), f"Sequence length {sequence} doesn't match {end_i=} - {start_i=} + 1"
            if len(sequence) == 2:
                print(f"WARNING: Sus short sequence {k_pdb} {k_chain} {start_i}:{end_i}     {sequence}")
            slicing = not (subchain_index == 0 and end_i == len(v))
            if slicing:
                print(f"Slicing {k_pdb} {k_chain} {start_i}:{end_i}     {sequence}")
            key = (k_pdb, k_chain, subchain_index) if slicing else (k_pdb, k_chain, None)
            steps = v.iloc[start_i:end_i]
            groups_dict[key] = 

        for i in range(0, len(v)):
            if v['ntix_1'].array[i] != indices[-1]:
                all_indices.add(indices[-1])
                yield_subchain()

                start_i = end_i
                subchain_index += 1
                sequence = [ v['nt_1'].array[i] ]
                indices = [ v['ntix_1'].array[i] ]

            if v['ntix_1'].array[i] in all_indices:
                if start_i == i:
                    start_i += 1
                print(f"WARNING: Loop or something weird detected, skipping at {k_pdb} {k_chain} {v['ntix_1'].array[i]}")
                continue

            assert v['ntix_1'].array[i] == indices[-1]
            assert v['nt_1'].array[i] == sequence[-1]

            all_indices.add(v['ntix_1'].array[i])

            end_i = i + 1
            sequence.append(v['nt_2'].array[i])
            indices.append(v['ntix_2'].array[i])

        yield_subchain()

    return table, groups_dict

def get_joined_arrays(chains: Dict[Tuple[str, str, Optional[int]], dict]):
    """
    Joins the separated chains from load_csv_file into single arrays:
    * sequence: string - joined sequence separated by spaces
    * sequence_full: string array - joined sequence separated by spaces
    * is_dna: bool array - on separators it's False
    * indices: string array - array of original PDB indices, on separators it's ''
    * chain_names: string array - array of chain names (entry for each nucleotide), on separators it's ' '
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
    sequence_indices = np.concatenate(insert_spacers([ c['indices'] for c in cs ], [ "" ]))
    chain_names = np.concatenate(insert_spacers([ np.repeat(k[1], len(c['sequence_full'])) for k, c in chains.items() ], [ ' ' ]))
    assert len(sequence_full) == len(is_dna) == len(sequence_indices) == len(chain_names)

    def join_arrays(arrays, zero_element, dtype=None):
        arrays = insert_spacers(arrays, np.array([ zero_element, zero_element ], dtype=dtype))
        return np.concatenate(arrays, axis=0, dtype=dtype)

    # print(sequence)
    result = {
        "sequence": sequence,
        "sequence_full": sequence_full,
        "is_dna": is_dna,
        "indices": sequence_indices,
        "chain_names": chain_names,
        "NtC": join_arrays([ c['steps']['NtC'].to_numpy(np.str_) for c in cs ], "NANT", dtype=np.str_),
        "nearest_NtC": join_arrays([ c['steps']['nearest_NtC'].to_numpy(np.str_) for c in cs ], "NANT", dtype=np.str_),
        "CANA": join_arrays([ c['steps']['CANA'].to_numpy(np.str_) for c in cs ], "NAN"),
    }
    for k in [ "d1", "e1", "z1", "a2", "b2", "g2", "d2", "ch1", "ch2", "NN", "CC", "mu", "rmsd", "confalA", "confalG", "confalH" ]:
        result[k] = join_arrays([ c['steps'][k].to_numpy(np.float32) for c in cs ], 0.0, dtype=np.float32)
    return result

def read_fr3d_basepairing(file: Union[str, TextIO], filter_model = None, filter_chains: Optional[Set[str]] = None) -> Dict[str, np.ndarray]:
    """
    Reads the fr3d basepairing file into a dictionary of
    * model_i: int - model identifier
    * nt1_chain: string
    * nt2_chain: string
    * nt1_base: string - first nucleotide of the step
    * nt2_base: string - second nucleotide
    * nt1_ix: int - second nucleotide index (index is 1-based in the whole sequence)
    * nt2_ix: int - second nucleotide index
    * pairing: string - basepairing type
    """
    if isinstance(file, str):
        if file.endswith('.gz'):
            import gzip
            with gzip.open(file, 'rt') as f:
                return read_fr3d_basepairing(f, filter_model, filter_chains)
        elif file.endswith('.zst'):
            import zstandard
            with zstandard.open(file, 'rt') as f:
                return read_fr3d_basepairing(f, filter_model, filter_chains)
        else:
            with open(file, 'rt') as f:
                return read_fr3d_basepairing(f, filter_model, filter_chains)

    pairs: Dict = {
        "model_i": [],
        "nt1_base": [],
        "nt2_base": [],
        "nt1_chain": [],
        "nt2_chain": [],
        "nt1_ix": [],
        "nt2_ix": [],
        "pairing": [],
    }
    def parse_UnitID(unit_id: str):
        split = unit_id.split('|')
        # more than 5 is ok, sometimes the unit ID is something like 2D34|1|A|DC|5||||8_665, but we don't care about that
        assert len(split) >= 5, f"Invalid unit id {unit_id}"
        pdbid, model_i, chain, nt, ntix = split[:5]
        assert len(pdbid) == 4, f"Invalid pdbid {pdbid}"
        return (pdbid, int(model_i), chain, nt, ntix)
    all_models = defaultdict(lambda: 0)
    all_chains = defaultdict(lambda: 0)
    for line in file:
        left_nt, basepair_type, right_nt, some_number_which_is_always_zero_so_whatetever = line.split()
        left_pdbid, left_model_i, left_chain, left_nt, left_ntix = parse_UnitID(left_nt)
        right_pdbid, right_model_i, right_chain, right_nt, right_ntix = parse_UnitID(right_nt)

        if right_model_i != left_model_i:
            print(f"WARNING: {left_pdbid}:{left_model_i} has pairing with different model {right_pdbid}:{right_model_i}")

        all_models[left_model_i] += 1
        # print("filter_model", filter_model)
        if filter_model is not None:
            if filter_model == 'first':
                filter_model = left_model_i
            elif left_model_i != filter_model:
                continue
        all_chains["-".join(sorted((left_chain, right_chain)))] += 1
        if filter_chains is not None:
            if left_chain not in filter_chains or right_chain not in filter_chains:
                continue

        pairs["model_i"].append(left_model_i)
        pairs["nt1_base"].append(left_nt)
        pairs["nt2_base"].append(right_nt)
        pairs["nt1_chain"].append(left_chain)
        pairs["nt2_chain"].append(right_chain)
        pairs["nt1_ix"].append(left_ntix)
        pairs["nt2_ix"].append(right_ntix)
        pairs["pairing"].append(basepair_type)

    if len(all_models) > 0 and filter_model not in all_models:
        print(f"WARNING: model filter ({filter_model}) filtered out all basepairs. All models: {dict(sorted(all_models.items()))}")
    if len(pairs["model_i"]) == 0:
        print(f"WARNING: chain filter ({filter_chains}) filtered out all basepairs. All chains: {dict(sorted(all_chains.items()))}")

    pairs["model_i"] = np.array(pairs["model_i"], dtype=np.int32)
    pairs["nt1_base"] = np.array(pairs["nt1_base"], dtype=np.str_)
    pairs["nt1_chain"] = np.array(pairs["nt1_chain"], dtype=np.str_)
    pairs["nt1_ix"] = np.array(pairs["nt1_ix"], dtype=np.str_)
    pairs["nt2_base"] = np.array(pairs["nt2_base"], dtype=np.str_)
    pairs["nt2_chain"] = np.array(pairs["nt2_chain"], dtype=np.str_)
    pairs["nt2_ix"] = np.array(pairs["nt2_ix"], dtype=np.str_)
    pairs["pairing"] = np.array(pairs["pairing"], dtype=np.str_)
    return pairs

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
        if csv_extensions.search(file):
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
