#!/usr/bin/env python3
from dataclasses import dataclass
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
    alt_letters = set()
    for row in v.itertuples():
        pdbid = row.pdbid
        chain = row.chain
        nt1 = row.ntix_1
        nt2 = row.ntix_2
        key = (pdbid, chain, nt1, row.ntalt_1)
        key2 = (pdbid, chain, nt2, row.ntalt_2)
        alt_letters.add(row.ntalt_1)
        alt_letters.add(row.ntalt_2)
        if key in steps:
            if not row.ntalt_2:
                raise ValueError(f"Duplicate step: {key}")
            
            # on alternates, use the lower numbered nucleotide, drop the other one
            if row.ntalt_2 < steps[key][1].ntalt_2:
                steps[key] = (key2, row)
        else:
            steps[key] = (key2, row)

    alt_letters.discard("")

    roots = set(steps.keys())
    for _, (key2, _) in steps.items():
        if key2 in roots:
            roots.remove(key2)

    # print("Chain roots", roots)
    assert len(roots) > 0
    # remove alternative chains to avoid id conflicts
    for root in reversed(list(roots)):
        pdbid, chain, nt1, alt = root
        if alt and len(roots) > 1:
            for alt_alt in alt_letters:
                if alt_alt != alt and (pdbid, chain, nt1, alt_alt) in steps:
                    print(f"Removing alternate chain: {root}, alternative {alt_alt} exists")
                    roots.remove(root)
                    del steps[root]
                    break

    subchains = []
    used_ids = set()
    for root in roots:
        if root[:3] in used_ids:
            print("WARNING: duplicate root", root, "->", steps[root][0], "Does this structure have a B variant which is longer than A variant?")
            continue
        subchain = []
        used_ids.add(root[:3])
        while root in steps:
            key2, row = steps[root]

            if key2[:3] in used_ids:
                print(f"WARNING: duplicate node {key2} -> {steps.get(key2, ['end'])[0]}, terminating current subchain len={len(subchain)}")
                break
            used_ids.add(key2[:3])

            del steps[root]
            used_ids.add(root[:3])
            subchain.append(row)
            root = key2
        assert len(subchain) > 0
        subchains.append(subchain)

    # remove all unused variants, we only dropped the roots, not all members of a potentially longer chain
    for key in list(steps):
        pdbid, chain, nt1, alt = key
        if alt:
            del steps[key]

    if len(steps) > 0:
        print(f"WARNING: {len(steps)} orphaned steps")
        print("WARNING: loop or something weird detected in ", list(sorted(steps.keys())), v['step_ID'].array)

    # subchains.sort(key=lambda subchain: (subchain[0]['pdbid'], subchain[0]['chain'], try_parse_int(subchain[0]['ntix_1'], subchain[0]['ntix_1'])))
    return subchains

def _process_subchain(steps):
    """
    Processes a list of steps (CSV rows) into a dictionary of
    * steps - original steps
    * sequence - single letter nucleotides
    * sequence_full - string array of nucleotides
    * is_dna - bool array
    """
    sequence = [ steps[0].nt_1 ]
    indices = [ steps[0].ntix_1 ]
    for s in steps:
        assert s.ntix_1 == indices[-1]
        assert s.nt_1 == sequence[-1], f"Nucleotide {s.nt_1} != {sequence[-1]}, at position {s.pdbid}:{s.chain}:{s.ntix_1}-{s.ntix_2}, all steps: {', '.join([ s.step_ID for s in steps])}"
        indices.append(s.ntix_2)
        sequence.append(s.nt_2)
    
    return {
        'steps': pd.DataFrame(steps),
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
        - ntalt_1: string - alternative code of the first nucleotide (e.g. '1' for 'A.1')
        - nt_2: string - second nucleotide
        - ntalt_2: string - alternative code of the second nucleotide
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
    splitStepID = list(zip(*[ s.split('_') for s in stepID ]))
    table['pdbid'] = splitStepID[0]
    table['model_i'] = [ [
        a := re.search(r"[a-z]{4}-m(\d)+", x, re.IGNORECASE),
        int(a.group(1)) if a else 1
    ][-1] for x in splitStepID[0] ]
    table['chain'] = splitStepID[1]
    table['nt_1'] = [ x.split('.')[0] for x in splitStepID[2] ]
    table['ntalt_1'] = [ x.split('.')[1] if '.' in x else '' for x in splitStepID[2] ]
    table['nt_2'] = [ x.split('.')[0] for x in splitStepID[4] ]
    table['ntalt_2'] = [ x.split('.')[1] if '.' in x else '' for x in splitStepID[4] ]
    table['ntix_1'] = splitStepID[3]
    table['ntix_2'] = splitStepID[5]

    for col in table.columns:
        if table[col].dtype == np.float64:
            table[col] = table[col].astype(np.float32)

    # some structures contain multiple variants of one nucleotide, the CSV then contain one step multiple times - for all possible conformations
    # table.drop_duplicates(subset=['pdbid', 'chain', 'ntix_1', 'ntalt_1', 'ntix_2', 'ntalt_2'], inplace=True, ignore_index=True)

    # create array columns for each column in table, grouped by pdbid and chain
    # identity = lambda x: x
    groups = dict(list(table.groupby(['pdbid', 'chain'])))

    groups_dict: Dict[Tuple[str, str, Optional[int]], Dict[str, Any]] = {}
    for (k_pdb, k_chain), v in groups.items():
        subchains_raw = _separate_subchains(v)
        subchains = [ _process_subchain(s) for s in subchains_raw ]

        if len(subchains) > 1:
            for i, subchain in enumerate(subchains):
                groups_dict[(k_pdb, k_chain, i)] = subchain
        else:
            groups_dict[(k_pdb, k_chain, None)] = subchains[0]

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

@dataclass
class StepID:
    pdbid: str
    model: int
    chain: str
    nt1_base: str
    nt1_alternate_id: str
    nt1_component_number: str
    nt1_insertion_code: str
    nt2_base: str
    nt2_alternate_id: str
    nt2_component_number: str
    nt2_insertion_code: str

    @property
    def unit1(self):
        return UnitID(self.pdbid, self.model, self.chain, self.nt1_base, self.nt1_component_number, alternate_id=self.nt1_alternate_id, insertion_code=self.nt1_insertion_code)

    @property
    def unit2(self):
        return UnitID(self.pdbid, self.model, self.chain, self.nt2_base, self.nt2_component_number, alternate_id=self.nt2_alternate_id, insertion_code=self.nt2_insertion_code)
    
    @property
    def nt1_ix(self):
        return self.nt1_component_number + "." + self.nt1_insertion_code if self.nt1_insertion_code else self.nt1_component_number
    @property
    def nt2_ix(self):
        return self.nt2_component_number + "." + self.nt2_insertion_code if self.nt2_insertion_code else self.nt2_component_number

    @staticmethod
    def parse_csv_step(unit_id: str) -> Tuple['UnitID', 'UnitID']:
        # format is {pdbid}[-m{model_i}]_{chain}_{nt1}[.{nt1-alternate}]_{nt1ix}[.{insertion_code}]_{nt2}[.{nt2-alternate}]_{nt2ix}[.{insertion_code}]

        def split_dot(s: str):
            if '.' in s:
                return s.split('.')
            else:
                return s, ""

        pdbid, chain, nt1, nt1ix, nt2, nt2ix = unit_id.split('_')
        if '-' in pdbid:
            pdbid, model_i = pdbid.split('-')
            model_i = int(model_i.lstrip('m'))
        else:
            model_i = 1
        
        nt1, nt1_alt = split_dot(nt1)
        nt2, nt2_alt = split_dot(nt2)
        nt1ix, nt1_ins = split_dot(nt1ix)
        nt2ix, nt2_ins = split_dot(nt2ix)

        return (
            UnitID(pdbid, model_i, chain, nt1, nt1ix, "", nt1_alt, nt1_ins, None),
            UnitID(pdbid, model_i, chain, nt2, nt2ix, "", nt2_alt, nt2_ins, None)
        )

@dataclass
class UnitID:
    """
    Represents https://www.bgsu.edu/research/rna/help/rna-3d-hub-help/unit-ids.html
    Used for parsing FR3D files
    """
    pdbid: str
    """
    PDB ID Code
        From PDBx/mmCIF item: _entry.id
        4 characters, case-insensitive
    """
    model_i: int
    """
    Model Number
        From PDBx/mmCIF item: _atom_site.pdbx_PDB_model_num
        integer, range 1-99
    """
    chain: str
    """
    Model Number
        From PDBx/mmCIF item: _atom_site.pdbx_PDB_model_num
        integer, range 1-99
    """
    residue_base: str
    """
    Residue/Nucleotide/Component Identifier
        From PDBx/mmCIF item: _atom_site.label_comp_id
        1-3 characters, case-insensitive
    """
    residue_component_number: str
    """
    Residue/Nucleotide/Component Number
        From PDBx/mmCIF item: _atom_site.auth_seq_id
        integer, range: -999..9999 (there are negative residue numbers)
    """
    atom_name: str = ""
    """
    Atom Name (Optional, default: blank)
        From PDBx/mmCIF item: _atom_site.label_atom_id
        0-4 characters, case-insensitive
        blank means all atoms
    """
    alternate_id: str = ""
    """
    Alternate ID (Optional, default: blank)
        From PDBx/mmCIF item: _atom_site.label_alt_id
        Default value: blank
        One of ['A', 'B', 'C', '0'], case-insensitive
        This represents alternate coordinates for the model of one or more atoms
    """
    insertion_code: str = ""
    """
    Insertion Code (Optional, default: blank)
        From PDBx/mmCIF item: _atom_site.pdbx_PDB_ins_code
        1 character, case-insensitive
    """
    symmetry_operation: Optional[str] = None
    """
    Symmetry Operation (Optional, default: 1_555)
        As defined in PDBx/mmCIF item: _pdbx_struct_oper_list.name
        5-6 characters, case-insensitive
        For viral icosahedral structures, use “P_” + operator number instead of symmetry operators. For example, 1A34|1|A|VAL|88|||P_1
    """

    @property
    def residue_id(self) -> str:
        """
        Residue ID in form {number}.{insertion_code}, if the insertion_code id is non-empty
        """
        if self.insertion_code:
            return str(self.residue_component_number) + "." + self.insertion_code
        else:
            return str(self.residue_component_number)
        
    @property
    def base_with_alt(self) -> str:
        """
        Base with alternate ID, if the alternate ID is non-empty
        """
        if self.alternate_id:
            return self.residue_base + "." + self.alternate_id
        else:
            return self.residue_base

    @property
    def residue_position(self) -> Tuple[str, int, str, str]:
        return (self.pdbid, self.model_i, self.chain, self.residue_id)

    @staticmethod
    def parse(unit_id: str) -> 'UnitID':
        split = unit_id.split('|')
        # more than 5 is ok, sometimes the unit ID is something like 2D34|1|A|DC|5||||8_665, but we don't care about that
        assert len(split) >= 5, f"Invalid unit id {unit_id}"
        pdbid, model_i, chain, nt, ntix = split[:5]
        assert len(pdbid) == 4, f"Invalid pdbid {pdbid}"

        if len(split) > 5:
            atom_name = split[5]
        else:
            atom_name = ""
        
        if len(split) > 6:
            alternate_id = split[6]
        else:
            alternate_id = ""
        
        if len(split) > 7:
            insertion_code = split[7]
        else:
            insertion_code = ""
        
        if len(split) > 8:
            symmetry_operation = split[8]
        else:
            symmetry_operation = None

        return UnitID(pdbid, int(model_i), chain, nt, ntix, atom_name, alternate_id, insertion_code, symmetry_operation)

    def __str__(self) -> str:
        components = [ self.pdbid, str(self.model_i), self.chain, self.residue_base, self.residue_id, self.atom_name, self.alternate_id, self.insertion_code, self.symmetry_operation ]

        while len(components) > 5 and not components[-1]:
            components.pop()

        return "|".join(components)
    
    def __repr__(self) -> str:
        return str(self)

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


def read_fr3d_basepairing(file: Union[str, TextIO], pdbid: str, filter_model = None, filter_chains: Optional[Set[str]] = None) -> Dict[str, np.ndarray]:
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
                return read_fr3d_basepairing(f, pdbid, filter_model, filter_chains)
        elif file.endswith('.zst'):
            import zstandard
            with zstandard.open(file, 'rt') as f:
                return read_fr3d_basepairing(f, pdbid, filter_model, filter_chains)
        else:
            with open(file, 'rt') as f:
                return read_fr3d_basepairing(f, pdbid, filter_model, filter_chains)

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
    all_models = defaultdict(lambda: 0)
    all_chains = defaultdict(lambda: 0)
    for line in file:
        left_unit_id, basepair_type, right_unit_id, some_number_which_is_always_zero_so_whatetever = line.split()
        left = UnitID.parse(left_unit_id)
        right = UnitID.parse(right_unit_id)

        if right.model_i != left.model_i:
            print(f"WARNING: {left} has pairing with different model {right}")

        all_models[left.model_i] += 1
        # print("filter_model", filter_model)
        if filter_model is not None:
            if filter_model == 'first':
                filter_model = left.model_i
            elif left.model_i != filter_model:
                continue
        all_chains["-".join(sorted((left.chain, right.chain)))] += 1
        if filter_chains is not None:
            if left.chain not in filter_chains or right.chain not in filter_chains:
                continue

        pairs["model_i"].append(left.model_i)
        pairs["nt1_base"].append(left.residue_base)
        pairs["nt2_base"].append(right_unit_id)
        pairs["nt1_chain"].append(left.chain)
        pairs["nt2_chain"].append(right.chain)
        pairs["nt1_ix"].append(left.residue_id)
        pairs["nt2_ix"].append(right.residue_id)
        pairs["pairing"].append(basepair_type)

    if filter_model is not None and len(all_models) > 0 and filter_model not in all_models:
        print(f"WARNING: model filter ({filter_model}) filtered out all basepairs in {pdbid}. All models: {dict(sorted(all_models.items()))}")
    if len(all_chains) > 0 and len(pairs["model_i"]) == 0:
        print(f"NOTE: chain filter ({filter_chains}) filtered out all basepairs in {pdbid}. All chains: {dict(sorted(all_chains.items()))}")

    pairs["model_i"] = np.array(pairs["model_i"], dtype=np.int32)
    pairs["nt1_base"] = np.array(pairs["nt1_base"], dtype=np.str_)
    pairs["nt1_chain"] = np.array(pairs["nt1_chain"], dtype=np.str_)
    pairs["nt1_ix"] = np.array(pairs["nt1_ix"], dtype=np.str_)
    pairs["nt2_base"] = np.array(pairs["nt2_base"], dtype=np.str_)
    pairs["nt2_chain"] = np.array(pairs["nt2_chain"], dtype=np.str_)
    pairs["nt2_ix"] = np.array(pairs["nt2_ix"], dtype=np.str_)
    pairs["pairing"] = np.array(pairs["pairing"], dtype=np.str_)
    return pairs

def save_parquet(df, file):
    import pyarrow
    import pyarrow.parquet as pq
    table = pyarrow.Table.from_pandas(df, preserve_index=False)
    # df.to_parquet(f'./table_{tmp_counter}.parquet', compression='brotli', index=False)
    pq.write_table(table, file, compression="ZSTD", compression_level=12)
def save_chains_parquet(structs, file):
    frame = pd.DataFrame(pd.concat(structs).groupby("name").apply(lambda x: x.to_dict(orient='records')), columns=['chains'])
    frame.to_parquet(file)

def load_csvs(path, pairing_path):
    """
    Dumps the csvs into parquet files for DuckDB querying
    """
    pairing_files = find_pairing_files(pairing_path) if pairing_path else None
    structs = []
    tables = []
    dirs: List[str] = os.listdir(path)
    tmp_counter = 0
    row_counter = 0
    for ix, file in enumerate(dirs):
        if csv_extensions.search(file):
            print(f'Loading {ix+1:<8}/{len(dirs)}: {file}                                 ', end='')
            table, dicts = load_csv_file(os.path.join(path, file))
            pdbid = re.split("[._]", file)[0]
            if pairing_files is not None:
                if pdbid not in pairing_files:
                    print(f"WARNING: no pairing file for {pdbid}")
                else:
                    pairing = read_fr3d_basepairing(pairing_files[pdbid], pdbid)
                    pairing_d = {}
                    for (model_i, chain1, ix1, chain2, ix2, ptype) in zip(pairing['model_i'], pairing['nt1_chain'], pairing['nt1_ix'], pairing['nt2_chain'], pairing['nt2_ix'], pairing['pairing']):
                        key = (model_i, chain1, ix1)
                        if key not in pairing_d:
                            pairing_d[key] = []
                        pairing_d[key].append({"chain": chain2, "ix": ix2, "type": ptype})
                    table['pairs1'] = [ pairing_d.get((model_i, chain, ix), []) for model_i, chain, ix in zip(table['model_i'], table['chain'], table['ntix_1']) ]
                    table['pairs2'] = [ pairing_d.get((model_i, chain, ix), []) for model_i, chain, ix in zip(table['model_i'], table['chain'], table['ntix_2']) ]
            structs.append(dicts)
            tables.append(table)
            row_counter += len(table)
            print('\r', end='')

        if row_counter > 600_000:
            # save_parquet(structs, f'./chains_{tmp_counter}.parquet')
            print(f"Saving table_{tmp_counter}.parquet with {row_counter} rows, {len(structs)} structures")
            save_parquet(pd.concat(tables), f'./table_{tmp_counter}.parquet')
            tmp_counter += 1
            structs.clear()
            tables.clear()
            row_counter = 0

    # save_parquet(structs, f'./chains_{tmp_counter}.parquet')
    save_parquet(pd.concat(tables), f'./table_{tmp_counter}.parquet')

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Parse CSVs')
    parser.add_argument('--path', type=str, help='directory containing CSVs to load', required=True)
    parser.add_argument('--pairing', type=str, help='directory containing basepair files to load')
    args = parser.parse_args()
    load_csvs(args.path, args.pairing)
