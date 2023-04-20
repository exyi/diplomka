import os
from typing import Dict, List, Optional, Tuple
from torch.utils.data import Dataset, DataLoader
import torch
import csv_loader
from utils import TensorDict

class StructuresDataset(Dataset):
    def __init__(self, dir: str, files: Optional[List[str]] = None):
        """
        Loads CSV files from a directory as a Torch dataset. Returns
        * input:
            - sequence: array as ints
            - is_dna: array of booleans indicating whether the nucleotide is DNA or RNA
        * target:
            - NtC: array as ints
        """
        self.dir = dir
        if files is None:
            self.files = [ d for d in os.listdir(dir) if d.endswith('.csv') or d.endswith('.csv.gz') ]
        else:
            self.files = files

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx) -> Tuple[TensorDict, TensorDict]:
        img_path = os.path.join(self.dir, self.files[idx])
        table, chains = csv_loader.load_csv_file(img_path)
        joined = csv_loader.get_joined_arrays(chains)
        input = {
            "sequence": torch.LongTensor(csv_loader.encode_nucleotides(joined['sequence'])),
            "is_dna": torch.BoolTensor(joined['is_dna'])
        }
        target: TensorDict = {
            "NtC": torch.LongTensor(csv_loader.encode_ntcs(joined['NtC'])),
            # "CANA": joined['CANA']
        }
        return input, target

    @staticmethod
    def collate_fn(batch: List[Tuple[TensorDict, TensorDict]]):
        """
        Converts list of items to a single batch
        Adds "lengths" field to the result
        """
        batch.sort(key=lambda x: x[0]["sequence"].shape[0], reverse=True)
        ## get sequence lengths
        lengths = torch.tensor([ t[0]["sequence"].shape[0] for t in batch ])
        result_in = dict()
        for key in [ "sequence", "is_dna" ]:
            result_in[key] = torch.nn.utils.rnn.pad_sequence([ t[0][key] for t in batch ], batch_first=True)
        result_in["lengths"] = lengths
        result_out = dict()
        for key in [ "NtC" ]:
            result_out[key] = torch.nn.utils.rnn.pad_sequence([ t[1][key] for t in batch ], batch_first=True)
        return result_in, result_out
