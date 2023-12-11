
from typing import Any, Dict, List, Union
import os, sys, json, dataclasses, numpy as np
from dataclasses import dataclass

@dataclass
class AlphabetDefinition:
    tok_to_idx: Dict[str, int]
    padding_idx: int
    append_eos: bool
    prepend_bos: bool

def alphabet_translation_table(alphabet: AlphabetDefinition, our_alphabet: List[str]):
    alpha_tokix = {k.upper(): i for k, i in alphabet.tok_to_idx.items()}
    def translation(x):
        if x == " ":
            return alpha_tokix["-"]
        return alpha_tokix.get(x.upper(), alpha_tokix["<UNK>"])

    return np.array([ translation(k) for k in our_alphabet ])

def get_alphabet(file: str):
    if os.path.exists(file + ".alphabet"):
        with open(file + ".alphabet") as f:
            d = json.load(f)
            return AlphabetDefinition(**d)
    else:
        raise Exception(f"Alphabet file not found: {file}.alphabet")

def og_rnafm_embedding():
    import fm
    import model.torch.torchutils as torchutils
    import torch, torch.nn.functional as F
    device = torchutils.device
    model, alphabet = fm.pretrained.rna_fm_t12()

    #for x in model.layers:
        #x.self_attn = torch.jit.script(x.self_attn)
    
    alphabet_def = AlphabetDefinition(
        tok_to_idx = alphabet.tok_to_idx,
        padding_idx = alphabet.padding_idx,
        append_eos = alphabet.append_eos,
        prepend_bos = alphabet.prepend_bos,
    )

    class ModelWrapper(torch.nn.Module):
        SIZE = 640
        def __init__(self, our_alphabet: list[str]):
            super().__init__()
            self.model = model
            # self.model = torch.jit.trace(model, torch.zeros(8, 514, dtype=torch.long, device=device))
            self.translation_table = torch.tensor(alphabet_translation_table(alphabet_def, our_alphabet))
            self.str_batch_converter = alphabet.get_batch_converter()
            self._allow_grad = True
            #self.requires_grad_(self._allow_grad)

        def enable_training_(self, x=True):
            self._allow_grad = x
            self.requires_grad_(self._allow_grad)
            self.train()

        def forward(self, batch: Union[list[str], torch.Tensor], lengths: torch.LongTensor):
            if self._allow_grad:
                return self.forward_core(batch, lengths)
            else:
                with torch.no_grad():
                    return self.forward_core(batch, lengths)
        
        def convert_batch(self, batch: Union[list[str], torch.Tensor], lengths: torch.LongTensor) -> torch.Tensor:
            if isinstance(batch, list):
                assert isinstance(batch[0], str)
                batch_t = self.str_batch_converter([ (f"label{i}", x) for i, x in enumerate(batch) ])[2]
            else:
                assert len(batch.shape) == 2
                batch_size = batch.shape[0]
                batch_t = torch.gather(self.translation_table.to(batch.device), 0, batch.view(-1)).view(batch.shape).to(device)
                # add paddings
                lenmask = (torch.arange(batch_t.shape[1]) < lengths.unsqueeze(-1)).to(device)
                batch_t = batch_t * lenmask + alphabet.padding_idx * (~lenmask)
                # add EOS
                if alphabet.append_eos:
                    batch_t = torch.cat([ batch_t, torch.full((batch_size, 1), alphabet.padding_idx, device=device) ], dim=1)
                    batch_t[torch.arange(batch_size, device=device), lengths.to(device)] = alphabet.eos_idx
                # add BOS
                if alphabet.prepend_bos:
                    batch_t = torch.cat([ torch.full((batch_size, 1), alphabet.cls_idx, device=device), batch_t ], dim=1)
            return batch_t

        def forward_core(self, batch: Union[list[str], torch.Tensor], lengths: torch.LongTensor):
            batch_t = self.convert_batch(batch, lengths)
            print(batch_t.shape)
            embedding = self.call_model(batch_t)

            if alphabet.append_eos:
                embedding = embedding[:, :-1, :]
            if alphabet.prepend_bos:
                embedding = embedding[:, 1:, :]
            return embedding
        
        def call_model(self, batch_t: torch.Tensor) -> torch.Tensor:
            max_len = 1024 - 8
            overlap = 0
            if batch_t.shape[1] > max_len:
                batches = [ batch_t[:, i:i+max_len] for i in range(0, batch_t.shape[1], max_len - overlap) ]
                embeddings = [ self.call_model(x) for x in batches ]
                return torch.cat([ x for x in embeddings ], dim=1)
                # sum the overlapping regions and concat
                embeddings1 = [ x[:, :-overlap, :] for x in embeddings[:-1] ]
                embeddings2 = [ x[:, overlap:, :] for x in embeddings[1:] ]

            
            results = self.model(batch_t, repr_layers=[12])
            return results["representations"][12]

    return ModelWrapper
