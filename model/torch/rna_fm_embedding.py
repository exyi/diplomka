
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

    for x in model.layers:
        print(x.self_attn)
        x.self_attn = torch.jit.script(x.self_attn)
    
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
            self.allow_grad = False
            self.requires_grad_(self.allow_grad)

        def forward(self, batch: Union[list[str], torch.Tensor], lengths: torch.LongTensor):
            if self.allow_grad:
                return self.forward_core(batch, lengths)
            else:
                with torch.no_grad():
                    return self.forward_core(batch, lengths)
        def forward_core(self, batch: Union[list[str], torch.Tensor], lengths: torch.LongTensor):
            if isinstance(batch, list):
                assert isinstance(batch[0], str)
                batch_t = self.str_batch_converter([ (f"label{i}", x) for i, x in enumerate(batch) ])

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

            results = self.model(batch_t, repr_layers=[12])
            embedding = results["representations"][12]

            if alphabet.append_eos:
                embedding = embedding[:, :-1, :]
            if alphabet.prepend_bos:
                embedding = embedding[:, 1:, :]
            return embedding

    return ModelWrapper
