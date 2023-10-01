from collections import namedtuple
import math
import time
from typing import Dict, List, Optional, Tuple
import torch, torch.nn as nn, torch.nn.functional as F, torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pandas as pd, numpy as np, os, sys
from model import csv_loader, sample_weight
# import torchtext
import random
from model.hyperparameters import Hyperparams
from .torchutils import TensorDict, to_cpu, to_device, to_torch, device
from .sequence_encoders import ConvEncoder, EncoderBaseline, EncoderRNN

class Decoder(nn.Module):
    def __init__(self, hidden_size, output_size, attention_span = 0):
        super(Decoder, self).__init__()
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.out = nn.Linear(hidden_size, output_size)

        if attention_span == 0:
            self.attn = None
        elif attention_span is None:
            # self-attention TODO
            self.mh_attention = nn.MultiheadAttention(
                hidden_size, num_heads=1)
            assert False
        else:
            self.attn = nn.Linear(self.hidden_size, attention_span)
            self.attn_combine = nn.Linear(self.hidden_size * 2, self.hidden_size)

    def forward(self, encoder_outputs, lengths: torch.LongTensor):
        output = F.relu(encoder_outputs)
        output = self.out(output).softmax(dim=1)
        return output

class Network(nn.Module):
    NUCLEOTIDE_LABELS = csv_loader.basic_nucleotides
    INPUT_SIZE = len(csv_loader.basic_nucleotides) + 1
    NTC_LABELS = csv_loader.ntcs
    CANA_LABELS = csv_loader.CANAs
    OUTPUT_TUPLE = namedtuple("Output", ["NtC"])
    def __init__(self, p: Hyperparams):
        super(Network, self).__init__()
        self.p = p
        embedding_size = p.conv_channels[-1]
        hidden_size = p.rnn_size if p.rnn_layers > 0 else embedding_size
        if len(p.conv_channels) > 0 or p.conv_window_size > 1:
            embedding = ConvEncoder(Network.INPUT_SIZE, channels=p.conv_channels, window_size=p.conv_window_size, kind=p.conv_kind)
        else:
            embedding = nn.Embedding(Network.INPUT_SIZE, embedding_size)

        if p.external_embedding is not None:
            if p.external_embedding == "rnafm":
                raise Exception("TODO")
            else:
                raise Exception("Unknown external embedding: " + p.external_embedding)
            
        if p.basepairing == "none":
            pass
        else:
            # TODO
            raise Exception("Unknown basepairing mode: " + p.basepairing)

        # self.encoder = EncoderBaseline(Network.INPUT_SIZE, hidden_size)
        self.encoder = EncoderRNN(embedding, embedding_size, hidden_size, num_layers=p.rnn_layers, dropout=p.rnn_dropout, bidirectional=True)

        self.ntc_decoder = Decoder(hidden_size, len(csv_loader.ntcs))
        self.cana_decoder = Decoder(hidden_size, len(csv_loader.CANAs))

        self.ntc_weights = torch.Tensor(sample_weight.get_ntc_weight(p.sample_weight.split("+")[0], Network.NTC_LABELS))
        print("NtC loss weights: ", self.ntc_weights)
        self.ntc_loss = nn.CrossEntropyLoss(
            weight=self.ntc_weights,
            label_smoothing=0.1,
            reduction="none"
        )
        print(self.ntc_loss.weight)
    
    def forward(self, input: TensorDict, whatever =None):
        # print(input["sequence"].shape, input["is_dna"].shape)
        in_tensor = torch.cat([
            F.one_hot(input["sequence"], num_classes=len(csv_loader.basic_nucleotides)),
            torch.unsqueeze(input["is_dna"], -1)
        ], dim=-1).to(device)
        in_tensor = in_tensor.type(torch.float32)
        lengths = to_cpu(input.get("lengths", None))
        encoder_output = self.encoder(in_tensor, lengths)
        encoder_output = encoder_output[:, 1:, :] # there is one less NtC than nucleotides
        decoder_output = self.ntc_decoder(encoder_output, lengths)

        outputs = { "lengths": lengths }
        return {
            "NtC": torch.swapaxes(decoder_output, -1, -2),
            "lengths": lengths
        }

    def loss(self, output: TensorDict, target):
        loss: torch.Tensor = self.ntc_loss(output["NtC"], target["NtC"])
        
        total_count = loss.shape.numel()
        if target.get("lengths", None) is not None:
            lengths = target["lengths"].to(loss.device)
            mask = torch.arange(target["NtC"].shape[1]).to(loss.device) < lengths.unsqueeze(-1)
            total_count = mask.sum()
            loss = loss * mask

        loss = loss.sum() / total_count
        return loss
