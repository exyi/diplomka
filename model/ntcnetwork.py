from collections import namedtuple
import math
import time
from typing import Dict, List, Optional, Tuple
import torch, torch.nn as nn, torch.nn.functional as F, torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pandas as pd, numpy as np, os, sys
import csv_loader
# import torchtext
import random
from dataset import StructuresDataset
from hparams import Hyperparams
from utils import ConvKind, TensorDict, clamp, device, make_conv, ResnetBlock

class ConvEncoder(nn.Module):
    def __init__(self, input_size, channels = [64, 64], window_size = 3, kind: ConvKind = "resnet") -> None:
        """
        set of convolutional layers
        """
        super().__init__()
        input_sizes = [input_size, *channels]
        self.convolutions = nn.Sequential(*[
            make_conv(kind, 1, in_size, out_size, window_size)
            for in_size, out_size in zip(input_sizes, channels)
        ])

    def forward(self, input: torch.Tensor, lengths = None):
        # conv expects (batch, channels, seq_len)
        x = input
        x = torch.swapaxes(x, -1, -2)
        x = self.convolutions(x)
        x = torch.swapaxes(x, -1, -2)
        return x

class EncoderBaseline(nn.Module):
    def __init__(self, input_size, hidden_size, window_size = 3) -> None:
        """
        Encoder which predicts NtC based on the  only small window
        Used to test if the more advanced predictor is actually working better than "nothing"
        """
        super().__init__()
        self.embedding = nn.Embedding(input_size, hidden_size)
        self.conv = nn.Conv1d(hidden_size, hidden_size, kernel_size=window_size, padding='same')

    def forward(self, input: torch.Tensor, lengths = None):
        x = self.embedding(input)
        # conv expects (batch, channels, seq_len)
        x = torch.swapaxes(x, -1, -2)
        x = self.conv(x)
        x = torch.swapaxes(x, -1, -2)
        return x

class EncoderRNN(nn.Module):
    def __init__(self,
            embedding: nn.Module,
            embedding_size,
            hidden_size,
            num_layers,
            dropout,
            bidirectional=False
        ):
        super(EncoderRNN, self).__init__()

        self.embedding_size = embedding_size
        self.hidden_size = hidden_size
        self.bidirectional = bidirectional
        self.embedded_dropout = nn.Dropout(0.1)
        self.embed_dropout = nn.Dropout(0.2)

        self.embedding = embedding
        self.embedded_dropout = nn.Dropout(dropout)
        self.rnn = nn.LSTM(embedding_size, hidden_size, bidirectional=bidirectional, num_layers=num_layers, dropout=dropout)
        self.output_dropout = nn.Dropout(dropout)

    def forward(self, input, lengths=None):
        if self.embedded_dropout.p > 0:
            input = self.embedded_dropout(input)
        embedded = self.embedding(input)
        embedded = self.embedded_dropout(embedded)

        if lengths is not None:
            embedded = torch.nn.utils.rnn.pack_padded_sequence(embedded, lengths, batch_first=True)
        output, _ = self.rnn(embedded)
        if lengths is not None:
            output, _ = torch.nn.utils.rnn.pad_packed_sequence(output, batch_first=True)
        if self.bidirectional:
            output = output[:, :, :self.hidden_size] + output[:, :, self.hidden_size:]
        output = self.output_dropout(output)
        return output

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

    def forward(self, encoder_outputs, lengths):
        output = F.relu(encoder_outputs)
        output = self.out(output).softmax(dim=1)
        return output

class Network(nn.Module):
    NUCLEOTIDE_LABELS = csv_loader.basic_nucleotides
    INPUT_SIZE = len(csv_loader.basic_nucleotides) + 1
    NTC_LABELS = csv_loader.ntcs
    OUTPUT_TUPLE = namedtuple("Output", ["NtC"])
    def __init__(self, p: Hyperparams):
        super(Network, self).__init__()
        self.p = p
        embedding_size = p.conv_channels[-1]
        hidden_size = p.rnn_size if p.rnn_layers > 0 else embedding_size
        if True:
            embedding = ConvEncoder(Network.INPUT_SIZE, channels=p.conv_channels, window_size=p.conv_window_size, kind=p.conv_kind)
        else:
            embedding = nn.Embedding(Network.INPUT_SIZE, embedding_size)
        self.encoder = EncoderRNN(embedding, embedding_size, hidden_size, num_layers=p.rnn_layers, dropout=p.rnn_dropout, bidirectional=True)
        # self.encoder = EncoderBaseline(Network.INPUT_SIZE, hidden_size)
        self.ntc_decoder = Decoder(hidden_size, len(csv_loader.ntcs))

        self.ntc_loss = nn.CrossEntropyLoss(
            # weight=torch.Tensor([
            #     0.01 if k == "NANT" else
            #     clamp(1 / (v / 20_000), 0.2, 1)
            #     for k, v in csv_loader.ntc_frequencies.items()
            # ]),
            weight=torch.Tensor([
                0.01 if k == "NANT" else
                1
                for k, v in csv_loader.ntc_frequencies.items()
            ]),
            label_smoothing=0.1
        )
        print(self.ntc_loss.weight)
    
    def forward(self, input: TensorDict, whatever =None):
        # print(input["sequence"].shape, input["is_dna"].shape)
        in_tensor = torch.cat([
            F.one_hot(input["sequence"], num_classes=len(csv_loader.basic_nucleotides)),
            torch.unsqueeze(input["is_dna"], -1)
        ], dim=-1)
        in_tensor = in_tensor.type(torch.float32)
        lengths = input.get("lengths", None)
        if lengths is not None:
            lengths = lengths.to("cpu")
        encoder_output = self.encoder(in_tensor, lengths)
        encoder_output = encoder_output[:, 1:, :] # there is one less NtC than nucleotides
        decoder_output = self.ntc_decoder(encoder_output, lengths)
        return {
            "NtC": torch.swapaxes(decoder_output, -1, -2)
        }

    def loss(self, output: TensorDict, target):
        # print(output["NtC"].shape)
        # print(target["NtC"].shape)
        loss = self.ntc_loss(output["NtC"], target["NtC"])
        return loss
