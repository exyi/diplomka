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
from utils import TensorDict, device

class EncoderRNN(nn.Module):
    def __init__(self, input_size, embedding_size, hidden_size, bidirectional=True):
        super(EncoderRNN, self).__init__()
        self.embedding_size = embedding_size
        self.hidden_size = hidden_size
        self.bidirectional = bidirectional

        self.embedding = nn.Embedding(input_size, embedding_size)
        self.rnn = nn.LSTM(embedding_size, hidden_size, bidirectional=bidirectional)

    def forward(self, input, lengths=None):
        embedded = self.embedding(input)
        if lengths is not None:
            embedded = torch.nn.utils.rnn.pack_padded_sequence(embedded, lengths, batch_first=True)
        output, _ = self.rnn(embedded)
        if lengths is not None:
            output, _ = torch.nn.utils.rnn.pad_packed_sequence(output, batch_first=True)
        if self.bidirectional:
            output = output[:, :, :self.hidden_size] + output[:, :, self.hidden_size:]
        return output

class Decoder(nn.Module):
    def __init__(self, hidden_size, output_size):
        super(Decoder, self).__init__()
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.out = nn.Linear(hidden_size, output_size)
        # self.attn = nn.Linear(self.hidden_size, self.max_length)
        # self.attn_combine = nn.Linear(self.hidden_size * 2, self.hidden_size)

    def forward(self, encoder_outputs):
        output = F.relu(encoder_outputs)
        output = self.out(output).softmax(dim=1)
        return output

class Network(nn.Module):
    NUCLEOTIDE_LABELS = csv_loader.basic_nucleotides
    NTC_LABELS = csv_loader.ntcs
    def __init__(self, embedding_size, hidden_size):
        super(Network, self).__init__()
        self.encoder = EncoderRNN(len(csv_loader.basic_nucleotides), embedding_size, hidden_size)
        self.ntc_decoder = Decoder(hidden_size, len(csv_loader.ntcs))

        self.ntc_loss = nn.CrossEntropyLoss()
    
    def forward(self, input: TensorDict):
        encoder_output = self.encoder(input["sequence"], input.get("lengths", None))
        encoder_output = encoder_output[:, 1:, :] # there is one less NtC than nucleotides
        decoder_output = self.ntc_decoder(encoder_output)
        return {
            "NtC": torch.swapaxes(decoder_output, -1, -2)
        }

    def loss(self, output: TensorDict, target):
        # print(output["NtC"].shape)
        # print(target["NtC"].shape)
        loss = self.ntc_loss(output["NtC"], target["NtC"])
        return loss
