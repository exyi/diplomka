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

def train(
    input_data: TensorDict,
    target_data: TensorDict,
    encoder: EncoderRNN, decoder: Decoder,
    encoder_optimizer, decoder_optimizer,
    criterion):

    encoder_optimizer.zero_grad()
    decoder_optimizer.zero_grad()

    input_sequence = input_data['sequence']
    target_ntc_sequence = target_data['NtC']

    encoder_output = encoder(input_sequence)

    encoder_output = encoder_output[:, 1:, :] # there is one less NtC than nucleotides

    decoder_output = decoder(encoder_output)

    loss = criterion(torch.swapaxes(decoder_output, -1, -2), target_ntc_sequence)

    loss.backward()

    encoder_optimizer.step()
    decoder_optimizer.step()

    return loss.item()

def trainIters(encoder, decoder, n_iters, print_every=10, plot_every=100, learning_rate=0.01):
    start = time.time()
    plot_losses = []
    print_loss_total = 0  # Reset every print_every
    plot_loss_total = 0  # Reset every plot_every

    encoder_optimizer = optim.Adam(encoder.parameters(), lr=learning_rate)
    decoder_optimizer = optim.Adam(decoder.parameters(), lr=learning_rate)
    criterion = nn.CrossEntropyLoss()

    dataset = StructuresDataset("data/sample/", None) # ["2d47_v41C35A23.csv"]
    datasetLoader = DataLoader(dataset, shuffle=True)

    iter = 1
    for epoch_i in range(2000):
        for input_tensor, target_tensor in datasetLoader:
            # print(input_tensor, target_tensor)
            loss = train(input_tensor, target_tensor, encoder,
                         decoder, encoder_optimizer, decoder_optimizer, criterion)
            print_loss_total += loss
            plot_loss_total += loss

            if iter % print_every == 0:
                print_loss_avg = print_loss_total / print_every
                print_loss_total = 0
                print('%s (%d %d%%) %.4f' % (time.time() - start,
                                            iter, iter / n_iters * 100, print_loss_avg))

            if iter % plot_every == 0:
                plot_loss_avg = plot_loss_total / plot_every
                plot_losses.append(plot_loss_avg)
                plot_loss_total = 0

            iter += 1

    # showPlot(plot_losses)

def asMinutes(s):
    m = math.floor(s / 60)
    s -= m * 60
    return f'{m}m {s}s'

def timeSince(since, percent):
    now = time.time()
    s = now - since
    es = s / (percent)
    rs = es - s
    return f'{asMinutes(s)} (- {asMinutes(rs)})'

if __name__ == "__main__":
    trainIters(EncoderRNN(input_size=len(csv_loader.basic_nucleotides), embedding_size=123, hidden_size=124), Decoder(124, len(csv_loader.ntcs)), 100000)
