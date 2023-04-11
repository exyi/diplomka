import math
import time
from typing import Dict, List, Optional, Tuple
import torch, torch.nn as nn, torch.nn.functional as F, torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pandas as pd, numpy as np, os, sys
import csv_loader
# import torchtext
import random

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class StructuresDataset(Dataset):
    def __init__(self, dir: str, files: Optional[List[str]]):
        self.dir = dir
        if files is None:
            self.files = [ d for d in os.listdir(dir) if d.endswith('.csv') ]
        else:
            self.files = files

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        img_path = os.path.join(self.dir, self.files[idx])
        table, chains = csv_loader.load_csv_file(img_path)
        joined = csv_loader.get_joined_arrays(chains)
        input = { "sequence": csv_loader.encode_nucleotides(joined['sequence']), "is_dna": joined['is_dna'] }
        target = {
            "NtC": csv_loader.encode_ntcs(joined['NtC']),
            # "CANA": joined['CANA']
        }
        return input, target

class EncoderRNN(nn.Module):
    def __init__(self, input_size, embedding_size, hidden_size):
        super(EncoderRNN, self).__init__()
        self.embedding_size = embedding_size
        self.hidden_size = hidden_size

        self.embedding = nn.Embedding(input_size, embedding_size)
        self.rnn = nn.LSTM(embedding_size, hidden_size)

    def forward(self, input):
        embedded = self.embedding(input)
        output, _ = self.rnn(embedded)
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

teacher_forcing_ratio = 0.5

def train(
    input_data: Dict[str, torch.Tensor],
    target_data: Dict[str, torch.Tensor],
    encoder: EncoderRNN, decoder: Decoder,
    encoder_optimizer, decoder_optimizer,
    criterion):

    encoder_optimizer.zero_grad()
    decoder_optimizer.zero_grad()

    input_sequence = input_data['sequence']
    target_ntc_sequence = target_data['NtC']
    # print(input_sequence.shape, target_ntc_sequence.shape)

    encoder_output = encoder(input_sequence)

    encoder_output = encoder_output[:, 1:, :] # there is one less nucleotide than nucleotide

    decoder_output = decoder(encoder_output)

    # print(f"{decoder_output.shape=} {target_ntc_sequence.shape=}")
    loss = criterion(torch.swapaxes(decoder_output, -1, -2), target_ntc_sequence)
    # loss = criterion(decoder_output, F.one_hot(target_ntc_sequence, num_classes=decoder.output_size))

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

    dataset = StructuresDataset("../data/sample/", None) # ["2d47_v41C35A23.csv"]
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
