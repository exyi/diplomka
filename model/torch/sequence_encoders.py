from typing import Optional
import math
import torch, torch.nn as nn, torch.nn.functional as F
from .torchutils import TensorDict, clamp, device, make_conv, ResnetBlock


# @torch.jit.script
# def sin_position_embedding(max_len: int, lengths: torch.LongTensor, output_dim: int, dtype: torch.dtype=torch.float32):

#     dim_half = 16
#     assert output_dim >= 2 * dim_half
#     column_indices = torch.arange(0, max_len, dtype=dtype)
#     column_indices = column_indices.unsqueeze(0).expand(dim_half, -1)
#     print(column_indices.shape)
#     power_range = torch.arange(dim_half, dtype=dtype)
#     power_range = power_range.unsqueeze(-1).expand(-1, max_len)
#     print(power_range.shape)
#     im = torch.sin(column_indices * math.pi / 2 ** power_range)
#     re = torch.cos(column_indices * math.pi / 2 ** power_range)
#     padding = column_indices * torch.zeros(output_dim - 2 * dim_half, dtype=dtype)
#     return torch.concatenate([im, re, padding], dim=-1)

class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Arguments:
            x: Tensor, shape ``[seq_len, batch_size, embedding_dim]``
        """
        x = x + self.pe[:x.size(0)]
        return self.dropout(x)

class ConvEncoder(nn.Module):
    def __init__(self, input_size, channels = [64, 64], window_size = 3, kind: str = "resnet", max_dilatation = 1) -> None:
        """
        set of convolutional layers
        """
        super().__init__()
        input_sizes = [input_size, *channels]
        dilations = [ round(2 ** (math.log2(max_dilatation) * (i / (max(1, len(channels) - 1))))) for i in range(len(channels))]
        if max_dilatation != 1:
            print(f"Using ConvEncoder dilations: {dilations}")
        assert dilations[-1] == max_dilatation
        self.convolutions = nn.Sequential(*[
            make_conv(kind, 1, in_size, out_size, window_size, dilation=dilation)
            for in_size, out_size, dilation in zip(input_sizes, channels, dilations)
        ])

    def forward(self, input: torch.Tensor, lengths: Optional[torch.LongTensor] = None):
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
        if num_layers > 0:
            self.rnn = nn.LSTM(embedding_size, hidden_size, bidirectional=bidirectional, num_layers=num_layers, dropout=dropout)
            self.rnn.flatten_parameters()
        else:
            self.rnn = None
        self.output_dropout = nn.Dropout(dropout)

    def forward(self, input: torch.Tensor, lengths: Optional[torch.LongTensor] = None):
        if self.embedded_dropout.p > 0:
            input = self.embedded_dropout(input)
        embedded = self.embedding(input)
        embedded = self.embedded_dropout(embedded)

        if self.rnn is None:
            return embedded

        if lengths is not None:
            packed = torch.nn.utils.rnn.pack_padded_sequence(embedded, lengths, batch_first=True)
            output_packed, _ = self.rnn(packed)
            output, _ = torch.nn.utils.rnn.pad_packed_sequence(output_packed, batch_first=True)
        else:
            output, _ = self.rnn(embedded)
        if self.bidirectional:
            output = output[:, :, :self.hidden_size] + output[:, :, self.hidden_size:]
        output = self.output_dropout(output)
        return output

