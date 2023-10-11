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
            embedding_size,
            hidden_size,
            num_layers,
            dropout,
            attention_heads,
            pairing_mode="none",
            layer_norm=False,
            bidirectional=False # TODO
        ):
        super(EncoderRNN, self).__init__()

        self.hidden_size = hidden_size
        self.bidirectional = bidirectional
        self.embedded_dropout = nn.Dropout(dropout)
        self.rnn_dropout = nn.Dropout(dropout)

        self.rnn = nn.ModuleList()
        for layer_i in range(num_layers):
            rnn = nn.LSTM(
                embedding_size if layer_i == 0 else hidden_size,
                hidden_size, bidirectional=bidirectional, num_layers=1
            ).to(device)
            rnn.flatten_parameters()
            self.rnn.append(rnn)

        if pairing_mode == "input-directconn":
            self.pairing_reducer = [
                nn.Linear(
                    2 * (hidden_size if i > 0 else embedding_size),
                    hidden_size if i > 0 else embedding_size
                )
                for i in range(num_layers)
            ]

        self.layer_norm = None
        if layer_norm:
            self.layer_norm = nn.ModuleList()
            for layer_i in range(num_layers):
                self.layer_norm.append(nn.LayerNorm(embedding_size if layer_i == 0 else hidden_size))

        self.attn_query, self.attn_value, self.attn = nn.ModuleList(), nn.ModuleList(), nn.ModuleList()
        if attention_heads > 0:
            for layer_i in range(num_layers):
                self.attn_query.append(nn.Linear(hidden_size, hidden_size))
                self.attn_value.append(nn.Linear(hidden_size, hidden_size))
                a = nn.MultiheadAttention(hidden_size, num_heads=attention_heads, dropout=dropout, batch_first=True, device=device)
                self.attn.append(a)

    def call_rnn(self, i, input, lengths):
        if lengths is not None:
            packed = torch.nn.utils.rnn.pack_padded_sequence(input, lengths, batch_first=True)
            output_packed, _ = self.rnn[i](packed)
            output, _ = torch.nn.utils.rnn.pad_packed_sequence(output_packed, batch_first=True)
        else:
            output, _ = self.rnn[i](input)
        if self.bidirectional:
            output = output[:, :, :self.hidden_size] + output[:, :, self.hidden_size:]

        return output


    def forward(self, embedded: torch.Tensor, lengths: Optional[torch.LongTensor] = None):
        return embedded
        embedded = self.embedded_dropout(embedded)

        if len(self.rnn)==0:
            return embedded
        
        x = embedded
        
        for layer_i in range(len(self.rnn)):
            bypass = x
            # TODO position embedding
            # TODO pairs with
            if self.layer_norm is not None:
                x = self.layer_norm[layer_i](x)
            # x = self.call_rnn(layer_i, x, lengths)
            x = self.rnn_dropout(x)

            if layer_i > 0:
                x = x + bypass

            if len(self.attn_query) > layer_i:
                bypass = x
                if self.layer_norm is not None:
                    x = self.layer_norm[layer_i](x)

                query = F.relu(self.attn_query[layer_i](x))
                value = F.relu(self.attn_value[layer_i](x))
                x = self.attn[layer_i](query, query, value)
                x = self.rnn_dropout(x)
                x = bypass + x

        return x

