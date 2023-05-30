from collections import namedtuple
import math
import time
from typing import Any, Dict, List, Optional, Tuple, Union
import tensorflow as tf
layers = tf.keras.layers
import pandas as pd, numpy as np, os, sys
import csv_loader
# import torchtext
import random
from hparams import Hyperparams
import dataset_tf as dataset

TensorDict = Dict[str, tf.Tensor]

class ResnetBlock(layers.Layer):
    def __init__(self, in_channels, out_channels = None, window_size = 3, stride=1, dilation = 1, name=None) -> None:
        super(ResnetBlock, self).__init__(name=name)
        name = self.name
        if out_channels is None:
            out_channels = in_channels
        self.in_channels = in_channels
        self.out_channels = out_channels

        self.bn1 = layers.BatchNormalization(input_shape=(None, in_channels), name=f"{name}/bn1")
        self.conv1 = layers.Conv1D(
            input_shape=(None, in_channels),
            filters=out_channels,
            strides=stride,
            kernel_size=window_size, use_bias=False, padding='same', dilation_rate=dilation,
            name=f"{name}/conv1"
        )
        
        self.bn2 = layers.BatchNormalization(input_shape=(None, out_channels), name=f"{name}/bn2")
        self.conv2 = layers.Conv1D(
            input_shape=(None, out_channels),
            filters=out_channels,
            strides=1, kernel_size=window_size, use_bias=False, padding='same', dilation_rate=1, name=f"{name}/conv2")

        if stride != 1 or in_channels != out_channels:
            self.conv_bypass = layers.Conv1D(
                input_shape=(None, in_channels),
                filters=out_channels,
                kernel_size=1, strides=stride, padding='same', use_bias=False, name=f"{name}/dense_bypass")
        else:
            self.conv_bypass = None
        # TODO: L2 regularization of conv layers

        self.layers = [self.bn1, self.conv1, self.bn2, self.conv2]
        if self.conv_bypass is not None:
            self.layers.append(self.conv_bypass)

    # @tf.Module.with_name_scope
    def call(self, input: tf.Tensor) -> tf.Tensor:
        x: Any = input
        x = self.bn1(x)
        x = tf.nn.relu(x)
        x = self.conv1(x)
        x = self.bn2(x)
        x = self.conv2(x)
        x = tf.nn.relu(x)
        x = self.conv2(x)

        if self.conv_bypass is None:
            bypass = input
        else:
            bypass = self.conv_bypass(input)

        return bypass + x

def swapaxes(tensor, dim1, dim2):
    dim1 = dim1 % len(tensor.shape)
    dim2 = dim2 % len(tensor.shape)
    transpose = list(range(len(tensor.shape)))
    transpose[dim1], transpose[dim2] = transpose[dim2], transpose[dim1]
    return tf.transpose(tensor, transpose)

def pad_start(tensor: tf.RaggedTensor, pad_width: int):
    paddingshape = (tensor.nrows(), pad_width, *tensor.values.shape[1:])
    return tf.concat([tf.zeros(paddingshape, dtype=tensor.dtype), tensor], axis=1)
def ragged_reshape(tensor: tf.RaggedTensor, new_shape: Tuple[int, ...]):
    x = tensor.values
    x = tf.reshape(x, (tf.shape(x)[0], *new_shape))
    return tf.RaggedTensor.from_row_splits(x, tensor.row_splits)

class ConvEncoder(layers.Layer):
    def __init__(self, input_size, channels = [64, 64], window_size = 3, max_dilatation = 1, kind = "resnet", name=None) -> None:
        """
        set of convolutional layers
        """
        super().__init__(name=name)
        self.max_window_size = window_size
        input_sizes = [input_size, *channels]
        dilations = [ round(2 ** (math.log2(max_dilatation) * (i / (max(1, len(channels) - 1))))) for i in range(len(channels))]
        # print("dilations", dilations)
        self.convolutions = tf.keras.Sequential([
            ResnetBlock(in_size, out_size, window_size, dilation=dilation)
            for in_size, out_size, dilation in zip(input_sizes, channels, dilations)
        ])
        # self.convolutions.build(input_shape=(None, None, input_size))

        self.layers = [ *self.convolutions.layers ]

    # @tf.Module.with_name_scope
    def call(self, input: Union[tf.Tensor, tf.RaggedTensor], lengths: Optional[tf.Tensor] = None):
        x: Any = input
        isragged = isinstance(x, tf.RaggedTensor)
        if isragged:
            lengths = x.row_lengths()
            x = x.to_tensor()
            # x = pad_start(x, self.max_window_size - 1)
            # x = x.values
            # x = tf.expand_dims(x, axis=0) # virtual batch to make batchnorm happy

        # x = swapaxes(x, -1, -2)
        x = self.convolutions(x)
        # x = swapaxes(x, -1, -2)
        if isragged:
            # x = tf.squeeze(x, axis=0)
            # x = tf.RaggedTensor.from_row_lengths(x, row_lengths=(lengths + (self.max_window_size - 1)))
            # x = x[:, (self.max_window_size - 1):]
            x = tf.RaggedTensor.from_tensor(x, lengths=lengths)
            tf.assert_equal(x.values.shape[-2], input.values.shape[-2])
        return x

class EncoderRNN(layers.Layer):
    def __init__(self,
            embedding: tf.Module,
            embedding_size,
            hidden_size,
            num_layers,
            dropout,
            attention_heads,
            bidirectional=False,
            layer_norm=True,
            name=None
        ):
        super(EncoderRNN, self).__init__(name=name)

        if name is None:
            name = "encoder_rnn"

        self.embedding_size = embedding_size
        self.hidden_size = hidden_size
        self.bidirectional = bidirectional
        self.input_dropout = 0.1
        self.dropout = dropout
        self.embedding = embedding
        self.rnn = []
        for layer_i in range(num_layers):
            # TODO multiple layers
            rnn = layers.LSTM(
                input_shape=(None, embedding_size),
                units=hidden_size,
                return_sequences=True,
                name=f"{name}/rnn{layer_i}"
            )
            if bidirectional:
                self.rnn.append(layers.Bidirectional(rnn, merge_mode="sum", name=f"{name}/birnn{layer_i}"))
            else:
                self.rnn.append(rnn)

        self.layer_norm = None
        if layer_norm:
            self.layer_norm = []
            for layer_i in range(num_layers - 1):
                self.layer_norm.append(layers.LayerNormalization(name=f"{name}/ln{layer_i+1}"))

        self.attn_query, self.attn_value, self.attn = [], [], []
        if attention_heads > 0:
            for layer_i in range(num_layers):
                self.attn_query.append(layers.Dense(hidden_size, name=f"{name}/attn_query{layer_i}"))
                self.attn_value.append(layers.Dense(hidden_size, name=f"{name}/attn_value{layer_i}"))
                self.attn.append(layers.MultiHeadAttention(num_heads=attention_heads, key_dim=self.hidden_size//attention_heads, dropout=dropout, name=f"{name}/attn{layer_i}"))

        self.layers = [ self.embedding, *self.rnn, *(self.layer_norm or []), *self.attn_query, *self.attn_value, *self.attn ]
    def get_config(self):
        return {
            "embedding_size": self.embedding_size,
            "hidden_size": self.hidden_size,
            "bidirectional": self.bidirectional,
            "input_dropout": self.input_dropout,
            "dropout": self.dropout,
            "embedding": self.embedding.get_config(),
            "rnn_layers": len(self.rnn),
            "layer_norm": self.layer_norm is not None
        }

    # @tf.Module.with_name_scope
    def call(self, input: tf.RaggedTensor):
        if self.input_dropout > 0:
            input = tf.nn.dropout(input, rate=self.input_dropout)
        embedded = self.embedding(input)
        tf.assert_equal(input.values.shape[-2], embedded.values.shape[-2])

        embedded = tf.nn.dropout(embedded, rate=self.dropout)

        if len(self.rnn) == 0:
            return embedded

        x = embedded
        for rnn_i in range(len(self.rnn)):
            bypass = x
            if rnn_i > 0 and self.layer_norm is not None:
                x = self.layer_norm[rnn_i-1](x)
            x = self.rnn[rnn_i](x)
            x = tf.nn.dropout(x, rate=self.dropout)
            if rnn_i > 0:
                x = x + bypass

            bypass = x
            if len(self.attn_query) > rnn_i:
                query: tf.RaggedTensor = tf.nn.relu(self.attn_query[rnn_i](x))
                # query = ragged_reshape(query, (4, self.hidden_size//4))
                value: tf.RaggedTensor = tf.nn.relu(self.attn_value[rnn_i](x))
                # value = ragged_reshape(value, (4, self.hidden_size//4))
                x = self.attn[rnn_i](query, value)
                # x = ragged_reshape(x, (self.hidden_size,))
                x = tf.nn.dropout(x, rate=self.dropout)
                x = bypass + x
        return x

class Decoder(layers.Layer):
    def __init__(self, hidden_size, output_size, attention = 0, name=None):
        super(Decoder, self).__init__(name=name)
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.out = layers.Dense(input_shape=(None, hidden_size), units=output_size)

        if attention == 0:
            self.attn = None
        elif attention is None:
            # self-attention TODO
            # self.mh_attention = layers.MultiHeadAttention(
            #     hidden_size, num_heads=1)
            assert False
        else:
            # self.attn = nn.Linear(self.hidden_size, attention)
            # self.attn_combine = nn.Linear(self.hidden_size * 2, self.hidden_size)
            assert False

        self.layers = [ self.out ]

    # @tf.Module.with_name_scope
    def call(self, encoder_outputs: tf.RaggedTensor) -> tf.RaggedTensor:
        output: Any = tf.nn.relu(encoder_outputs)
        output = self.out(output)
        output = tf.nn.softmax(output)
        return output

def clamp(v, min_v, max_v):
    return tf.minimum(max_v, tf.maximum(min_v, v))

class Network(tf.keras.Model):
    INPUT_SIZE_NUCLEOTIDE = int(dataset.NtcDatasetLoader.letters_mapping.vocabulary_size())
    INPUT_SIZE = INPUT_SIZE_NUCLEOTIDE + 1 # +1 UNK token, +1 is_dna
    OUTPUT_NTC_SIZE = int(dataset.NtcDatasetLoader.ntc_mapping.vocabulary_size())
    def __init__(self, p: Hyperparams):
        super(Network, self).__init__()
        self.p = p
        embedding_size = p.conv_channels[-1]
        hidden_size = p.rnn_size if p.rnn_layers > 0 else embedding_size
        if len(p.conv_channels) > 0 or p.conv_window_size > 1:
            embedding = ConvEncoder(Network.INPUT_SIZE, channels=p.conv_channels, window_size=p.conv_window_size, kind=p.conv_kind, max_dilatation=p.conv_dilation)
        else:
            embedding = layers.Embedding(Network.INPUT_SIZE, embedding_size)
        self.encoder = EncoderRNN(embedding, embedding_size, hidden_size, num_layers=p.rnn_layers, dropout=p.rnn_dropout, attention_heads=p.attention_heads, bidirectional=True)
        # self.encoder = EncoderBaseline(Network.INPUT_SIZE, hidden_size)
        self.ntc_decoder = Decoder(hidden_size, self.OUTPUT_NTC_SIZE)

        min_frequency = 100
        max_frequency = max(csv_loader.ntc_frequencies.values())
        clip_min = 0.1
        clip_max = 5.0
        if p.sample_weight == "flat":
            self.ntc_loss_weights = tf.convert_to_tensor([
                    0 if k == "[UNK]" else
                    0.01 if k == "NANT" else
                    1
                    for k in dataset.NtcDatasetLoader.ntc_mapping.get_vocabulary()
                ], dtype=self.compute_dtype)
        elif p.sample_weight == "clip-linear":
            self.ntc_loss_weights = tf.convert_to_tensor([
                    0 if k == "[UNK]" else
                    0.01 if k == "NANT" else
                    clamp(5 / (csv_loader.ntc_frequencies[k] / 20_000), clip_min, clip_max)
                    for k in dataset.NtcDatasetLoader.ntc_mapping.get_vocabulary()
                ], dtype=self.compute_dtype)
        elif p.sample_weight == "linear":
            self.ntc_loss_weights = tf.convert_to_tensor([
                    0 if k == "[UNK]" else
                    0.01 if k == "NANT" else
                    clamp(max_frequency / csv_loader.ntc_frequencies[k] * 0.03, 0.0, 7.0)
                    for k in dataset.NtcDatasetLoader.ntc_mapping.get_vocabulary()])
        elif p.sample_weight == "almostlinear":
            self.ntc_loss_weights = tf.convert_to_tensor([
                    0 if k == "[UNK]" else
                    0.01 if k == "NANT" else
                    clamp((max_frequency / csv_loader.ntc_frequencies[k] * 0.01) ** 0.7, 0.0, 7.0)
                    for k in dataset.NtcDatasetLoader.ntc_mapping.get_vocabulary()])
        elif p.sample_weight == "clip-sqrt":
            self.ntc_loss_weights = tf.convert_to_tensor([
                    0 if k == "[UNK]" else
                    0.01 if k == "NANT" else
                    clamp(5 / tf.math.sqrt(csv_loader.ntc_frequencies[k] / min_frequency), clip_min, clip_max)
                    for k in dataset.NtcDatasetLoader.ntc_mapping.get_vocabulary()
                ], dtype=self.compute_dtype)
        elif p.sample_weight == "sqrtB":
            self.ntc_loss_weights = tf.convert_to_tensor([
                    0 if k == "[UNK]" else
                    0.01 if k == "NANT" else
                    clamp(math.sqrt(max_frequency / csv_loader.ntc_frequencies[k]) * 0.01, 0.0, 20.0)
                    for k in dataset.NtcDatasetLoader.ntc_mapping.get_vocabulary()
                ], dtype=self.compute_dtype)
        elif p.sample_weight == "sqrtB-clip":
            self.ntc_loss_weights = tf.convert_to_tensor([
                    0 if k == "[UNK]" else
                    0.01 if k == "NANT" else
                    clamp(math.sqrt(max_frequency / csv_loader.ntc_frequencies[k]) * 0.03, 0.1, 3.0)
                    for k in dataset.NtcDatasetLoader.ntc_mapping.get_vocabulary()
                ], dtype=self.compute_dtype)

        elif p.sample_weight == "log":
            self.ntc_loss_weights = tf.convert_to_tensor([
                    0 if k == "[UNK]" else
                    0.01 if k == "NANT" else
                    5.0 / tf.math.log(5.0 + csv_loader.ntc_frequencies[k] / min_frequency)
                    for k in dataset.NtcDatasetLoader.ntc_mapping.get_vocabulary()
                ], dtype=self.compute_dtype)
            
        elif p.sample_weight == "ignore-AAs":
            self.ntc_loss_weights = tf.convert_to_tensor([
                    0 if k == "[UNK]" else
                    0.01 if k == "NANT" else
                    0.01 if k in ["AA00", "AA08", "AA04"] else
                    1
                    for k in dataset.NtcDatasetLoader.ntc_mapping.get_vocabulary() ])

        # weight=torch.Tensor([
            #     0.01 if k == "NANT" else
            #     clamp(1 / (v / 20_000), 0.2, 1)
            #     for k, v in csv_loader.ntc_frequencies.items()
            # ]),
        # print("NtC loss weights: ", self.ntc_loss_weights.shape)
        print("NtC loss weights: ", self.ntc_loss_weights)


    def call(self, input: TensorDict, whatever =None):
        # print(input["sequence"].shape, input["is_dna"].shape)
        one_hot_seq = tf.one_hot(input["sequence"], depth=self.INPUT_SIZE_NUCLEOTIDE, dtype=self.compute_dtype)
        # tf.print("Sequence: ", input["sequence"])
        in_tensor = tf.concat([
            one_hot_seq,
            tf.cast(tf.expand_dims(input["is_dna"], -1), self.compute_dtype)
        ], axis=-1)
        tf.assert_equal(in_tensor.shape[-1], Network.INPUT_SIZE)
        tf.assert_equal(in_tensor.shape[-2], None)

        encoder_output: tf.RaggedTensor = self.encoder(in_tensor)
        # tf.print("Input: ", in_tensor.shape, in_tensor.values.shape)
        # tf.print("Encoder output: ", encoder_output.shape, encoder_output.values.shape)
        tf.assert_equal(encoder_output.shape[-1], self.p.rnn_size)
        tf.assert_equal(encoder_output.shape[-2], None)
        tf.assert_equal(encoder_output.values.shape[-2], in_tensor.values.shape[-2])
        encoder_output = encoder_output[:, 1:, :] # there is one less NtC than nucleotides
        decoder_output = self.ntc_decoder(encoder_output)
        assert decoder_output.shape[-1] == self.OUTPUT_NTC_SIZE
        assert decoder_output.shape[-2] == None
        tf.assert_equal(decoder_output.shape[-3], encoder_output.shape[-3])

        # tf.print("Input size: ", input["sequence"].row_lengths())
        # tf.print("Output: ", decoder_output)
        return {
            "NtC": decoder_output,
            # "nearest_NtC": decoder_output,
            # "CANA": 
        }

    def ntcloss(self, target: tf.RaggedTensor, output: tf.RaggedTensor):
        target_onehot = tf.one_hot(target.values, depth=self.OUTPUT_NTC_SIZE, dtype=self.compute_dtype)

        loss = tf.losses.categorical_crossentropy(
            target_onehot,
            output.values,
            label_smoothing=0.1 #self.p.label_smoothing
        )
        # remove out of range indices
        # target_masked = tf.where(target.values < self.OUTPUT_NTC_SIZE, target.values, tf.constant(0, dtype=tf.int64))
        weights = tf.gather(self.ntc_loss_weights, target.values)
        loss = loss * weights
        # assert isinstance(loss, tf.RaggedTensor)

        # return tf.reduce_sum(loss) / tf.reduce_sum(weights)
        return tf.reduce_mean(loss)
