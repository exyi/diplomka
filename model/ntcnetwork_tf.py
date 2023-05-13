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
        if out_channels is None:
            out_channels = in_channels
        self.in_channels = in_channels
        self.out_channels = out_channels

        self.bn1 = layers.BatchNormalization(input_shape=(None, in_channels))
        self.conv1 = layers.Conv1D(
            input_shape=(None, in_channels),
            filters=out_channels,
            strides=stride,
            kernel_size=window_size, use_bias=False, padding='same', dilation_rate=dilation,
        )
        
        self.bn2 = layers.BatchNormalization(input_shape=(None, out_channels))
        self.conv2 = layers.Conv1D(
            input_shape=(None, out_channels),
            filters=out_channels,
            strides=1, kernel_size=window_size, use_bias=False, padding='same', dilation_rate=dilation)

        if stride != 1 or in_channels != out_channels:
            self.conv_bypass = layers.Conv1D(
                input_shape=(None, in_channels),
                filters=out_channels,
                kernel_size=1, strides=stride, padding='same', use_bias=False)
        else:
            self.conv_bypass = None
        # TODO: L2 regularization of conv layers

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

class ConvEncoder(layers.Layer):
    def __init__(self, input_size, channels = [64, 64], window_size = 3, kind = "resnet", name=None) -> None:
        """
        set of convolutional layers
        """
        super().__init__(name=name)
        self.max_window_size = window_size
        input_sizes = [input_size, *channels]
        self.convolutions = tf.keras.Sequential([
            ResnetBlock(in_size, out_size, window_size)
            for in_size, out_size in zip(input_sizes, channels)
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
            bidirectional=False,
            name=None
        ):
        super(EncoderRNN, self).__init__(name=name)

        self.embedding_size = embedding_size
        self.hidden_size = hidden_size
        self.bidirectional = bidirectional
        self.input_dropout = 0.1
        self.dropout = dropout
        self.embedding = embedding
        if num_layers > 0:
            # TODO multiple layers
            self.rnn = layers.LSTM(
                input_shape=(None, embedding_size),
                units=hidden_size,
                return_sequences=True)
            if bidirectional:
                self.rnn = layers.Bidirectional(self.rnn, merge_mode="sum")
        else:
            self.rnn = None

        self.layers = [ self.embedding, self.rnn ]

    # @tf.Module.with_name_scope
    def call(self, input: tf.RaggedTensor):
        if self.input_dropout > 0:
            input = tf.nn.dropout(input, rate=self.input_dropout)
        embedded = self.embedding(input)
        tf.assert_equal(input.values.shape[-2], embedded.values.shape[-2])

        embedded = tf.nn.dropout(embedded, rate=self.dropout)

        if self.rnn is None:
            return embedded

        output = self.rnn(embedded)
        output = tf.nn.dropout(output, rate=self.dropout)
        return output

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

class Network(tf.keras.Model):
    INPUT_SIZE = dataset.NtcDatasetLoader.letters_mapping.vocabulary_size() + 1 # +1 UNK token, +1 is_dna
    OUTPUT_NTC_SIZE = dataset.NtcDatasetLoader.ntc_mapping.vocabulary_size()
    def __init__(self, p: Hyperparams):
        super(Network, self).__init__()
        self.p = p
        embedding_size = p.conv_channels[-1]
        hidden_size = p.rnn_size if p.rnn_layers > 0 else embedding_size
        if len(p.conv_channels) > 0 or p.conv_window_size > 1:
            embedding = ConvEncoder(Network.INPUT_SIZE, channels=p.conv_channels, window_size=p.conv_window_size, kind=p.conv_kind)
        else:
            embedding = layers.Embedding(Network.INPUT_SIZE, embedding_size)
        self.encoder = EncoderRNN(embedding, embedding_size, hidden_size, num_layers=p.rnn_layers, dropout=p.rnn_dropout, bidirectional=True)
        # self.encoder = EncoderBaseline(Network.INPUT_SIZE, hidden_size)
        self.ntc_decoder = Decoder(hidden_size, self.OUTPUT_NTC_SIZE)

        self.ntc_loss_weights = tf.convert_to_tensor([
                0 if k == "[UNK]" else
                0.01 if k == "NANT" else
                1
                for k in dataset.NtcDatasetLoader.ntc_mapping.get_vocabulary()
            ], dtype=self.compute_dtype)
        # weight=torch.Tensor([
            #     0.01 if k == "NANT" else
            #     clamp(1 / (v / 20_000), 0.2, 1)
            #     for k, v in csv_loader.ntc_frequencies.items()
            # ]),
        # print("NtC loss weights: ", self.ntc_loss_weights.shape)
        # print("NtC loss weights: ", self.ntc_loss_weights)


    def call(self, input: TensorDict, whatever =None):
        # print(input["sequence"].shape, input["is_dna"].shape)
        one_hot_seq = tf.one_hot(input["sequence"], depth=dataset.NtcDatasetLoader.letters_mapping.vocabulary_size(), dtype=self.compute_dtype)
        # tf.print("Sequence: ", input["sequence"])
        in_tensor = tf.concat([
            one_hot_seq,
            tf.cast(tf.expand_dims(input["is_dna"], -1), self.compute_dtype)
        ], axis=-1)
        assert in_tensor.shape[-1] == Network.INPUT_SIZE
        assert in_tensor.shape[-2] == None

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
