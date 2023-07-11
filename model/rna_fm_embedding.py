
# class BatchConverter(object):
#     """Callable to convert an unprocessed (labels + strings) batch to a
#     processed (labels + tensor) batch.
#     """

#     def __init__(self, alphabet):
#         self.alphabet = alphabet

#     def __call__(self, raw_batch: Sequence[Tuple[str, str]]):
#         # RoBERTa uses an eos token, while ESM-1 does not.
#         batch_size = len(raw_batch)
#         max_len = max(len(seq_str) for _, seq_str in raw_batch)
#         tokens = torch.empty(
#             (
#                 batch_size,
#                 max_len
#                 + int(self.alphabet.prepend_bos)
#                 + int(self.alphabet.append_eos),
#             ),
#             dtype=torch.int64,
#         )
#         tokens.fill_(self.alphabet.padding_idx)
#         labels = []
#         strs = []

#         for i, (label, seq_str) in enumerate(raw_batch):
#             labels.append(label)
#             strs.append(seq_str)
#             if self.alphabet.prepend_bos:
#                 tokens[i, 0] = self.alphabet.cls_idx
#             seq = torch.tensor(
#                 [self.alphabet.get_idx(s) for s in seq_str], dtype=torch.int64
#             )
#             tokens[
#                 i,
#                 int(self.alphabet.prepend_bos) : len(seq_str)
#                 + int(self.alphabet.prepend_bos),
#             ] = seq
#             if self.alphabet.append_eos:
#                 tokens[
#                     i, len(seq_str) + int(self.alphabet.prepend_bos)
#                 ] = self.alphabet.eos_idx

#         return labels, strs, tokens

from typing import Any, Dict, List
import os, sys, json, dataclasses, numpy as np
from dataclasses import dataclass

@dataclass
class AlphabetDefinition:
    tok_to_idx: Dict[str, int]
    padding_idx: int
    append_eos: bool
    prepend_bos: bool

class RnaFMOnnxRuntimeEmbedding:
    def __init__(self, file, our_alphabet: List[str]) -> None:
        import onnxruntime as ort
        sess_opt = ort.SessionOptions()
        sess_opt.execution_mode = ort.ExecutionMode.ORT_PARALLEL
        sess_opt.intra_op_num_threads = 16
        sess_opt.inter_op_num_threads = 16
        self.sess = ort.InferenceSession(file, sess_opt)
        self.alphabet = get_alphabet(file)
        self.our_alphabet = our_alphabet
        self.output_dim = self.sess.get_outputs()[0].shape[-1]

        self.alphabet_translation = alphabet_translation_table(self.alphabet, our_alphabet)

        print(f"Initialized RnaFMOnnxRuntimeEmbedding, {self.output_dim=}, {self.alphabet_translation=}")

    def __call__(self, lengths: np.ndarray, input: np.ndarray, is_dna: np.ndarray) -> Any:
        assert input.shape[0] == lengths.shape[0]

        einput = np.zeros((input.shape[0], input.shape[1] + int(self.alphabet.append_eos) + int(self.alphabet.append_eos)), dtype=np.int64)
        einput.fill(self.alphabet.tok_to_idx["<pad>"])
        if self.alphabet.prepend_bos:
            einput[:, 0] = self.alphabet.tok_to_idx["<cls>"]

        for line_i, line_len in enumerate(lengths):
            if self.alphabet.append_eos:
                einput[line_i, line_len + int(self.alphabet.prepend_bos)] = self.alphabet.tok_to_idx["<eos>"]

            np.choose(input[line_i, 0:line_len], self.alphabet_translation, out=einput[line_i, int(self.alphabet.prepend_bos):line_len + int(self.alphabet.prepend_bos)])

        # print(input)
        # print(einput.shape)
        # print(einput)

        output = self.sess.run(None, { "input": einput })[0]

        trimmed_out = output[:, int(self.alphabet.prepend_bos):output.shape[1] - int(self.alphabet.append_eos)]
        assert trimmed_out.shape[0] == input.shape[0], f"{trimmed_out.shape} != {(input.shape[0], input.shape[1], self.output_dim)}"
        assert trimmed_out.shape[1] == input.shape[1], f"{trimmed_out.shape} != {(input.shape[0], input.shape[1], self.output_dim)}"
        assert trimmed_out.shape[2] == self.output_dim, f"{trimmed_out.shape} != {(input.shape[0], input.shape[1], self.output_dim)}"
        return trimmed_out
    


import tensorflow as tf
class RnaFMOnnxTFEmbedding(tf.keras.layers.Layer):
    def __init__(self, file, our_alphabet: List[str]) -> None:
        super().__init__(name="RnaFMOnnxTFEmbedding")
        import onnx
        import onnx_tf
        onnx_model = onnx.load(file)
        tf_repr = onnx_tf.backend.prepare(onnx_model)
        self.tf_module = tf_repr.tf_module
        self.alphabet = get_alphabet(file)
        self.our_alphabet = our_alphabet
        self.output_dim = 640 # ?? self.sess.get_outputs()[0].shape[-1]

        self.alphabet_translation = alphabet_translation_table(self.alphabet, our_alphabet)

        print(f"Initialized RnaFMOnnxTFEmbedding, {self.output_dim=}, {self.alphabet_translation=}")

    def call(self, input: tf.RaggedTensor, is_dna: tf.RaggedTensor) -> Any:
        batch_size = input.nrows()
        tinput: tf.RaggedTensor = tf.gather(self.alphabet_translation, input)
        print(self.alphabet)
        print(self.alphabet_translation)
        if self.alphabet.prepend_bos:
            tinput = tf.concat([tf.fill([batch_size, 1], tf.constant(self.alphabet.tok_to_idx["<cls>"], dtype=tf.int64)), tinput], axis=1)
        if self.alphabet.append_eos:
            tinput = tf.concat([tinput, tf.fill([batch_size, 1], tf.constant(self.alphabet.tok_to_idx["<eos>"], dtype=tf.int64))], axis=1)

        einput = tinput.to_tensor(default_value=self.alphabet.tok_to_idx["<pad>"])

        output = self.tf_module(input=einput)['output']
        output = tf.ensure_shape(output, (None, None, self.output_dim))
        assert output.shape[-1] == self.output_dim, f"{output.shape}"
        trimmed_out = output
        if self.alphabet.prepend_bos:
            trimmed_out = trimmed_out[:, 1:]
        
        routput = tf.RaggedTensor.from_tensor(trimmed_out, lengths=input.row_lengths())
        assert routput.shape[-1] == self.output_dim, f"{routput.shape}"
        return routput

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
