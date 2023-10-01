import os

from model.tf import rna_fm_embedding
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
from model import dataset_tf
import tensorflow as tf

onnxfile = "/home/exyi/tmp/rna-fm-venv/rna-fm.onnx"
em = rna_fm_embedding.RnaFMOnnxTFEmbedding(onnxfile, dataset_tf.NtcDatasetLoader.letters_mapping.get_vocabulary())
em_rt = rna_fm_embedding.RnaFMOnnxRuntimeEmbedding(onnxfile, dataset_tf.NtcDatasetLoader.letters_mapping.get_vocabulary())
print(em_rt.sess.run(None, { 'input':  [[0, 7, 6, 5, 7, 5, 5, 7, 4, 6, 7, 4, 5, 6, 4, 6, 4, 6, 6, 4, 4, 5, 6, 6, 4, 6, 7, 6, 2]] }))

print("---");

seq = "UGCUCCUAGUACGAGAGGAACGGAGUG"
seq_ix = dataset_tf.NtcDatasetLoader.letters_mapping(list(seq))
seq_ix = tf.RaggedTensor.from_row_lengths(seq_ix, [len(seq_ix)])
seq_em = em(seq_ix, None)
seq_em_rt = em_rt(seq_ix.row_lengths(), seq_ix.to_tensor(), None)

print(tf.shape(seq_em))
print(seq)
tf.print(seq_ix)
tf.print(seq_em)
tf.print(seq_em_rt)

