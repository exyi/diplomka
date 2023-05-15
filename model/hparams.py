import dataclasses
from dataclasses import dataclass, field
from typing import List

def hyperparameter(default, help, **kwargs):
    return field(default=default, metadata={"hyperparameter": True, "help": help, **kwargs})

@dataclass
class Hyperparams:
    seq_length_limit: int = hyperparameter(512, "Maximum length of a sequence to train on, longer sequences will be randomly truncated")
    learning_rate: float = hyperparameter(0.001, "ADAM Learning rate")
    lr_decay: str = hyperparameter("cosine", "Learning rate decay")
    batch_size: int = hyperparameter(32, "Training batch size")
    epochs: int = hyperparameter(30, "Number of epochs to train")
    conv_channels: List[int] = hyperparameter((64, 64), "Number of channels in each convolutional layer", list=True)
    conv_window_size: int = hyperparameter(11, "Size of convolutional window")
    conv_kind: str = hyperparameter("resnet", "Type of convolutional layer to use", choices=["resnet", "plain"])
    conv_dilation: int = hyperparameter(1, "Max dilatation of convolutional layers.")
    rnn_size: int = hyperparameter(64, "Size of hidden state in RNN layer")
    rnn_layers: int = hyperparameter(1, "Number of RNN layers")
    rnn_dropout: float = hyperparameter(0.4, "Dropout rate in RNN layers")
    clip_grad: float = hyperparameter(None, "Gradient clipping (see ADAM global_clipnorm argument)")
    attention_heads: int = hyperparameter(4, "Number of attention heads. If != 0 multihead attention is inserted after each RNN layer")


