import dataclasses
from dataclasses import dataclass, field
from typing import List

def hyperparameter(default, help, **kwargs):
    return field(default=default, metadata={"hyperparameter": True, "help": help, **kwargs})

@dataclass()
class Hyperparams:
    # seq_length_limit: int = hyperparameter(512, "Maximum length of a sequence to train on, longer sequences will be randomly truncated")
    seq_length_schedule: str = hyperparameter("x*512", "Maximum length of a sequence to train on, longer sequences will be randomly truncated. Format is `((x|\\d+)\\*\\d+;)*`")
    learning_rate: float = hyperparameter(0.001, "ADAM Learning rate")
    lr_decay: str = hyperparameter("cosine", "Learning rate decay")
    max_batch_size: int = hyperparameter(32, "Maximum batch size, limits batch_size adjustion when seq_length low")
    batch_size: int = hyperparameter(64, "Training batch size given minimal seq_length. When seq_len is increased, batch size is decreased accordingly.")
    epochs: int = hyperparameter(30, "Number of epochs to train")
    conv_channels: List[int] = hyperparameter((64, 64, 128, 128), "Number of channels in each convolutional layer", list=True)
    conv_window_size: int = hyperparameter(11, "Size of convolutional window")
    conv_kind: str = hyperparameter("resnet", "Type of convolutional layer to use", choices=["resnet", "plain"])
    conv_dilation: int = hyperparameter(1, "Max dilatation of convolutional layers.")
    rnn_size: int = hyperparameter(256, "Size of hidden state in RNN layer")
    rnn_layers: int = hyperparameter(1, "Number of RNN layers")
    rnn_dropout: float = hyperparameter(0.4, "Dropout rate in RNN layers")
    clip_grad: float = hyperparameter(0.1, "Gradient clipping (see ADAM global_clipnorm argument)")
    clip_grad_value: float = hyperparameter(None, "Another method of gradient clipping (see ADAM clipvalue argument)")
    attention_heads: int = hyperparameter(0, "Number of attention heads. If != 0 multihead attention is inserted after each RNN layer")
    sample_weight: str = hyperparameter("flat", "Sample weight mode", choices=["flat", "log", "clip-sqrt", "sqrt", "sqrtB", "clip-linear", "linear", "almostlinear", "sqrtB-clip", "ignore-AAs"])
    basepairing: str = hyperparameter("none", "What to do with basepairing information. Nothing / Input by directly connecting the basepaired nucleotides", choices=["none", "input-directconn"])

    def get_nondefault(self):
        default = dataclasses.asdict(Hyperparams())
        return {k: v for k, v in dataclasses.asdict(self).items() if default[k] != v}


