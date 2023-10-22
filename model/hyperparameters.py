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
    max_batch_size: int = hyperparameter(None, "Maximum batch size, limits batch_size adjustion when seq_length low")
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
    sample_weight: str = hyperparameter("flat", "Sample weight mode", choices=["flat", "flat+helix", "log", "log+helix", "clip-sqrt", "sqrt", "sqrtB", "sqrtB+helix", "clip-linear", "linear", "almostlinear", "almostlinear+helix", "sqrtB-clip", "ignore-AAs", "one", "one+helix"])
    basepairing: str = hyperparameter("none", "What to do with basepairing information. Nothing / Input by directly connecting the basepaired nucleotides", choices=["none", "input-directconn", "input-conv"])
    outputs: List[str] = hyperparameter(("NtC",), "What to predict - NtC, CANA, geometry or combination", list=True, choices=["NtC", "CANA", "geometry"])
    external_embedding: str = hyperparameter(None, "path to external ONNX embedding model (rna-fm.onnx)")
    nearest_ntc_threshold: float = hyperparameter(0.0, "RMSD Threshold for accepting nearest NtC as the target label; not applied to validation data")

    def get_nondefault(self):
        default = dataclasses.asdict(Hyperparams())
        return {k: v for k, v in dataclasses.asdict(self).items() if default[k] != v}
    
    @staticmethod
    def from_args(args):
        return Hyperparams(**{ k: v for k, v in vars(args).items() if k in Hyperparams.__dataclass_fields__ })


def add_parser_args(parser, dataclass = Hyperparams):
    for k, w in dataclass.__dataclass_fields__.items():
        p_config = {}
        if w.metadata.get("list", False):
            p_config["nargs"] = "+"
            p_config["type"] = w.type.__args__[0]
        elif dataclasses.MISSING != w.type:
            p_config["type"] = w.type
        else: 
            p_config["type"] = type(w.default)
        if dataclasses.MISSING != w.default:
            p_config["default"] = w.default
        else:
            p_config["required"] = True
        
        if "help" in w.metadata:
            p_config["help"] = w.metadata["help"]

        if "choices" in w.metadata:
            p_config["choices"] = w.metadata["choices"]
        
        parser.add_argument(f'--{k}', **p_config)
