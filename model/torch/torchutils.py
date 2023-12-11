
import math
import os
import time
from typing import Any, Callable, Dict, Literal, Optional, TypeVar, Union
import numpy as np
import torch, torch.nn as nn, torch.nn.functional as F
import torch.jit

def to_torch(x) -> Any | torch.Tensor:
    if x is None:
        return None
    if isinstance(x, torch.Tensor):
        return x
    if isinstance(x, dict):
        return { k: to_torch(v) for k, v in x.items() }
    if isinstance(x, np.ndarray):
        return torch.from_numpy(x)
    if hasattr(x, "numpy"):
        return torch.from_numpy(x.numpy())
    return torch.tensor(x)

# device = torch.device("cpu")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
def to_device(x, d=device) -> Any | torch.Tensor:
    if x is None:
        return None
    if isinstance(x, torch.Tensor):
        return x.to(d)
    if isinstance(x, dict):
        return { k: to_device(v, d) for k, v in x.items() }
    
    return to_device(to_torch(x), d)
def to_cpu(x) -> Any | torch.Tensor:
    if x is None:
        return None
    if isinstance(x, torch.Tensor):
        return x.cpu()
    if isinstance(x, dict):
        return { k: to_cpu(v) for k, v in x.items() }
    return to_torch(x)

def pad_nested(x: torch.Tensor, padding = 0) -> torch.Tensor:
    if x.is_nested:
        return torch.nested.to_padded_tensor(x, padding)
    return x

TensorDict = Dict[str, torch.Tensor]

if os.environ.get("NTCNET_EAGER", "0") == "1":
    MaybeScriptModule = torch.nn.Module
    T = TypeVar("T")
    def maybe_jit_script(obj: T, optimize = None, _frames_up = 0, _rcb = None, example_inputs=None) -> T:
        return obj
    TModel = TypeVar("TModel", bound=Optional[Callable])
    def maybe_compile(model, *,
            fullgraph: bool = False,
            dynamic: bool = False,
            backend: Union[str, Callable] = "inductor",
            mode: Union[str, None] = None,
            options: Optional[Dict[str, Union[str, int, bool]]] = None,
            disable: bool = False):
        return model
else:
    MaybeScriptModule: Any = torch.jit.ScriptModule # type: ignore
    maybe_jit_script = torch.jit.script # type: ignore
    maybe_compile = torch.compile # type: ignore

MaybeScriptModule: type[nn.Module]

# stolen from https://pytorch.org/tutorials/beginner/transformer_tutorial.html
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

def conv_nd(dim):
    if dim == 1:
        return nn.Conv1d
    if dim == 2:
        return nn.Conv2d
    if dim == 3:
        return nn.Conv3d
    raise Exception("")

class ResnetBlock(MaybeScriptModule):
    def __init__(self, in_channels, out_channels = None, window_size = 3, stride=1, dilation = 1) -> None:
        super(ResnetBlock, self).__init__()
        if out_channels is None:
            out_channels = in_channels
        self.in_channels = in_channels
        self.out_channels = out_channels

        self.bn1 = nn.BatchNorm1d(in_channels)
        self.conv1 = nn.Conv1d(in_channels, out_channels, stride=stride, kernel_size=window_size, padding='same', dilation=dilation)
        self.bn2 = nn.BatchNorm1d(out_channels)
        self.conv2 = nn.Conv1d(out_channels, out_channels, stride=1, kernel_size=window_size, bias=False, padding='same', dilation=dilation)

        if stride != 1 or in_channels != out_channels:
            self.conv_bypass = nn.Conv1d(in_channels, out_channels, 1, stride, padding='same', bias=False)
        else:
            self.conv_bypass = None
        # TODO: L2 regularization of conv layers

    def forward(self, input: torch.Tensor):
        x: torch.Tensor = input
        x = self.bn1(x.contiguous())
        x = F.relu(x)
        x = self.conv1(x)
        x = self.bn2(x.contiguous())
        x = self.conv2(x)
        x = F.relu(x)
        x = self.conv2(x)

        if self.conv_bypass is None:
            bypass = input
        else:
            bypass = self.conv_bypass(input)

        return bypass + x


def make_conv(kind: str,
              dim: int,
              in_channels, out_channels, window_size, stride = 1, dilation = 1, bias = True) -> nn.Module:
    if kind == "plain":
        return conv_nd(dim)(in_channels, out_channels, window_size, stride, padding='same', dilation=dilation, bias=bias)
    if kind == "resnet":
        return ResnetBlock(in_channels, out_channels, window_size, stride, dilation)
    raise Exception(f"Unknown conv kind {kind}")


def clamp(v, min_v, max_v):
    return min(max_v, max(min_v, v))

def count_parameters(module:  nn.Module) -> int:
    return sum(p.numel() for p in module.parameters() if p.requires_grad)
