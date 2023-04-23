
from typing import Dict, Literal, Union
import torch, torch.nn as nn, torch.nn.functional as F


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
TensorDict = Dict[str, torch.Tensor]


def conv_nd(dim):
    if dim == 1:
        return nn.Conv1d
    if dim == 2:
        return nn.Conv2d
    if dim == 3:
        return nn.Conv3d
    raise Exception("")

class ResnetBlock(nn.Module):
    def __init__(self, dim, in_channels, out_channels = None, window_size = 3, stride=1, dilation = 1) -> None:
        super(ResnetBlock, self).__init__()
        if out_channels is None:
            out_channels = in_channels
        self.in_channels = in_channels
        self.out_channels = out_channels

        self.bn1 = nn.BatchNorm1d(in_channels)
        self.conv1 = conv_nd(dim)(in_channels, out_channels, stride=stride, kernel_size=window_size, padding='same', dilation=dilation)
        self.bn2 = nn.BatchNorm1d(out_channels)
        self.conv2 = conv_nd(dim)(out_channels, out_channels, stride=1, kernel_size=window_size, bias=False, padding='same', dilation=dilation)

        if stride != 1 or in_channels != out_channels:
            self.conv_bypass = nn.Conv1d(in_channels, out_channels, 1, stride, padding='same', bias=False)
        else:
            self.conv_bypass = None
        # TODO: L2 regularization of conv layers

    def forward(self, input: torch.Tensor):
        x = input
        x = self.bn1(x)
        x = F.relu(x)
        x = self.conv1(x)
        x = self.bn2(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = self.conv2(x)

        if self.conv_bypass is None:
            bypass = input
        else:
            bypass = self.conv_bypass(input)

        return bypass + x


ConvKind = Union[Literal["resnet"], Literal["plain"]]
def make_conv(kind: ConvKind,
              dim: int,
              in_channels, out_channels, window_size, stride = 1, dilation = 1, bias = True) -> nn.Module:
    if kind == "plain":
        return conv_nd(dim)(in_channels, out_channels, window_size, stride, padding='same', dilation=dilation, bias=bias)
    if kind == "resnet":
        return ResnetBlock(dim, in_channels, out_channels, window_size, stride, dilation)


def clamp(v, min_v, max_v):
    return min(max_v, max(min_v, v))

def count_parameters(module:  nn.Module) -> int:
    return sum(p.numel() for p in module.parameters() if p.requires_grad)
