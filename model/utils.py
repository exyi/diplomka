
from typing import Dict
import torch


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
TensorDict = Dict[str, torch.Tensor]
