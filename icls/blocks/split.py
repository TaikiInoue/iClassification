import torch
from torch.nn import Module
from torch import Tensor


class Split(Module):
    def __init__(
        self,
        channels: int = None,
        num_chunks: int = None,
        dim: int = None,
    ):

        super(Split, self).__init__()

        self.channels = channels
        self.num_chunks = num_chunks
        self.dim = dim

    def forward(self, x: Tensor) -> Tensor:

        split_size_or_sections = self.channels // self.num_chunks
        x = torch.split(x, split_size_or_sections, self.dim)
        return x
