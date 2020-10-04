import icls.types as T
import torch
import torch.nn as nn


class Split(nn.Module):
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

    def forward(self, x: T.Tensor) -> T.Tensor:

        split_size_or_sections = self.channels // self.num_chunks
        x = torch.split(x, split_size_or_sections, self.dim)
        return x
