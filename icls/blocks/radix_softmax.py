from torch.nn import Module
import torch.nn.functional as F
from torch import Tensor


class RadixSoftmax(Module):
    def __init__(
        self,
        radix: int = None,
        cardinality: int = None,
    ):

        super(RadixSoftmax, self).__init__()

        self.radix = radix
        self.cardinality = cardinality

    def forward(self, x: Tensor) -> Tensor:

        batch, channel, height, width = x.shape
        x = x.view(batch, self.cardinality, self.radix, -1)
        x = F.softmax(x, dim=1)
        x = x.reshape(batch, -1)
        return x
