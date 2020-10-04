import icls.types as T
import torch.nn as nn
import torch.nn.functional as F


class RadixSoftmax(nn.Module):
    def __init__(
        self,
        radix: int = None,
        cardinality: int = None,
    ):

        super(RadixSoftmax, self).__init__()

        self.radix = radix
        self.cardinality = cardinality

    def forward(self, x: T.Tensor) -> T.Tensor:

        batch, channel, height, width = x.shape
        x = x.view(batch, self.cardinality, self.radix, -1)
        x = F.softmax(x, dim=1)
        x = x.reshape(batch, -1)
        return x
