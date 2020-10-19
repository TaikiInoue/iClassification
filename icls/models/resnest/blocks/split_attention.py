from operator import mul

from torch.nn import Module
from icls.models import Builder
from omegaconf import ListConfig
from torch import Tensor


class SplitAttention(Module, Builder):

    split: Module
    avgpool: Module
    conv11_bn_relu: Module
    conv11: Module
    radix_softmax: Module

    def __init__(self, cfg: ListConfig) -> None:

        """
        Args:
            cfg (ListConfig):
                - split: icls.blocks - Split
                - avgpool: torch.nn - AdaptiveAvgPool2d
                - conv11_bn_relu: icls.blocks - Conv11BnReLU
                - conv11: icls.blocks - Conv11
                - radix_softmax: icls.blocks - RadixSoftmax
        """

        super(SplitAttention, self).__init__()
        self.build_blocks_from_cfg(cfg)

    def forward(self, x: Tensor) -> Tensor:

        batches, channels, height, width = x.shape
        self.split.channels = channels

        x_splited = self.split(x)
        attention = sum(x_splited)
        attention = self.avgpool(attention)
        attention = self.conv11_bn_relu(attention)
        attention = self.conv11(attention)
        attention = self.radix_softmax(attention)
        attention = attention.view(batches, -1, 1, 1)
        attention_splited = self.split(attention)

        out = map(mul, x_splited, attention_splited)
        out = sum(out)
        return out.contiguous()
