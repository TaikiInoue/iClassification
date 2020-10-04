from operator import mul

import icls.types as T
import torch.nn as nn
from icls.models import Builder


class SplitAttention(nn.Module, Builder):

    split: T.Module
    avgpool: T.Module
    conv11_bn_relu: T.Module
    conv11: T.Module
    radix_softmax: T.Module

    def __init__(self, object_cfg: T.ListConfig) -> None:

        """
        Args:
            object_cfg (T.ListConfig):
                - split: icls.blocks - Split
                - avgpool: torch.nn - AdaptiveAvgPool2d
                - conv11_bn_relu: icls.blocks - Conv11BnReLU
                - conv11: icls.blocks - Conv11
                - radix_softmax: icls.blocks - RadixSoftmax
        """

        super(SplitAttention, self).__init__()
        self.build_blocks(object_cfg)

    def forward(self, x: T.Tensor) -> T.Tensor:

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
