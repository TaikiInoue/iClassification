import icls.types as T
import torch.nn as nn
from icls.models import Builder


class FirstConv(nn.Module, Builder):

    conv_bn_relu: T.Module
    maxpool: T.Module

    def __init__(self, object_cfg: T.ListConfig) -> None:

        """
        Args:
            object_cfg (T.ListConfig):
                - conv_bn_relu: icls.blocks - ConvBnReLU
                - maxpool: torch.nn - MaxPool2d
        """

        super(FirstConv, self).__init__()
        self.build_blocks(object_cfg)

    def forward(self, x: T.Tensor) -> T.Tensor:

        x = self.conv_bn_relu(x)
        x = self.maxpool(x)
        return x
