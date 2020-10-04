import icls.types as T
import torch.nn as nn
from icls.models import Builder


class FirstConv(nn.Module, Builder):

    conv33_bn_relu_0: T.Module
    conv33_bn_relu_1: T.Module
    conv33_bn_relu_2: T.Module
    maxpool: T.Module

    def __init__(self, object_cfg: T.ListConfig) -> None:

        """
        Args:
            object_cfg (T.ListConfig):
                - conv33_bn_relu_0: icls.blocks - Conv33BnReLU
                - conv33_bn_relu_1: icls.blocks - Conv33BnReLU
                - conv33_bn_relu_2: icls.blocks - Conv33BnReLU
                - maxpool: torch.nn - MaxPool2d
        """

        super(FirstConv, self).__init__()
        self.build_blocks(object_cfg)

    def forward(self, x: T.Tensor) -> T.Tensor:

        x = self.conv33_bn_relu_0(x)
        x = self.conv33_bn_relu_1(x)
        x = self.conv33_bn_relu_2(x)
        x = self.maxpool(x)
        return x
