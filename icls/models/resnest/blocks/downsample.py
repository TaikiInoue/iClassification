import icls.types as T
import torch.nn as nn
from icls.models import Builder


class Downsample(nn.Module, Builder):

    avgpool: T.Module
    conv11_bn_relu: T.Module

    def __init__(self, object_cfg: T.ListConfig) -> None:

        """
        Args:
            object_cfg (T.ListConfig):
                - avgpool: torch.nn - AvgPool2d
                - conv11_bn_relu: icls.blocks - Conv11BnReLU
        """

        super(Downsample, self).__init__()
        self.build_blocks(object_cfg)

    def forward(self, x: T.Tensor) -> T.Tensor:

        x = self.avgpool(x)
        x = self.conv11_bn_relu(x)
        return x
