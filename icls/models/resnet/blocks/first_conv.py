from torch.nn import Module
from icls.models import Builder
from omegaconf import ListConfig
from torch import Tensor


class FirstConv(Module, Builder):

    conv_bn_relu: Module
    maxpool: Module

    def __init__(self, cfg: ListConfig) -> None:

        """
        Args:
            cfg (ListConfig):
                - conv_bn_relu: icls.blocks - ConvBnReLU
                - maxpool: torch.nn - MaxPool2d
        """

        super(FirstConv, self).__init__()
        self.build_blocks_from_cfg(cfg)

    def forward(self, x: Tensor) -> Tensor:

        x = self.conv_bn_relu(x)
        x = self.maxpool(x)
        return x
