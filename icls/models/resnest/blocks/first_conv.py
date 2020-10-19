from torch.nn import Module
from icls.models import Builder
from omegaconf import ListConfig
from torch import Tensor


class FirstConv(Module, Builder):

    conv33_bn_relu_0: Module
    conv33_bn_relu_1: Module
    conv33_bn_relu_2: Module
    maxpool: Module

    def __init__(self, cfg: ListConfig) -> None:

        """
        Args:
            cfg (ListConfig):
                - conv33_bn_relu_0: icls.blocks - Conv33BnReLU
                - conv33_bn_relu_1: icls.blocks - Conv33BnReLU
                - conv33_bn_relu_2: icls.blocks - Conv33BnReLU
                - maxpool: torch.nn - MaxPool2d
        """

        super(FirstConv, self).__init__()
        self.build_blocks_from_cfg(cfg)

    def forward(self, x: Tensor) -> Tensor:

        x = self.conv33_bn_relu_0(x)
        x = self.conv33_bn_relu_1(x)
        x = self.conv33_bn_relu_2(x)
        x = self.maxpool(x)
        return x
