from torch.nn import Module
from icls.models import Builder
from omegaconf import ListConfig
from torch import Tensor


class Downsample(Module, Builder):

    avgpool: Module
    conv11_bn_relu: Module

    def __init__(self, cfg: ListConfig) -> None:

        """
        Args:
            cfg (ListConfig):
                - avgpool: torch.nn - AvgPool2d
                - conv11_bn_relu: icls.blocks - Conv11BnReLU
        """

        super().__init__()
        self.build_blocks_from_cfg(cfg)

    def forward(self, x: Tensor) -> Tensor:

        x = self.avgpool(x)
        x = self.conv11_bn_relu(x)
        return x
