from torch.nn import Module
from icls.models import Builder
from omegaconf import ListConfig
from torch import Tensor


class Bottleneck(Module, Builder):

    conv11_bn_relu: Module
    conv33_bn_relu: Module
    split_attention: Module
    conv11_bn: Module
    downsample: Module
    relu: Module

    def __init__(self, cfg: ListConfig) -> None:

        """
        Args:
            cfg (ListConfig):
                - conv11_bn_relu: icls.blocks - Conv11BnReLU
                - conv33_bn_relu: icls.blocks - Conv33BnReLU
                - split_attention: icls.models.resnest - SplitAttention
                - conv11_bn: icls.blocks - Conv11Bn
                - downsample: icls.blocks - Conv11Bn
                - relu: torch.nn - ReLU
        """

        super(Bottleneck, self).__init__()
        self.build_blocks_from_cfg(cfg)

    def forward(self, x: Tensor) -> Tensor:

        residual = x
        out = self.conv11_bn_relu(x)
        out = self.conv33_bn_relu(out)
        out = self.split_attention(out)
        out = self.conv11_bn(out)

        if hasattr(self, "downsample"):
            residual = self.downsample(residual)

        out += residual
        out = self.relu(out)
        return out
