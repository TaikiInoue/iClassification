import icls.types as T
import torch.nn as nn
from icls.models import Builder


class Bottleneck(nn.Module, Builder):

    conv11_bn_relu: T.Module
    conv33_bn_relu: T.Module
    split_attention: T.Module
    conv11_bn: T.Module
    downsample: T.Module
    relu: T.Module

    def __init__(self, object_cfg: T.ListConfig) -> None:

        """
        Args:
            object_cfg (T.ListConfig):
                - conv11_bn_relu: icls.blocks - Conv11BnReLU
                - conv33_bn_relu: icls.blocks - Conv33BnReLU
                - split_attention: icls.models.resnest - SplitAttention
                - conv11_bn: icls.blocks - Conv11Bn
                - downsample: icls.blocks - Conv11Bn
                - relu: torch.nn - ReLU
        """

        super(Bottleneck, self).__init__()
        self.build_blocks(object_cfg)

    def forward(self, x: T.Tensor) -> T.Tensor:

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
