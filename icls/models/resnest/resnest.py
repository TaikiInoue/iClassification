from typing import Dict, Tuple

import torch
from torch.nn import Module
from icls.models import Builder
from omegaconf import ListConfig
from torch import Tensor


class ResNeSt50(Module, Builder):

    first_conv: Module
    residual_block_0: Module
    residual_block_1: Module
    residual_block_2: Module
    residual_block_3: Module
    avgpool: Module
    fc: Module

    def __init__(self, cfg: ListConfig) -> None:

        """
        Args:
            cfg (ListConfig):
                - first_conv: icls.backbone.resnet.blocks - FirstConv
                - residual_block_0: icls.backbone.resnet.blocks - ResidualBlock
                - residual_block_1: icls.backbone.resnet.blocks - ResidualBlock
                - residual_block_2: icls.backbone.resnet.blocks - ResidualBlock
                - residual_block_3: icls.backbone.resnet.blocks - ResidualBlock
                - avgpool: torch.nn - AdaptiveAvgPool2d
                - fc: torch.nn - Linear
        """

        super(ResNeSt50, self).__init__()
        self.build_blocks_from_cfg(cfg)

    def forward(self, x: Tensor) -> Tuple[Dict[str, Tensor], Tensor]:

        x = self.first_conv(x)
        x_0 = self.residual_block_0(x)
        x_1 = self.residual_block_1(x_0)
        x_2 = self.residual_block_2(x_1)
        x_3 = self.residual_block_3(x_2)
        y = self.avgpool(x_3)
        y = torch.flatten(y, start_dim=1)
        y = self.fc(y)

        feature_dict = {"res_0": x_0, "res_1": x_1, "res_2": x_2, "res_3": x_3}
        return (feature_dict, y)
