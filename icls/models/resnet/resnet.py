import os
from typing import Dict, Tuple

import torch
from torch.nn import Module
from icls.models import Builder
from omegaconf import OmegaConf
from torch import Tensor


class ResNet(Module, Builder):

    first_conv: Module
    res_0: Module
    res_1: Module
    res_2: Module
    res_3: Module
    avgpool: Module
    fc: Module

    def __init__(self, cfg_path: str) -> None:

        """
        Args:
            cfg_path: Path to config file

        cfg:
            - first_conv: icls.backbone.resnet.blocks - FirstConv
            - res_0: icls.backbone.resnet.blocks - Res
            - res_1: icls.backbone.resnet.blocks - Res
            - res_2: icls.backbone.resnet.blocks - Res
            - res_3: icls.backbone.resnet.blocks - Res
            - avgpool: torch.nn - AdaptiveAvgPool2d
            - fc: torch.nn - Linear
        """

        super(ResNet, self).__init__()
        cfg = OmegaConf.load(cfg_path)
        self.build_blocks_from_cfg(cfg)

    def forward(self, x: Tensor) -> Tuple[Dict[str, Tensor], Tensor]:

        x = self.first_conv(x)
        x_res_0 = self.res_0(x)
        x_res_1 = self.res_1(x_res_0)
        x_res_2 = self.res_2(x_res_1)
        x_res_3 = self.res_3(x_res_2)
        y = self.avgpool(x_res_3)
        y = torch.flatten(y, start_dim=1)
        y = self.fc(y)

        feature_dict = {"res_0": x_res_0, "res_1": x_res_1, "res_2": x_res_2, "res_3": x_res_3}
        return (feature_dict, y)


class ResNet50(ResNet):

    """
    ResNet50 model from
    Deep Residual Learning for Image Recognition (https://arxiv.org/pdf/1512.03385.pdf)

    References:
        - https://github.com/pytorch/vision/blob/a9c78f1b017d912fa9583ae6d93bff0a3807c29b/torchvision/models/resnet.py#L256
        - https://download.pytorch.org/models/resnet50-19c8e357.pth
    """

    def __init__(self, pretrained: bool = False) -> None:

        dirname = os.path.dirname(__file__)
        super().__init__(cfg_path=f"{dirname}/yamls/resnet50.yaml")

        if pretrained:
            state_dict = torch.load(f"{dirname}/checkpoints/resnet50.pth")
            self.load_state_dict(state_dict, strict=False)


class ResNet101(ResNet):

    """
    ResNet101 model from
    Deep Residual Learning for Image Recognition (https://arxiv.org/pdf/1512.03385.pdf)

    References:
        - https://github.com/pytorch/vision/blob/a9c78f1b017d912fa9583ae6d93bff0a3807c29b/torchvision/models/resnet.py#L268
        - https://download.pytorch.org/models/resnet101-5d3b4d8f.pth
    """

    def __init__(self, pretrained: bool = False) -> None:

        dirname = os.path.dirname(__file__)
        super().__init__(cfg_path=f"{dirname}/yamls/resnet101.yaml")

        if pretrained:
            state_dict = torch.load(f"{dirname}/checkpoints/resnet101.pth")
            self.load_state_dict(state_dict, strict=False)


class ResNet152(ResNet):

    """
    ResNet152 model from
    Deep Residual Learning for Image Recognition (https://arxiv.org/pdf/1512.03385.pdf)

    References:
        - https://github.com/pytorch/vision/blob/a9c78f1b017d912fa9583ae6d93bff0a3807c29b/torchvision/models/resnet.py#L280
        - https://download.pytorch.org/models/resnet152-b121ed2d.pth
    """

    def __init__(self, pretrained: bool = False) -> None:

        dirname = os.path.dirname(__file__)
        super().__init__(cfg_path=f"{dirname}/yamls/resnet152.yaml")

        if pretrained:
            state_dict = torch.load(f"{dirname}/checkpoints/resnet152.pth")
            self.load_state_dict(state_dict, strict=False)
