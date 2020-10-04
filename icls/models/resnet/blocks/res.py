import torch.nn as nn

import icls.types as T
from icls.models import Builder


class Res(nn.Module, Builder):
    def __init__(self, object_cfg: T.ListConfig) -> None:

        """
        Args:
            object_cfg (T.ListConfig):
                - bottleneck_*: icls.models.resnet.blocks - Bottleneck
        """

        super(Res, self).__init__()
        self.build_blocks(object_cfg)

    def forward(self, x: T.Tensor) -> T.Tensor:

        i = 0
        while True:
            var_name = f"bottleneck_{i}"
            if hasattr(self, var_name):
                bottleneck = getattr(self, var_name)
                x = bottleneck(x)
                i += 1
            else:
                break

        return x
