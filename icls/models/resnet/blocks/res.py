from torch.nn import Module
from icls.models import Builder
from omegaconf import ListConfig
from torch import Tensor


class Res(Module, Builder):
    def __init__(self, cfg: ListConfig) -> None:

        """
        Args:
            cfg (ListConfig):
                - bottleneck_*: icls.models.resnet.blocks - Bottleneck
        """

        super(Res, self).__init__()
        self.build_blocks_from_cfg(cfg)

    def forward(self, x: Tensor) -> Tensor:

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
