import icls.types as T
import torch.nn as nn
from icls.blocks import Conv11Bn, Conv11BnReLU, Conv33BnReLU
from icls.models import Builder


class ResLayers(nn.Module, Builder):
    def __init__(
        self,
        in_channels: int = None,
        hidden_channels: int = None,
        out_channels: int = None,
        num_bottleneck: int = None,
    ) -> None:

        super(ResLayers, self).__init__()

        self.first_bottleneck = nn.Sequential(
            [
                Conv11BnReLU(in_channels, hidden_channels),
                Conv33BnReLU(hidden_channels, hidden_channels),
                Conv11Bn(hidden_channels, out_channels),
            ]
        )

        self.downsample = Conv11Bn(in_channels, out_channels)

        self.repeated_bottleneck_list = []
        for i in range(num_bottleneck - 1):
            self.repeated_bottleneck_list.append(
                nn.Sequential(
                    Conv11BnReLU(out_channels, hidden_channels),
                    Conv33BnReLU(hidden_channels, hidden_channels),
                    Conv11Bn(hidden_channels, out_channels),
                )
            )

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
