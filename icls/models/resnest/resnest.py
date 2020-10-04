import icls.types as T
import onnx
import torch
import torch.nn as nn
import torchvision
from icls.models import Builder
from omegaconf import OmegaConf


class ResNeSt50(nn.Module, Builder):

    first_conv: T.Module
    residual_block_0: T.Module
    residual_block_1: T.Module
    residual_block_2: T.Module
    residual_block_3: T.Module
    avgpool: T.Module
    fc: T.Module

    def __init__(self, object_cfg: T.ListConfig) -> None:

        """
        Args:
            object_cfg (T.ListConfig):
                - first_conv: icls.backbone.resnet.blocks - FirstConv
                - residual_block_0: icls.backbone.resnet.blocks - ResidualBlock
                - residual_block_1: icls.backbone.resnet.blocks - ResidualBlock
                - residual_block_2: icls.backbone.resnet.blocks - ResidualBlock
                - residual_block_3: icls.backbone.resnet.blocks - ResidualBlock
                - avgpool: torch.nn - AdaptiveAvgPool2d
                - fc: torch.nn - Linear
        """

        super(ResNeSt50, self).__init__()
        self.build_blocks(object_cfg)

    def forward(self, x: T.Tensor) -> T.Tuple[T.Dict[str, T.Tensor], T.Tensor]:

        x = self.first_conv(x)
        x_0 = self.residual_block_0(x)
        x_1 = self.residual_block_1(x_0)
        x_2 = self.residual_block_2(x_1)
        x_3 = self.residual_block_3(x_2)
        y = self.avgpool(x_3)
        y = torch.flatten(y, start_dim=1)
        y = self.fc(y)

        feature_dict = {"res_0": x_0, "res_1": x_1, "res_2": x_2, "res_3": x_res_3}
        return (feature_dict, y)


if __name__ == "__main__":

    x = torch.randn(8, 3, 128, 128)

    cfg = OmegaConf.load("/dgx/github/iClassification/icls/models/resnet/resnet50.yaml")
    model = ResNet50(cfg)
    filename = "resnet50.onnx"
    torch.onnx.export(model, x, filename, export_params=True, opset_version=8)
    onnx.save(onnx.shape_inference.infer_shapes(onnx.load(filename)), filename)
