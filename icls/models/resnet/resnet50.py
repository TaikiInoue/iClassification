import icls.types as T
import onnx
import torch
import torch.nn as nn
import torchvision
from icls.models import Builder
from omegaconf import OmegaConf


class ResNet50(nn.Module, Builder):

    first_conv: T.Module
    res_0: T.Module
    res_1: T.Module
    res_2: T.Module
    res_3: T.Module
    avgpool: T.Module
    fc: T.Module

    def __init__(self, object_cfg: T.ListConfig) -> None:

        """
        Args:
            object_cfg (T.ListConfig):
                - first_conv: icls.backbone.resnet.blocks - FirstConv
                - res_0: icls.backbone.resnet.blocks - Res
                - res_1: icls.backbone.resnet.blocks - Res
                - res_2: icls.backbone.resnet.blocks - Res
                - res_3: icls.backbone.resnet.blocks - Res
                - avgpool: torch.nn - AdaptiveAvgPool2d
                - fc: torch.nn - Linear
        """

        super(ResNet50, self).__init__()
        self.build_blocks(object_cfg)

    def forward(self, x: T.Tensor) -> T.Tuple[T.Dict[str, T.Tensor], T.Tensor]:

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


if __name__ == "__main__":

    x = torch.randn(8, 3, 128, 128)

    model = torchvision.models.resnet50(pretrained=False)
    filename = "resnet50_torchvision.onnx"
    torch.onnx.export(model, x, filename, export_params=True, opset_version=8)
    onnx.save(onnx.shape_inference.infer_shapes(onnx.load(filename)), filename)

    cfg = OmegaConf.load("/dgx/github/iClassification/icls/models/resnet/resnet50.yaml")
    model = ResNet50(cfg)
    filename = "resnet50.onnx"
    torch.onnx.export(model, x, filename, export_params=True, opset_version=8)
    onnx.save(onnx.shape_inference.infer_shapes(onnx.load(filename)), filename)
