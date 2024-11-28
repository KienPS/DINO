from ultralytics import YOLO
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models._utils import IntermediateLayerGetter
from util.misc import NestedTensor


class YOLOBackbone(nn.Module):
    def __init__(self, backbone: nn.Module, num_channels: list[int]):
        super().__init__()
        self.num_channels = num_channels
        self.body = IntermediateLayerGetter(backbone, return_layers={'4': '0', '6': '1', '10': '2'})

    def forward(self, tensor_list: NestedTensor):
        xs = self.body(tensor_list.tensors)
        out: dict[str, NestedTensor] = {}
        for name, x in xs.items():
            m = tensor_list.mask
            assert m is not None
            mask = F.interpolate(m[None].float(), size=x.shape[-2:]).to(torch.bool)[0]
            out[name] = NestedTensor(x, mask)

        return out


def build_yolo(model_name: str):
    yolo_num_channels = {
        'yolo11n.pt': [128, 128, 256],
        'yolo11s.pt': [256, 256, 512],
        'yolo11m.pt': [512, 512, 512],
        'yolo11l.pt': [512, 512, 512],
        'yolo11x.pt': [768, 768, 768]
    }

    yolo_model = YOLO(model_name)
    backbone = YOLOBackbone(backbone=yolo_model.model.model[:11], num_channels=yolo_num_channels[model_name])
    return backbone
    