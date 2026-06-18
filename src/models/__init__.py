from .mask_transformer import Mask2FormerTiny, OneFormerTiny
from .segformer import SegFormerEdge, SegFormerTiny
from .setr import SETRTiny


def build_model(cfg):
    model_cfg = cfg["model"]
    dataset_cfg = cfg["dataset"]
    name = model_cfg["name"]
    num_classes = int(dataset_cfg["num_classes"])

    if name == "setr_tiny":
        return SETRTiny(num_classes=num_classes, **_without_name(model_cfg))
    if name == "segformer_tiny":
        return SegFormerTiny(num_classes=num_classes, **_without_name(model_cfg))
    if name == "segformer_edge":
        kwargs = _without_name(model_cfg)
        kwargs.pop("edge_loss_weight", None)
        return SegFormerEdge(num_classes=num_classes, **kwargs)
    if name == "mask2former_tiny":
        return Mask2FormerTiny(num_classes=num_classes, **_without_name(model_cfg))
    if name == "oneformer_tiny":
        return OneFormerTiny(num_classes=num_classes, **_without_name(model_cfg))
    raise ValueError(f"Unsupported model: {name}")


def _without_name(model_cfg):
    return {k: v for k, v in model_cfg.items() if k != "name"}
