import argparse
import sys
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import mindspore as ms

from src.config import apply_overrides, load_config
from src.datasets import ADE20KSegDataset, PetSegDataset
from src.models import build_model
from src.runtime import configure_runtime


IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)[:, None, None]
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)[:, None, None]


def denormalize(image):
    image = image * IMAGENET_STD + IMAGENET_MEAN
    image = np.clip(image, 0, 1)
    return (image.transpose(1, 2, 0) * 255).astype(np.uint8)


def colorize_mask(mask):
    mask = mask.astype(np.int32)
    valid = mask != 255
    safe_mask = np.where(valid, mask, 0)
    vis = np.zeros((*mask.shape, 3), dtype=np.uint8)
    vis[..., 0] = ((safe_mask * 37 + 17) % 255).astype(np.uint8)
    vis[..., 1] = ((safe_mask * 67 + 29) % 255).astype(np.uint8)
    vis[..., 2] = ((safe_mask * 97 + 43) % 255).astype(np.uint8)
    vis[~valid] = [0, 0, 0]
    return vis


def parse_args():
    parser = argparse.ArgumentParser(description="Save segmentation prediction visualizations.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--out", default="outputs/visualizations")
    parser.add_argument("--count", type=int, default=8)
    parser.add_argument("--device-target", default=None, choices=["Ascend", "GPU", "CPU"])
    parser.add_argument("--data-root", default=None)
    parser.add_argument("--output-dir", default=None)
    return parser.parse_args()


def main():
    args = parse_args()
    cfg = apply_overrides(load_config(args.config), args)
    configure_runtime(cfg)

    network = build_model(cfg)
    params = ms.load_checkpoint(args.checkpoint)
    ms.load_param_into_net(network, params)
    network.set_train(False)

    dataset_name = cfg["dataset"]["name"]
    if dataset_name == "pet":
        dataset = PetSegDataset(
            cfg["dataset"]["root"],
            split="val",
            image_size=cfg["dataset"]["image_size"],
            max_samples=args.count,
            augment=False,
            seed=cfg.get("seed", 42),
            class_mode=cfg["dataset"].get("class_mode", "trimap"),
        )
    elif dataset_name == "ade20k":
        dataset = ADE20KSegDataset(
            cfg["dataset"]["root"],
            split="val",
            image_size=cfg["dataset"]["image_size"],
            max_samples=args.count,
            augment=False,
            seed=cfg.get("seed", 42),
        )
    else:
        raise ValueError(f"Unsupported visualization dataset: {dataset_name}")

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    for i in range(min(args.count, len(dataset))):
        image, mask, _ = dataset[i]
        logits = network(ms.Tensor(image[None, ...], ms.float32))
        if isinstance(logits, tuple):
            logits = logits[0]
        pred = logits.asnumpy().argmax(axis=1)[0].astype(np.int32)
        panel = np.concatenate(
            [denormalize(image), colorize_mask(mask), colorize_mask(pred)],
            axis=1,
        )
        Image.fromarray(panel).save(out_dir / f"{dataset_name}_pred_{i:02d}.png")

    print(f"Saved visualizations to {out_dir}")


if __name__ == "__main__":
    main()
