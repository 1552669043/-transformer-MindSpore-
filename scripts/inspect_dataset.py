import argparse
import sys
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.datasets import ADE20KSegDataset, PetSegDataset


def colorize_mask(mask):
    mask = mask.astype(np.int32)
    valid = mask != 255
    vis = np.zeros((*mask.shape, 3), dtype=np.uint8)
    vis[..., 0] = ((mask * 37 + 17) % 255).astype(np.uint8)
    vis[..., 1] = ((mask * 67 + 29) % 255).astype(np.uint8)
    vis[..., 2] = ((mask * 97 + 43) % 255).astype(np.uint8)
    vis[~valid] = [0, 0, 0]
    return vis


def denormalize(image):
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)[:, None, None]
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)[:, None, None]
    image = image * std + mean
    image = np.clip(image, 0, 1)
    return (image.transpose(1, 2, 0) * 255).astype(np.uint8)


def main():
    parser = argparse.ArgumentParser(description="Create segmentation dataset preview images.")
    parser.add_argument("--dataset", default="ade20k", choices=["ade20k", "pet"])
    parser.add_argument("--root", default="data/ade20k")
    parser.add_argument("--out", default="outputs/dataset_preview")
    parser.add_argument("--count", type=int, default=8)
    args = parser.parse_args()

    if args.dataset == "pet":
        dataset = PetSegDataset(args.root, "train", image_size=256, max_samples=args.count)
    else:
        dataset = ADE20KSegDataset(args.root, "train", image_size=512, max_samples=args.count)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    for i in range(min(args.count, len(dataset))):
        image, mask, edge = dataset[i]
        rgb = denormalize(image)
        mask_vis = colorize_mask(mask)
        edge_vis = np.zeros_like(rgb)
        edge_vis[edge == 1] = [255, 255, 0]
        canvas = np.concatenate([rgb, mask_vis, edge_vis], axis=1)
        Image.fromarray(canvas).save(out / f"sample_{i:02d}.png")
    print(f"Saved previews to {out}")


if __name__ == "__main__":
    main()
