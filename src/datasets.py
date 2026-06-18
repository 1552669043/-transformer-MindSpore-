from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw


IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def _resampling(name):
    if hasattr(Image, "Resampling"):
        return getattr(Image.Resampling, name)
    return getattr(Image, name)


def _normalize_image(image):
    arr = np.asarray(image, dtype=np.float32) / 255.0
    arr = (arr - IMAGENET_MEAN) / IMAGENET_STD
    return arr.transpose(2, 0, 1).astype(np.float32)


def _edge_from_mask(mask):
    edge = np.zeros(mask.shape, dtype=np.int32)
    edge[:, 1:] |= mask[:, 1:] != mask[:, :-1]
    edge[:, :-1] |= mask[:, 1:] != mask[:, :-1]
    edge[1:, :] |= mask[1:, :] != mask[:-1, :]
    edge[:-1, :] |= mask[1:, :] != mask[:-1, :]
    edge[mask == 255] = 255
    return edge


class ADE20KSegDataset:
    """ADE20K / ADEChallengeData2016 semantic segmentation dataset."""

    def __init__(
        self,
        root,
        split,
        image_size=512,
        max_samples=None,
        augment=False,
        seed=42,
        random_subset=False,
    ):
        self.root = Path(root)
        self.split = split
        self.image_size = int(image_size)
        self.augment = augment
        self.rng = np.random.default_rng(seed)

        base = self.root / "ADEChallengeData2016"
        if not base.exists():
            base = self.root

        if split == "train":
            ade_split = "training"
        elif split in {"val", "test"}:
            ade_split = "validation"
        else:
            raise ValueError(f"Unsupported split: {split}")

        image_dir = base / "images" / ade_split
        mask_dir = base / "annotations" / ade_split
        if not image_dir.exists() or not mask_dir.exists():
            raise FileNotFoundError(
                f"Can not find ADE20K split under {base}. "
                f"Run: python scripts/download_ade20k.py --root {self.root}"
            )

        images = sorted(image_dir.glob("*.jpg"))
        samples = [(p, mask_dir / f"{p.stem}.png") for p in images]
        if max_samples is not None:
            max_samples = min(int(max_samples), len(samples))
            if random_subset:
                indices = self.rng.choice(len(samples), size=max_samples, replace=False)
                samples = [samples[int(i)] for i in sorted(indices)]
            else:
                samples = samples[:max_samples]
        self.samples = samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        image_path, mask_path = self.samples[index]
        image = Image.open(image_path).convert("RGB")
        mask_img = Image.open(mask_path)

        if self.augment and self.rng.random() < 0.5:
            image = image.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
            mask_img = mask_img.transpose(Image.Transpose.FLIP_LEFT_RIGHT)

        size = (self.image_size, self.image_size)
        image = image.resize(size, _resampling("BILINEAR"))
        mask_img = mask_img.resize(size, _resampling("NEAREST"))

        image = _normalize_image(image)
        raw = np.asarray(mask_img, dtype=np.int32)
        mask = raw - 1
        mask[raw == 0] = 255
        edge = _edge_from_mask(mask)
        return image, mask.astype(np.int32), edge.astype(np.int32)


class PetSegDataset:
    """Pet foreground/background/boundary segmentation dataset.

    Supported layouts:
    1. root/images, root/masks, root/splits/train.txt, root/splits/val.txt
    2. root/images, root/annotations/trimaps, root/annotations/trainval.txt, root/annotations/test.txt
    """

    def __init__(
        self,
        root,
        split,
        image_size=256,
        max_samples=None,
        augment=False,
        seed=42,
        random_subset=False,
        class_mode="trimap",
    ):
        self.root = Path(root)
        self.split = split
        self.image_size = int(image_size)
        self.augment = augment
        self.rng = np.random.default_rng(seed)
        self.class_mode = class_mode

        samples = self._load_samples(split)
        if max_samples is not None:
            max_samples = min(int(max_samples), len(samples))
            if random_subset:
                indices = self.rng.choice(len(samples), size=max_samples, replace=False)
                samples = [samples[int(i)] for i in sorted(indices)]
            else:
                samples = samples[:max_samples]
        self.samples = samples

    def _load_samples(self, split):
        custom_split = self.root / "splits" / ("train.txt" if split == "train" else "val.txt")
        if custom_split.exists():
            samples = []
            with open(custom_split, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    image_rel, mask_rel = line.split()[:2]
                    samples.append((self.root / image_rel, self.root / mask_rel))
            return samples

        ann_dir = self.root / "annotations"
        split_file = ann_dir / ("trainval.txt" if split == "train" else "test.txt")
        mask_dir = ann_dir / "trimaps"
        image_dir = self.root / "images"
        if split_file.exists() and mask_dir.exists() and image_dir.exists():
            samples = []
            with open(split_file, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line or line.startswith("#"):
                        continue
                    name = line.split()[0]
                    samples.append((image_dir / f"{name}.jpg", mask_dir / f"{name}.png"))
            return samples

        raise FileNotFoundError(
            f"Can not find pet dataset splits under {self.root}. "
            "Expected root/images, root/masks, root/splits or Oxford annotations/trimaps layout."
        )

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        image_path, mask_path = self.samples[index]
        image = Image.open(image_path).convert("RGB")
        mask_img = Image.open(mask_path)

        if self.augment and self.rng.random() < 0.5:
            image = image.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
            mask_img = mask_img.transpose(Image.Transpose.FLIP_LEFT_RIGHT)

        size = (self.image_size, self.image_size)
        image = image.resize(size, _resampling("BILINEAR"))
        mask_img = mask_img.resize(size, _resampling("NEAREST"))

        image = _normalize_image(image)
        raw = np.asarray(mask_img, dtype=np.int32)
        mask = self._convert_mask(raw)
        edge = _edge_from_mask(mask)
        return image, mask.astype(np.int32), edge.astype(np.int32)

    def _convert_mask(self, raw):
        values = set(np.unique(raw).tolist())
        if values.issubset({0, 1, 2}):
            mask = raw.copy()
        else:
            mask = np.full(raw.shape, 255, dtype=np.int32)
            mask[raw == 2] = 0
            mask[raw == 1] = 1
            mask[raw == 3] = 2

        if self.class_mode == "binary":
            binary = np.full(mask.shape, 255, dtype=np.int32)
            binary[mask == 0] = 0
            binary[(mask == 1) | (mask == 2)] = 1
            return binary
        return mask


class SyntheticSegDataset:
    """Small generated dataset for pipeline smoke tests."""

    def __init__(self, length=32, image_size=128, seed=0):
        self.length = int(length)
        self.image_size = int(image_size)
        self.seed = int(seed)

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        rng = np.random.default_rng(self.seed + index)
        size = self.image_size
        image = Image.new("RGB", (size, size), (30, 40, 55))
        mask_img = Image.new("L", (size, size), 0)
        draw = ImageDraw.Draw(image)
        mask_draw = ImageDraw.Draw(mask_img)

        x0 = int(rng.integers(size // 8, size // 2))
        y0 = int(rng.integers(size // 8, size // 2))
        x1 = int(rng.integers(size // 2, size - size // 8))
        y1 = int(rng.integers(size // 2, size - size // 8))
        min_span = max(12, size // 8)
        x1 = min(size - 1, max(x1, x0 + min_span))
        y1 = min(size - 1, max(y1, y0 + min_span))
        color = tuple(int(v) for v in rng.integers(80, 230, size=3))
        draw.ellipse((x0, y0, x1, y1), fill=color)
        mask_draw.ellipse((x0, y0, x1, y1), fill=1)
        inner_margin = min(5, max(1, (x1 - x0) // 3), max(1, (y1 - y0) // 3))
        mask_draw.ellipse(
            (x0 + inner_margin, y0 + inner_margin, x1 - inner_margin, y1 - inner_margin),
            fill=2,
        )

        image = _normalize_image(image)
        mask = np.asarray(mask_img, dtype=np.int32)
        edge = _edge_from_mask(mask)
        return image, mask, edge.astype(np.int32)


def create_mindspore_dataset(cfg, split):
    import mindspore.dataset as ds

    data_cfg = cfg["dataset"]
    seed = cfg.get("seed", 42)
    name = data_cfg["name"]

    if name == "ade20k":
        max_key = "max_train_samples" if split == "train" else "max_val_samples"
        source = ADE20KSegDataset(
            root=data_cfg["root"],
            split=split,
            image_size=data_cfg["image_size"],
            max_samples=data_cfg.get(max_key),
            augment=(split == "train"),
            seed=seed + (0 if split == "train" else 10000),
            random_subset=bool(data_cfg.get("random_subset", False)),
        )
    elif name == "pet":
        max_key = "max_train_samples" if split == "train" else "max_val_samples"
        source = PetSegDataset(
            root=data_cfg["root"],
            split=split,
            image_size=data_cfg["image_size"],
            max_samples=data_cfg.get(max_key),
            augment=(split == "train"),
            seed=seed + (0 if split == "train" else 10000),
            random_subset=bool(data_cfg.get("random_subset", False)),
            class_mode=data_cfg.get("class_mode", "trimap"),
        )
    elif name == "synthetic":
        length_key = "train_length" if split == "train" else "val_length"
        source = SyntheticSegDataset(
            length=data_cfg.get(length_key, 32),
            image_size=data_cfg["image_size"],
            seed=seed + (0 if split == "train" else 10000),
        )
    else:
        raise ValueError(f"Unsupported dataset: {name}")

    dataset = ds.GeneratorDataset(
        source,
        column_names=["image", "mask", "edge"],
        shuffle=(split == "train"),
        num_parallel_workers=int(data_cfg.get("num_workers", 1)),
        python_multiprocessing=False,
    )
    dataset = dataset.batch(
        int(data_cfg["batch_size"]),
        drop_remainder=(split == "train"),
    )
    return dataset
