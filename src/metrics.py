import numpy as np
import sys


def _print_progress(prefix, current, total, extra="", width=30):
    total = max(int(total), 1)
    current = min(int(current), total)
    percent = current * 100.0 / total
    filled = int(width * current / total)
    bar = "#" * filled + "-" * (width - filled)
    suffix = f" {extra}" if extra else ""
    sys.stdout.write(
        f"\r{prefix} [{bar}] {percent:6.2f}% ({current}/{total}){suffix}"
    )
    sys.stdout.flush()
    if current >= total:
        sys.stdout.write("\n")
        sys.stdout.flush()


class SegmentationMetric:
    def __init__(self, num_classes, ignore_index=255):
        self.num_classes = int(num_classes)
        self.ignore_index = int(ignore_index)
        self.confusion = np.zeros((self.num_classes, self.num_classes), dtype=np.int64)

    def update(self, pred, target):
        pred = pred.reshape(-1).astype(np.int64)
        target = target.reshape(-1).astype(np.int64)
        valid = target != self.ignore_index
        valid &= target >= 0
        valid &= target < self.num_classes
        pred = pred[valid]
        target = target[valid]
        encoded = self.num_classes * target + pred
        bins = np.bincount(encoded, minlength=self.num_classes ** 2)
        self.confusion += bins.reshape(self.num_classes, self.num_classes)

    def compute(self):
        hist = self.confusion.astype(np.float64)
        true_positive = np.diag(hist)
        denominator = hist.sum(axis=1) + hist.sum(axis=0) - true_positive
        valid_classes = denominator > 0
        iou = np.full(self.num_classes, np.nan, dtype=np.float64)
        iou[valid_classes] = true_positive[valid_classes] / denominator[valid_classes]
        acc = true_positive.sum() / np.maximum(hist.sum(), 1.0)
        mean_iou = float(np.nanmean(iou)) if np.any(valid_classes) else 0.0
        return {
            "pixel_acc": float(acc),
            "mean_iou": mean_iou,
            "valid_classes": int(valid_classes.sum()),
            "class_iou": [None if np.isnan(v) else float(v) for v in iou],
        }


def evaluate_model(
    network,
    dataset,
    num_classes,
    ignore_index=255,
    show_progress=False,
    progress_prefix="val",
):
    network.set_train(False)
    metric = SegmentationMetric(num_classes, ignore_index)
    total = dataset.get_dataset_size() if hasattr(dataset, "get_dataset_size") else 0

    for step, batch in enumerate(dataset.create_dict_iterator(num_epochs=1, output_numpy=False), start=1):
        output = network(batch["image"])
        if isinstance(output, tuple):
            output = output[0]
        pred = output.asnumpy().argmax(axis=1).astype(np.int32)
        target = batch["mask"].asnumpy().astype(np.int32)
        metric.update(pred, target)
        if show_progress:
            _print_progress(progress_prefix, step, total)

    return metric.compute()
