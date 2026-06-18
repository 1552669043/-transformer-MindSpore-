import argparse
import math
import sys
import time
from pathlib import Path

import mindspore as ms
from mindspore import nn

from src.config import apply_overrides, load_config, save_config
from src.datasets import create_mindspore_dataset
from src.losses import SegmentationLossCell
from src.metrics import evaluate_model
from src.models import build_model
from src.runtime import configure_runtime


def print_progress(prefix, current, total, extra="", width=30):
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


def build_learning_rate(cfg, steps_per_epoch):
    train_cfg = cfg["train"]
    base_lr = float(train_cfg["learning_rate"])
    schedule = train_cfg.get("lr_schedule", "constant").lower()
    if schedule == "constant":
        return base_lr

    total_epochs = int(train_cfg["epochs"])
    total_steps = max(int(steps_per_epoch) * total_epochs, 1)
    min_lr = float(train_cfg.get("min_learning_rate", base_lr * 0.1))
    warmup_epochs = float(train_cfg.get("warmup_epochs", 0))
    warmup_steps = int(max(warmup_epochs, 0.0) * int(steps_per_epoch))
    start_lr = float(train_cfg.get("warmup_start_lr", base_lr * 0.2))

    if schedule != "cosine":
        raise ValueError(f"Unsupported lr_schedule: {schedule}")

    values = []
    decay_steps = max(total_steps - warmup_steps, 1)
    for step in range(total_steps):
        if warmup_steps > 0 and step < warmup_steps:
            ratio = step / max(warmup_steps, 1)
            lr = start_lr + (base_lr - start_lr) * ratio
        else:
            decay_step = step - warmup_steps
            cosine = 0.5 * (1.0 + math.cos(math.pi * decay_step / decay_steps))
            lr = min_lr + (base_lr - min_lr) * cosine
        values.append(float(lr))
    return values


def parse_args():
    parser = argparse.ArgumentParser(description="Train transformer semantic segmentation models with MindSpore.")
    parser.add_argument("--config", required=True, help="Path to a JSON config.")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--learning-rate", type=float, default=None)
    parser.add_argument("--device-target", default=None, choices=["Ascend", "GPU", "CPU"])
    parser.add_argument("--data-root", default=None)
    parser.add_argument("--output-dir", default=None)
    return parser.parse_args()


def main():
    args = parse_args()
    cfg = apply_overrides(load_config(args.config), args)
    configure_runtime(cfg)

    output_dir = Path(cfg["train"]["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    save_config(cfg, output_dir / "resolved_config.json")

    train_ds = create_mindspore_dataset(cfg, "train")
    val_ds = create_mindspore_dataset(cfg, "val")
    steps_per_epoch = train_ds.get_dataset_size()
    print(f"Experiment: {cfg['experiment_name']}")
    print(f"Steps per epoch: {steps_per_epoch}")

    network = build_model(cfg)
    use_edge = cfg["model"]["name"] == "segformer_edge"
    loss_cell = SegmentationLossCell(
        network,
        use_edge=use_edge,
        edge_loss_weight=cfg["model"].get("edge_loss_weight", 0.3),
    )
    learning_rate = build_learning_rate(cfg, steps_per_epoch)
    optimizer = nn.AdamWeightDecay(
        network.trainable_params(),
        learning_rate=learning_rate,
        weight_decay=float(cfg["train"]["weight_decay"]),
    )
    train_cell = nn.TrainOneStepCell(loss_cell, optimizer)
    train_cell.set_train()

    best_miou = -1.0
    metrics_log = output_dir / "metrics.csv"
    with open(metrics_log, "w", encoding="utf-8") as f:
        f.write("epoch,loss,pixel_acc,mean_iou,class_iou,elapsed_sec\n")

    total_epochs = int(cfg["train"]["epochs"])
    eval_every = int(cfg["train"].get("eval_every", 5))

    for epoch in range(1, total_epochs + 1):
        start = time.time()
        loss_sum = 0.0
        step_count = 0
        for step, batch in enumerate(train_ds.create_dict_iterator(num_epochs=1, output_numpy=False), start=1):
            loss = train_cell(batch["image"], batch["mask"], batch["edge"])
            loss_value = float(loss.asnumpy())
            loss_sum += loss_value
            step_count += 1
            print_progress(
                f"epoch {epoch}/{total_epochs} train",
                step,
                steps_per_epoch,
                extra=f"loss={loss_value:.4f}",
            )

        avg_loss = loss_sum / max(step_count, 1)
        metrics = {"pixel_acc": 0.0, "mean_iou": 0.0, "class_iou": []}
        should_eval = epoch % eval_every == 0 or epoch == total_epochs
        if should_eval:
            metrics = evaluate_model(
                network,
                val_ds,
                num_classes=int(cfg["dataset"]["num_classes"]),
                show_progress=True,
                progress_prefix=f"epoch {epoch}/{total_epochs} val",
            )
            print(
                f"epoch={epoch} avg_loss={avg_loss:.4f} "
                f"pixel_acc={metrics['pixel_acc']:.4f} mean_iou={metrics['mean_iou']:.4f}"
            )
            if cfg["train"].get("save_best", True) and metrics["mean_iou"] > best_miou:
                best_miou = metrics["mean_iou"]
                ms.save_checkpoint(network, str(output_dir / "best.ckpt"))
                print(f"saved best checkpoint: {output_dir / 'best.ckpt'}")
        else:
            print(
                f"epoch={epoch} avg_loss={avg_loss:.4f} "
                f"eval=skipped next_eval_epoch={min(((epoch // eval_every) + 1) * eval_every, total_epochs)}"
            )

        elapsed = time.time() - start
        with open(metrics_log, "a", encoding="utf-8") as f:
            f.write(
                f"{epoch},{avg_loss:.6f},{metrics['pixel_acc']:.6f},"
                f"{metrics['mean_iou']:.6f},\"{metrics['class_iou']}\",{elapsed:.2f}\n"
            )

    ms.save_checkpoint(network, str(output_dir / "last.ckpt"))
    print(f"Training finished. Last checkpoint: {output_dir / 'last.ckpt'}")


if __name__ == "__main__":
    main()
