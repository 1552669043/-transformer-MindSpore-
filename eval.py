import argparse

import mindspore as ms

from src.config import apply_overrides, load_config
from src.datasets import create_mindspore_dataset
from src.metrics import evaluate_model
from src.models import build_model
from src.runtime import configure_runtime


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate transformer semantic segmentation models.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
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
    val_ds = create_mindspore_dataset(cfg, "val")
    metrics = evaluate_model(
        network,
        val_ds,
        int(cfg["dataset"]["num_classes"]),
        show_progress=True,
        progress_prefix="eval",
    )
    print("Evaluation result:")
    print(metrics)


if __name__ == "__main__":
    main()
