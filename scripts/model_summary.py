import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.config import load_config
from src.models import build_model


def count_params(network):
    total = 0
    trainable = 0
    for param in network.get_parameters():
        size = 1
        for dim in param.shape:
            size *= int(dim)
        total += size
        if param.requires_grad:
            trainable += size
    return total, trainable


def main():
    parser = argparse.ArgumentParser(description="Print model parameter counts.")
    parser.add_argument("configs", nargs="+")
    args = parser.parse_args()

    for config_path in args.configs:
        cfg = load_config(config_path)
        network = build_model(cfg)
        total, trainable = count_params(network)
        print(
            f"{cfg['experiment_name']}: "
            f"total={total / 1e6:.2f}M, trainable={trainable / 1e6:.2f}M"
        )


if __name__ == "__main__":
    main()
