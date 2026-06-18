import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.config import load_config, pretty_print_config


def main():
    parser = argparse.ArgumentParser(description="Check config files without importing MindSpore.")
    parser.add_argument("configs", nargs="+")
    args = parser.parse_args()

    for config_path in args.configs:
        cfg = load_config(config_path)
        print(f"\n== {config_path} ==")
        print(f"experiment: {cfg['experiment_name']}")
        print(f"model: {cfg['model']['name']}")
        print(f"dataset: {cfg['dataset']['name']}")
        print(f"output: {cfg['train']['output_dir']}")
        if cfg["dataset"]["name"] == "ade20k":
            root = Path(cfg["dataset"]["root"])
            print(f"dataset root exists: {root.exists()}")
        pretty_print_config(cfg)


if __name__ == "__main__":
    main()
