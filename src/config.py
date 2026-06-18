import copy
import json
from pathlib import Path


def load_config(path):
    with open(path, "r", encoding="utf-8") as f:
        cfg = json.load(f)
    return cfg


def save_config(cfg, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(cfg, f, indent=2, ensure_ascii=False)


def apply_overrides(cfg, args):
    cfg = copy.deepcopy(cfg)
    if getattr(args, "epochs", None) is not None:
        cfg["train"]["epochs"] = args.epochs
    if getattr(args, "batch_size", None) is not None:
        cfg["dataset"]["batch_size"] = args.batch_size
    if getattr(args, "learning_rate", None) is not None:
        cfg["train"]["learning_rate"] = args.learning_rate
    if getattr(args, "device_target", None) is not None:
        cfg["device_target"] = args.device_target
    if getattr(args, "data_root", None) is not None:
        cfg["dataset"]["root"] = args.data_root
    if getattr(args, "output_dir", None) is not None:
        cfg["train"]["output_dir"] = args.output_dir
    return cfg


def pretty_print_config(cfg):
    print(json.dumps(cfg, indent=2, ensure_ascii=False))
