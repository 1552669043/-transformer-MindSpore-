import mindspore as ms


def configure_runtime(cfg):
    mode_name = cfg.get("mode", "GRAPH").upper()
    mode = ms.GRAPH_MODE if mode_name == "GRAPH" else ms.PYNATIVE_MODE
    ms.set_context(mode=mode, device_target=cfg.get("device_target", "Ascend"))
    ms.set_seed(int(cfg.get("seed", 42)))
