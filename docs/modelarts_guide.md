# 华为云 ModelArts 运行说明

## 环境

本实验面向以下环境：

- 镜像：`mindspore_ascend:mindspore_2.7.2-cann_8.5.2-py_3.11-hce_2.0.2512-aarch64-snt9b`
- 规格：`1 * ascend-snt9b1 | 24 vCPUs | 192 GiB`
- Python：3.11
- MindSpore：2.7.2

## 1. 上传工程

将整个 `semanticnew` 文件夹上传到 ModelArts Notebook 工作目录，或用 Git/OBS 同步到云端。

进入项目根目录：

```bash
cd semanticnew
```

## 2. 下载 ADE20K

正式实验数据集使用 ADE20K / ADEChallengeData2016：

```bash
python scripts/download_ade20k.py --root data/ade20k
```

如果 Notebook 需要代理，可以使用：

```bash
export HTTP_PROXY=http://127.0.0.1:7890
export HTTPS_PROXY=http://127.0.0.1:7890
python scripts/download_ade20k.py --root data/ade20k --proxy http://127.0.0.1:7890
```

下载完成后检查：

```bash
python scripts/inspect_dataset.py --root data/ade20k --out outputs/ade20k_preview
```

## 3. 链路测试

先用合成数据跑 1 个 epoch，确认 MindSpore、Ascend、训练循环都正常：

```bash
python train.py --config configs/setr_synthetic_smoke.json --epochs 1 --device-target Ascend
python train.py --config configs/segformer_tiny_synthetic_smoke.json --epochs 1 --device-target Ascend
python train.py --config configs/segformer_edge_synthetic_smoke.json --epochs 1 --device-target Ascend
python train.py --config configs/mask2former_synthetic_smoke.json --epochs 1 --device-target Ascend
python train.py --config configs/oneformer_synthetic_smoke.json --epochs 1 --device-target Ascend
```

## 4. 正式训练

如果云端时间有限，优先使用快速复现实验配置：

```bash
python train.py --config configs/setr_tiny_ade20k_fast.json --device-target Ascend
python train.py --config configs/segformer_tiny_ade20k_fast.json --device-target Ascend
python train.py --config configs/segformer_edge_ade20k_fast.json --device-target Ascend
python train.py --config configs/mask2former_tiny_ade20k_fast.json --device-target Ascend
python train.py --config configs/oneformer_tiny_ade20k_fast.json --device-target Ascend
```

快速配置使用 `256 x 256` 输入、按固定随机种子从 ADE20K training 抽取 4000 张样本、从 validation 抽取 1000 张样本，训练 40 个 epoch，并采用 warmup + cosine 学习率衰减。该设置用于课程设计资源受限时的统一对比。若云端资源和时间充足，再使用完整配置：

```bash
python train.py --config configs/setr_tiny_ade20k.json --device-target Ascend
python train.py --config configs/segformer_tiny_ade20k.json --device-target Ascend
python train.py --config configs/segformer_edge_ade20k.json --device-target Ascend
python train.py --config configs/mask2former_tiny_ade20k.json --device-target Ascend
python train.py --config configs/oneformer_tiny_ade20k.json --device-target Ascend
```

如果显存不足，可以降低 batch size：

```bash
python train.py --config configs/segformer_tiny_ade20k.json --batch-size 2
```

如果想先快速确认 ADE20K 真实数据训练可跑，可以临时限制样本数：把配置中的 `max_train_samples` 设为 `128`，`max_val_samples` 设为 `64`。

## 5. 单独评估

```bash
python eval.py --config configs/segformer_edge_ade20k.json --checkpoint outputs/segformer_edge_ade20k/best.ckpt
python scripts/visualize_predictions.py --config configs/segformer_edge_ade20k.json --checkpoint outputs/segformer_edge_ade20k/best.ckpt --out outputs/segformer_edge_vis
```

五个模型训练完成后，可分别评估：

```bash
python eval.py --config configs/setr_tiny_ade20k.json --checkpoint outputs/setr_tiny_ade20k/best.ckpt
python eval.py --config configs/segformer_tiny_ade20k.json --checkpoint outputs/segformer_tiny_ade20k/best.ckpt
python eval.py --config configs/segformer_edge_ade20k.json --checkpoint outputs/segformer_edge_ade20k/best.ckpt
python eval.py --config configs/mask2former_tiny_ade20k.json --checkpoint outputs/mask2former_tiny_ade20k/best.ckpt
python eval.py --config configs/oneformer_tiny_ade20k.json --checkpoint outputs/oneformer_tiny_ade20k/best.ckpt
```

## 6. 输出文件

每次训练输出：

- `resolved_config.json`：实际使用配置。
- `metrics.csv`：每个 epoch 的 loss，以及按 `eval_every` 记录的 Pixel Accuracy、mIoU、类别 IoU。
- `best.ckpt`：验证集 mIoU 最优 checkpoint。
- `last.ckpt`：最后一个 epoch checkpoint。

## 7. 常见问题

1. 数据集文件找不到：确认 `data/ade20k/ADEChallengeData2016/images/training` 与 `annotations/training` 是否存在。
2. 下载慢：ADE20K 压缩包较大，网络慢时可先在本地下载 `ADEChallengeData2016.zip`，上传到 `data/ade20k` 后运行脚本解压或手动解压。
3. Ascend 编译耗时：第一次运行 GRAPH 模式会进行图编译，首个 step 可能明显较慢。
4. batch size 过大：降低 `--batch-size`，或将配置中的 `image_size` 改为 384 做快速实验。
5. Mask2Former-tiny / OneFormer-tiny 首次图编译较慢：这两个模型包含 query 解码器，第一次运行等待时间可能长于 SegFormer-tiny。
