# 基于 Transformer 的图像语义分割方法研究

本目录补充《视觉计算》课程设计的实验部分，主题固定为“基于 Transformer 的图像语义分割方法研究”。工程面向华为云 ModelArts / Ascend 环境：

- 镜像：`mindspore_ascend:mindspore_2.7.2-cann_8.5.2-py_3.11-hce_2.0.2512-aarch64-snt9b`
- 规格：`1 * ascend-snt9b1 | 24 vCPUs | 192 GiB`
- 建议入口：`train.py`、`eval.py`

## 实验对象

本工程围绕论文常用标准数据集 ADE20K 搭建复现实验，并实现五个模型：

1. `setr_tiny`：参考 SETR，将图像切成 patch 后用纯 Transformer 编码，再用轻量解码头恢复像素预测。
2. `segformer_tiny`：参考 SegFormer，用层次化 Transformer 编码器和 MLP 解码器聚合多尺度特征。
3. `segformer_edge`：在 `segformer_tiny` 上增加边界辅助分支，用边界监督改善物体轮廓区域的预测，作为课程设计改良方案。
4. `mask2former_tiny`：参考 Mask2Former 的 mask query 思路，用查询向量预测类别与 mask，并聚合为语义分割 logits。
5. `oneformer_tiny`：参考 OneFormer 的 task-conditioned query 思路，在 mask query 解码器中加入语义分割任务 token。

其中 Mask2Former 与 OneFormer 在官方项目中属于更完整的通用分割框架；本工程实现的是适配 ADE20K 语义分割、便于在 MindSpore / Ascend 环境中训练的轻量复现版本。

正式实验数据集为 ADE20K / ADEChallengeData2016。SETR、SegFormer、Mask2Former 等代表性论文都在 ADE20K 上报告语义分割结果，因此该数据集适合做论文复现基础上的统一对比。

## 快速运行

在 ModelArts Notebook 中进入项目目录后：

```bash
python scripts/download_ade20k.py --root data/ade20k
python scripts/inspect_dataset.py --root data/ade20k --out outputs/ade20k_preview

python train.py --config configs/setr_tiny_ade20k.json
python train.py --config configs/segformer_tiny_ade20k.json
python train.py --config configs/segformer_edge_ade20k.json
python train.py --config configs/mask2former_tiny_ade20k.json
python train.py --config configs/oneformer_tiny_ade20k.json

python eval.py --config configs/segformer_edge_ade20k.json --checkpoint outputs/segformer_edge_ade20k/best.ckpt
python scripts/visualize_predictions.py --config configs/segformer_edge_ade20k.json --checkpoint outputs/segformer_edge_ade20k/best.ckpt --out outputs/segformer_edge_vis
```

如果云端训练时间有限，可以先使用快速复现实验配置。该配置使用固定随机种子抽取 ADE20K 子集、`256 x 256` 输入、40 个 epoch 和 cosine 学习率衰减，用于在有限资源下得到五个模型的可比结果：

```bash
python train.py --config configs/setr_tiny_ade20k_fast.json --device-target Ascend
python train.py --config configs/segformer_tiny_ade20k_fast.json --device-target Ascend
python train.py --config configs/segformer_edge_ade20k_fast.json --device-target Ascend
python train.py --config configs/mask2former_tiny_ade20k_fast.json --device-target Ascend
python train.py --config configs/oneformer_tiny_ade20k_fast.json --device-target Ascend
```

如果需要一个更容易收敛、结果更直观的辅助实验，可使用 `data/pet` 宠物分割数据集：

```bash
python scripts/inspect_dataset.py --dataset pet --root data/pet --out outputs/pet_preview
nohup bash scripts/run_pet_background.sh pet_fast_run > outputs/pet_fast_run.nohup.log 2>&1 &
tail -f outputs/pet_fast_run.nohup.log
```

如果需要先验证训练链路但暂时不下载 ADE20K：

```bash
python train.py --config configs/setr_synthetic_smoke.json --epochs 1 --device-target Ascend
python train.py --config configs/segformer_tiny_synthetic_smoke.json --epochs 1 --device-target Ascend
python train.py --config configs/segformer_edge_synthetic_smoke.json --epochs 1 --device-target Ascend
python train.py --config configs/mask2former_synthetic_smoke.json --epochs 1 --device-target Ascend
python train.py --config configs/oneformer_synthetic_smoke.json --epochs 1 --device-target Ascend
```

## 目录说明

- `configs/`：实验配置文件。
- `src/`：MindSpore 数据集、模型、损失、指标实现。
- `scripts/`：ADE20K 下载、数据预览、配置检查脚本。
- `docs/`：文献阅读、论文项目核验表、实验方案、ModelArts 运行说明、三人分工、报告实验章节草稿。
- `reports/`：实验结果记录模板。

## 本地说明

当前本地 Windows 环境没有安装 MindSpore，本工程已做 Python 语法检查；完整训练需要在上述华为云 Ascend 镜像中运行。
