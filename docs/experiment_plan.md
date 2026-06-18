# 实验方案

## 1. 实验目标

围绕“基于 Transformer 的图像语义分割方法研究”，完成以下实验目标：

1. 阅读并分析近五年代表性 Transformer 分割方法。
2. 基于 MindSpore 搭建可运行的 ADE20K 语义分割训练与评估流程。
3. 在 ADE20K 上复现轻量版 SETR、SegFormer、Mask2Former 与 OneFormer，比较纯 Transformer、层次化 Transformer、mask query 与任务条件化 query 的差异。
4. 提出并实现 SegFormer-Edge：在 SegFormer 的浅层特征上加入边界预测分支，通过辅助损失提升物体轮廓区域分割质量。
5. 使用 mIoU、Pixel Accuracy、per-class IoU、训练耗时和结构复杂度进行对比分析。

## 2. 数据集

正式实验使用 ADE20K / ADEChallengeData2016。ADE20K 是 SETR、SegFormer、Mask2Former 等论文常用的场景解析与语义分割基准，包含 150 个语义类别。数据集划分为 training 与 validation，本工程按论文常用做法使用 training 训练、validation 评估。

标注处理方式：

- 原始 ADE20K 标注像素值为 `0-150`。
- `0` 表示忽略区域，在训练和评估中设为 `ignore_index=255`。
- `1-150` 映射为模型类别 `0-149`。

下载与检查：

```bash
python scripts/download_ade20k.py --root data/ade20k
python scripts/inspect_dataset.py --root data/ade20k --out outputs/ade20k_preview
```

## 3. 模型设置

### SETR-tiny

输入图像缩放到 `512 x 512`，patch size 为 16，因此 token 数为 `32 x 32 = 1024`。Transformer 编码器深度为 6，embedding 维度为 192，最后使用两层卷积解码头上采样到输入分辨率。

### SegFormer-tiny

编码器输出四级特征，通道数为 `[32, 64, 160, 256]`。每一级使用 overlap patch embedding 和 spatial-reduction self-attention。解码器将四级特征通过 MLP 投影到统一维度，上采样到 `1/4` 分辨率后拼接融合，再恢复到输入大小。

### SegFormer-Edge

在 SegFormer-tiny 的第一级浅层特征上增加二分类边界分支。训练损失为：

```text
L = L_seg + lambda * L_edge
```

其中 `L_seg` 为 ADE20K 150 类语义分割交叉熵，`L_edge` 为边界二分类交叉熵，默认 `lambda = 0.2`。

改良动机：Transformer 具有较强的全局上下文建模能力，但语义分割中物体边界常出现粘连和模糊。浅层特征保留更多局部纹理，边界辅助监督能约束模型关注类别交界处，从而尝试改善边界像素预测。

### Mask2Former-tiny

Mask2Former 的完整官方实现使用 mask query、masked attention、集合预测损失与多任务分割训练。本课设为了在 MindSpore / Ascend 环境中保持训练链路可控，采用轻量语义分割复现：先用层次化 Transformer 编码器和 pixel decoder 得到高分辨率 mask feature，再用一组可学习 query 通过 Transformer 解码器预测 query 类别和 query mask，最后将二者聚合为 ADE20K 的 150 类语义分割 logits，并继续使用像素级交叉熵训练。

### OneFormer-tiny

OneFormer 的核心思想是使用任务条件化 token 统一语义、实例和全景分割。本课设聚焦语义分割任务，因此在 Mask2Former-tiny 的 query 解码器基础上加入固定的 semantic task token，让 query 在解码前接收任务条件信息。该实现用于和 Mask2Former-tiny 对比 task-conditioned query 对语义分割训练的影响。

## 4. 训练配置

推荐在华为云 Ascend 环境运行：

```bash
python train.py --config configs/setr_tiny_ade20k.json
python train.py --config configs/segformer_tiny_ade20k.json
python train.py --config configs/segformer_edge_ade20k.json
python train.py --config configs/mask2former_tiny_ade20k.json
python train.py --config configs/oneformer_tiny_ade20k.json
```

关键超参数：

| 配置 | 值 |
| --- | --- |
| 输入尺寸 | `512 x 512` |
| batch size | SETR 为 2，其余模型为 4 |
| epoch | 80 |
| optimizer | AdamWeightDecay |
| SETR 学习率 | `1e-4` |
| SegFormer / Mask2Former / OneFormer 学习率 | `6e-5` |
| weight decay | `0.01` |
| 指标 | Pixel Accuracy, mIoU, per-class IoU |

## 5. 结果记录模板

训练完成后，每个实验会在 `outputs/<experiment_name>/metrics.csv` 中记录每个 epoch 的 loss；正式 ADE20K 配置默认每 5 个 epoch 在 validation split 上计算 Pixel Accuracy、mIoU 和 per-class IoU，最后一个 epoch 会强制评估一次。

报告中建议整理为：

| 方法 | 数据集 | mIoU | Pixel Acc | 训练轮数 | 现象分析 |
| --- | --- | ---: | ---: | ---: | --- |
| SETR-tiny | ADE20K val | 待填 | 待填 | 80 | 全局建模直接，但计算开销较大 |
| SegFormer-tiny | ADE20K val | 待填 | 待填 | 80 | 多尺度特征更适合场景解析 |
| SegFormer-Edge | ADE20K val | 待填 | 待填 | 80 | 重点观察边界区域与总体 mIoU 变化 |
| Mask2Former-tiny | ADE20K val | 待填 | 待填 | 80 | 用 mask query 聚合语义预测 |
| OneFormer-tiny | ADE20K val | 待填 | 待填 | 80 | 在 query 解码中加入语义任务 token |

## 6. 预期结论

根据方法结构，可以预期：

1. SETR-tiny 能验证纯 Transformer 分割可行性，但由于缺少层次化特征，训练效率和局部细节恢复可能弱于 SegFormer。
2. SegFormer-tiny 通过层次化特征和轻量解码器，在 ADE20K 场景解析任务中更均衡，应作为主要复现基线。
3. SegFormer-Edge 若整体 mIoU 不下降且边界可视化更清晰，可以说明边界辅助监督对 Transformer 分割有一定改进价值；若 mIoU 下降，则需分析辅助损失权重、边界类别不平衡和从零训练稳定性。
4. Mask2Former-tiny 与 OneFormer-tiny 能体现近年来分割方法从逐像素分类向 query/mask 分类范式发展的趋势；若结果不及 SegFormer-tiny，需要结合轻量实现、未使用预训练权重和未采用完整集合匹配损失进行说明。
