# 实验部分正文草稿

## 实验环境

本实验基于华为云 ModelArts 平台完成，使用 MindSpore Ascend 镜像 `mindspore_ascend:mindspore_2.7.2-cann_8.5.2-py_3.11-hce_2.0.2512-aarch64-snt9b`，实例规格为 `1 * ascend-snt9b1 | 24 vCPUs | 192 GiB`。实验代码使用 MindSpore 实现 Transformer 语义分割模型的训练、验证与指标统计。

## 数据集

正式实验采用 ADE20K / ADEChallengeData2016 数据集。ADE20K 是语义分割和场景解析领域常用公开基准，SETR、SegFormer、Mask2Former 等代表性 Transformer 分割论文均在该数据集上报告结果，因此适合作为本文复现与改良实验的统一数据集。数据集包含 150 个语义类别，训练集用于模型参数学习，验证集用于计算 Pixel Accuracy、mean IoU 与 per-class IoU。

数据预处理包括图像缩放、归一化和标签映射。原始 ADE20K 标注中像素值 `0` 表示忽略区域，实验中设为 `ignore_index=255`；像素值 `1-150` 映射为模型类别 `0-149`。输入图像统一缩放为 `512 x 512`，并使用 ImageNet 均值和方差归一化。

## 对比方法

实验实现五个模型。第一，SETR-tiny 作为纯 Transformer 基线，将图像划分为 patch 序列，通过多层 Transformer 编码器建模全局上下文，再由轻量卷积解码头恢复像素级分类结果。第二，SegFormer-tiny 作为主复现模型，采用层次化 Transformer 编码器和 MLP 解码器，能够融合多尺度特征。第三，SegFormer-Edge 为本文改良方案，在 SegFormer-tiny 的浅层特征上增加边界辅助分支，通过边界监督增强模型对类别交界区域的关注。第四，Mask2Former-tiny 参考 mask query 分割范式，使用 query 解码器预测类别和 mask，并聚合为语义分割 logits。第五，OneFormer-tiny 在 query 解码器中加入语义分割任务 token，用于模拟 OneFormer 的任务条件化建模思想。

需要说明的是，Mask2Former 与 OneFormer 的官方实现面向语义、实例、全景分割等通用分割任务，包含更复杂的数据组织、集合匹配损失和多任务训练机制。本文实现的是面向 ADE20K 语义分割的轻量复现版本，重点用于比较不同 Transformer 分割范式在统一训练流程下的表现。

SegFormer-Edge 的总损失定义为：

```text
L = L_seg + lambda * L_edge
```

其中 `L_seg` 是 150 类语义分割交叉熵损失，`L_edge` 是边界二分类交叉熵损失，`lambda` 取 0.2。边界标签由语义标注自动生成：若像素与上下左右相邻像素类别不同，则标记为边界像素。

## 训练设置

五个模型均使用 AdamWeightDecay 优化器训练。SETR-tiny 初始学习率设为 `1e-4`，SegFormer-tiny、SegFormer-Edge、Mask2Former-tiny 与 OneFormer-tiny 初始学习率设为 `6e-5`，weight decay 设为 `0.01`。考虑到 SETR 在 `512 x 512` 输入下自注意力显存占用较高，SETR-tiny 的 batch size 设为 2，其余模型 batch size 设为 4，训练 80 个 epoch。训练过程中每个 epoch 显示百分比进度条，正式 ADE20K 配置默认每 5 个 epoch 在 validation split 上计算指标，并保存 mIoU 最优的 checkpoint。训练完成后，使用可视化脚本从验证集中保存原图、真实标注和预测结果的横向拼接图，用于观察模型是否能正确识别天空、道路、建筑、人体、家具等常见语义类别，以及类别边界处是否存在粘连。

若云端资源和训练时间不足，实验可采用快速复现设置：输入尺寸调整为 `256 x 256`，训练集按固定随机种子从 ADE20K training 中抽取 4000 张图像，验证集从 validation 中抽取 1000 张图像，训练 40 个 epoch，并采用 1 个 epoch warmup 后进行 cosine 学习率衰减。五个模型使用相同随机种子、相同数据规模和相同验证频率。该设置不能与原论文完整 ADE20K 结果直接比较绝对精度，但可以用于分析不同模型结构在同一资源约束下的相对趋势。

## 实验结果记录

| 方法 | 数据集 | mIoU | Pixel Acc | 训练轮数 | 备注 |
| --- | --- | ---: | ---: | ---: | --- |
| SETR-tiny | ADE20K val | 待填 | 待填 | 80 | 纯 Transformer 基线 |
| SegFormer-tiny | ADE20K val | 待填 | 待填 | 80 | 层次化 Transformer |
| SegFormer-Edge | ADE20K val | 待填 | 待填 | 80 | 加入边界辅助监督 |
| Mask2Former-tiny | ADE20K val | 待填 | 待填 | 80 | mask query 语义分割 |
| OneFormer-tiny | ADE20K val | 待填 | 待填 | 80 | 加入语义任务 token |

## 结果分析写作要点

完成云端训练后，可从以下角度分析结果：

1. 若 SegFormer-tiny 的 mIoU 高于 SETR-tiny，说明层次化多尺度特征对 ADE20K 场景解析更有利。
2. 若 SegFormer-Edge 的 mIoU 与 SegFormer-tiny 接近但边界可视化更清晰，说明边界辅助监督对轮廓质量有帮助。
3. 若 SegFormer-Edge 的整体 mIoU 下降，需要分析边界损失权重、边界像素占比和 150 类长尾分布造成的优化冲突。
4. 若 Mask2Former-tiny 和 OneFormer-tiny 的结果不及官方论文，需要说明本文没有采用官方完整集合预测损失、预训练权重和多任务训练设置，重点比较的是轻量统一实现下的趋势。
5. 与原论文结果比较时，应明确本实验为课程设计轻量复现，模型规模、预训练权重、训练轮数和数据增强策略与原论文不同，因此重点关注方法趋势和改良方案的相对变化。
