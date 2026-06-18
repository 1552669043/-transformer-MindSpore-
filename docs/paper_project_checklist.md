# 论文、GitHub 项目与 ADE20K 结果核验表

本课设主题为“基于 Transformer 的图像语义分割方法研究”。为保证选题、数据集和对比结果符合课程要求，正式实验与报告采用以下口径。

## 1. 论文与项目来源

| 论文/方法 | 年份 | Transformer 分割类型 | 官方或主要 GitHub 项目 | 本课设使用方式 |
| --- | ---: | --- | --- | --- |
| SETR: Rethinking Semantic Segmentation from a Sequence-to-Sequence Perspective with Transformers | CVPR 2021；扩展版 IJCV 2024 | 纯 ViT encoder + segmentation decoder | https://github.com/fudan-zvg/SETR | 作为 `setr_tiny` 实现来源和纯 Transformer 基线 |
| SegFormer: Simple and Efficient Design for Semantic Segmentation with Transformers | NeurIPS 2021 | 层次化 Transformer encoder + MLP decoder | https://github.com/NVlabs/SegFormer | 作为 `segformer_tiny` 主复现来源 |
| Mask2Former: Masked-attention Mask Transformer for Universal Image Segmentation | CVPR 2022 | masked-attention mask Transformer | https://github.com/facebookresearch/Mask2Former | 作为 `mask2former_tiny` 的 mask-query 复现来源 |
| OneFormer: One Transformer to Rule Universal Image Segmentation | CVPR 2023 | task-conditioned universal segmentation Transformer | https://github.com/SHI-Labs/OneFormer | 作为 `oneformer_tiny` 的任务条件化 query 复现来源 |

说明：课程要求为“近五年来相关技术资料”。本报告按课程自然年份口径选取 2021-2026 年内代表性工作，其中 SETR 与 SegFormer 均为 2021 年正式会议论文，Mask2Former 为 2022 年，OneFormer 为 2023 年。

## 2. ADE20K 结果可比性

正式复现实验数据集统一使用 ADE20K / ADEChallengeData2016。上述论文的官方项目或论文均提供 ADE20K 实验结果，因此可以作为报告中的横向参考。

| 方法 | 原项目/论文中的 ADE20K 结果示例 | 可比性说明 |
| --- | --- | --- |
| SETR | 官方 GitHub 在 ADE20K val 上列出 SETR-Naive、SETR-MLA、SETR-PUP 等结果，例如 SETR-PUP `48.62 mIoU / 50.09 mIoU(ms+flip)` | 与本课设 `setr_tiny_ade20k` 同属 ADE20K 语义分割任务，但本课设为轻量版且训练设置更简化 |
| SegFormer | 论文报告 SegFormer-B4 在 ADE20K 上达到 `50.3% mIoU`；官方 GitHub 提供 ADE20K 训练、评估配置和预训练权重 | 与本课设 `segformer_tiny_ade20k` 结构思想一致，本课设用 MindSpore 轻量复现 |
| Mask2Former | 官方 Model Zoo 给出 ADE20K semantic segmentation 结果，例如 Swin-L(IN21k) `56.1 mIoU / 57.3 mIoU(ms+flip)` | 本课设实现面向 ADE20K 语义分割的轻量 mask-query 版本，不等同于官方完整通用分割实现 |
| OneFormer | 官方 README 给出 ADE20K 结果，例如 DiNAT-L `58.3 mIoU / 58.4 mIoU(ms+flip)` | 本课设实现面向 ADE20K 语义分割的轻量 task-token 版本，不等同于官方完整通用分割实现 |

## 3. 本课设的实现与改良关系

本课设不是简单调用原 PyTorch 项目，而是在 MindSpore 中实现轻量复现版本：

- `setr_tiny`：参考 SETR 的 patch tokenization、Transformer encoder 和上采样分割头。
- `segformer_tiny`：参考 SegFormer 的 overlap patch embedding、多阶段 Transformer encoder、spatial-reduction attention 和 MLP decoder。
- `segformer_edge`：在 `segformer_tiny` 基础上加入浅层边界辅助分支，这是本课设的改良方案。
- `mask2former_tiny`：参考 Mask2Former 的 query/mask 分类思想，实现面向 ADE20K 语义分割的轻量版本。
- `oneformer_tiny`：参考 OneFormer 的 task token 思想，在 query 解码中加入语义分割任务条件。

改良方案的损失函数为：

```text
L = L_seg + lambda * L_edge
```

其中 `L_seg` 为 ADE20K 150 类语义分割交叉熵，`L_edge` 为由语义标签自动生成的边界二分类交叉熵，默认 `lambda = 0.2`。

## 4. 报告中应避免的表述

- 不说“完全复现原论文精度”，因为本课设使用轻量模型、较少训练资源，并且不一定加载原论文同等预训练权重。
- 不把 Oxford-IIIT Pet、Penn-Fudan 等非论文主基准当作正式实验数据集。
- 不把 Mask2Former、OneFormer 写成官方级完整通用分割实现；本工程实现的是面向 ADE20K 语义分割的轻量复现版本。

## 5. 报告推荐表述

“本文以 ADE20K 作为统一实验数据集，参考 SETR、SegFormer、Mask2Former 与 OneFormer 官方开源项目，在 MindSpore 中实现轻量版 Transformer 语义分割模型，并在 SegFormer 基础上增加边界辅助监督分支。Mask2Former 与 OneFormer 的复现版本聚焦 ADE20K 语义分割任务，用于分析 mask-query 与任务条件化 query 在统一训练流程下的表现，并与其 ADE20K 公开结果进行参考对比。”
