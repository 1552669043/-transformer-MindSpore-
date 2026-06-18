# 近五年代表性文献阅读与技术脉络

课设主题：基于 Transformer 的图像语义分割方法研究。

## 代表性论文

| 年份 | 论文 | 代表意义 | 官方/常用代码 |
| --- | --- | --- | --- |
| 2021 | SETR: Rethinking Semantic Segmentation from a Sequence-to-Sequence Perspective with Transformers | 将语义分割显式视为序列到序列预测，使用纯 Transformer 编码图像 patch，全局注意力贯穿每层，是 Transformer 进入语义分割的重要起点。官方项目提供 ADE20K 训练结果。 | https://github.com/fudan-zvg/SETR |
| 2021 | SegFormer: Simple and Efficient Design for Semantic Segmentation with Transformers | 提出层次化 Transformer 编码器与轻量 MLP 解码器，不使用位置编码，兼顾精度、效率和尺度泛化，是本课设主复现对象。官方项目提供 ADE20K 训练与评估配置。 | https://github.com/NVlabs/SegFormer |
| 2022 | Mask2Former: Masked-attention Mask Transformer for Universal Image Segmentation | 以 mask query 和 masked attention 统一语义、实例、全景分割，把像素分类逐步转向 mask 分类范式。 | https://github.com/facebookresearch/Mask2Former |
| 2023 | OneFormer: One Transformer to Rule Universal Image Segmentation | 通过 task token 和多任务联合训练，让同一模型一次训练后适配语义、实例、全景分割，体现统一分割框架的发展方向。 | https://github.com/SHI-Labs/OneFormer |

更详细的论文、GitHub 项目、ADE20K 结果对应关系见 `docs/paper_project_checklist.md`。

## 方法理解

### SETR

SETR 的核心思想是抛开传统 FCN 编码器的逐步下采样结构，将输入图像分割为固定大小 patch，并把 patch embedding 看成序列 token。Transformer 编码器在所有 token 间做全局自注意力，因此每一层都能直接建模长距离上下文。最后通过简单解码器把 token 还原为空间特征图并上采样为像素类别。

优点是全局建模直接、结构概念清晰；不足是计算量随 token 数二次增长，缺少卷积归纳偏置，小数据集从头训练时容易不稳定。

### SegFormer

SegFormer 使用 Mix Transformer 编码器生成四级多尺度特征。其注意力层引入 spatial reduction，在计算 key/value 前降低空间分辨率，减少注意力计算量。解码端只使用 MLP 将多层特征投影到统一通道数，再上采样、拼接、融合，避免复杂解码器。

本课设选择 SegFormer 作为主复现模型，原因是其结构适中，适合 MindSpore 手写复现，同时能体现 Transformer 分割方法从“纯全局注意力”走向“层次化高效建模”的演进。

### Mask2Former

Mask2Former 的关键变化是从逐像素分类转向 mask 分类：模型预测一组 mask query，每个 query 对应一个候选区域及其类别。masked attention 让 query 的交互区域集中在预测 mask 内，从而提升训练效率与区域一致性。该思想对边界和实例结构更友好，但实现复杂度明显高于 SETR/SegFormer。

### OneFormer

OneFormer 进一步把任务信息显式注入模型，通过 task token 指示当前要完成语义、实例还是全景分割，并用 query-text contrastive loss 改善类别与任务区分。它代表了近年统一分割模型方向，本课设实现其中面向语义分割任务的轻量 task token 复现版本。

## 本课设复现取舍

完整官方级复现 Mask2Former/OneFormer 需要 Detectron2 风格的数据管线、匈牙利匹配、mask query 损失等组件，工作量较大。为了保证课程设计实验可控，本工程采用 MindSpore 轻量复现版本，在统一 ADE20K 语义分割训练链路下比较以下方法：

- SETR-tiny：验证纯 Transformer 分割基线。
- SegFormer-tiny：主实验模型，体现高效层次化 Transformer。
- SegFormer-Edge：在 SegFormer 上加入边界辅助监督，作为改良方法。
- Mask2Former-tiny：复现 mask query 语义分割思路。
- OneFormer-tiny：复现 semantic task token 条件化 query 思路。

该选择覆盖了五类问题：全局上下文建模、多尺度特征融合、边界细节优化、mask query 分割范式、任务条件化统一分割思想。

## 资料来源

- SETR arXiv: https://arxiv.org/abs/2012.15840
- SegFormer arXiv: https://arxiv.org/abs/2105.15203
- Mask2Former arXiv: https://arxiv.org/abs/2112.01527
- OneFormer arXiv: https://arxiv.org/abs/2211.06220
- ADE20K / Scene Parsing Benchmark: https://sceneparsing.csail.mit.edu/
- MindSpore 2.7 `ops.interpolate`: https://www.mindspore.cn/docs/en/r2.7.0/api_python/ops/mindspore.ops.interpolate.html
- MindSpore 2.7 `CrossEntropyLoss`: https://www.mindspore.cn/docs/en/r2.7.0/api_python/nn/mindspore.nn.CrossEntropyLoss.html
- MindSpore 2.7 `GeneratorDataset`: https://www.mindspore.cn/docs/en/r2.7.0/api_python/dataset/mindspore.dataset.GeneratorDataset.html
