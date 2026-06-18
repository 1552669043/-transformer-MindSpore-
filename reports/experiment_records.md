# 实验记录

## 环境

- 云平台：华为云 ModelArts
- 镜像：`mindspore_ascend:mindspore_2.7.2-cann_8.5.2-py_3.11-hce_2.0.2512-aarch64-snt9b`
- 规格：`1 * ascend-snt9b1 | 24 vCPUs | 192 GiB`
- 数据集：ADE20K / ADEChallengeData2016
- 类别数：150
- 输入尺寸：`512 x 512`

## 训练记录

| 日期 | 方法 | epoch | batch size | best mIoU | Pixel Acc | 备注 |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| 待填 | SETR-tiny | 80 | 4 | 待填 | 待填 | 纯 Transformer 基线 |
| 待填 | SegFormer-tiny | 80 | 4 | 待填 | 待填 | 主复现模型 |
| 待填 | SegFormer-Edge | 80 | 4 | 待填 | 待填 | 加边界辅助监督 |
| 待填 | Mask2Former-tiny | 80 | 4 | 待填 | 待填 | mask query 语义分割 |
| 待填 | OneFormer-tiny | 80 | 4 | 待填 | 待填 | 语义任务 token 条件化 |

## 与论文结果对照

| 方法 | 论文/代码报告数据集 | 论文 mIoU | 本课设实现 mIoU | 差异原因 |
| --- | --- | ---: | ---: | --- |
| SETR | ADE20K val | 待查阅填写 | 待填 | 本课设为轻量版，从零训练，训练轮数和预训练设置不同 |
| SegFormer | ADE20K val | 待查阅填写 | 待填 | 本课设为轻量版，未加载 ImageNet 预训练权重 |
| Mask2Former | ADE20K val | 待查阅填写 | 待填 | 本课设为轻量语义分割复现，未采用官方完整集合匹配损失 |
| OneFormer | ADE20K val | 待查阅填写 | 待填 | 本课设聚焦 semantic task token 条件化语义分割 |

## 现象分析

待训练完成后填写：

1. SETR-tiny 的 loss 曲线是否平稳下降。
2. SegFormer-tiny 相比 SETR-tiny 是否具有更高 mIoU。
3. SegFormer-Edge 是否在总体 mIoU 不下降的前提下改善边界可视化效果。
4. 若边界辅助导致整体 mIoU 下降，尝试降低 `edge_loss_weight` 到 0.1 或延后启用边界损失。
5. Mask2Former-tiny 与 OneFormer-tiny 是否能体现 mask query / task token 范式的训练趋势。
6. 对比论文报告结果时，说明本课设轻量实现与原论文在预训练、模型规模、训练轮数、数据增强、集合匹配损失上的差异。
