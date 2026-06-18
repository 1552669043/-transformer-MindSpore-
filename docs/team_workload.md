# 三人工作量与分工

报告中可以将姓名和学号替换到下表。

| 成员 | 主要任务 | 交付内容 |
| --- | --- | --- |
| 成员 A | 文献调研与理论分析 | 阅读 SETR、SegFormer、Mask2Former、OneFormer；整理 Transformer 分割技术路线；撰写相关工作和原理分析章节。 |
| 成员 B | 数据集与实验流程 | 下载并整理 ADE20K；完成标签映射、数据预览、训练配置、指标记录；整理华为云 ModelArts 环境和运行步骤。 |
| 成员 C | 模型实现与改良实验 | 使用 MindSpore 实现 SETR-tiny、SegFormer-tiny、SegFormer-Edge、Mask2Former-tiny、OneFormer-tiny；完成训练、评估、对比表格与结论分析。 |

## 具体任务拆分

### 成员 A

- 选择近五年代表论文，说明选择依据。
- 对比纯 Transformer、层次化 Transformer、mask query、统一分割框架。
- 总结各方法优缺点和适用场景。

### 成员 B

- 搭建华为云 ModelArts 实验环境。
- 下载并检查 ADE20K / ADEChallengeData2016 数据集。
- 完成 ADE20K 标注从 `1-150` 到 `0-149` 的类别映射，并处理 ignore label。
- 生成数据样例图，用于报告中的图文说明。
- 记录训练日志和指标。

### 成员 C

- 编写 MindSpore 模型代码。
- 调整超参数并完成五组 ADE20K 实验。
- 实现边界辅助监督改良方案。
- 分析 mIoU、Pixel Accuracy、per-class IoU 和边界可视化变化。
- 补充 Mask2Former / OneFormer 的轻量语义分割复现，并说明与官方完整实现的差异。

## 报告中的创新说明写法

已有工作：

- SETR、SegFormer、Mask2Former、OneFormer 的核心思想均来自公开论文。
- ADE20K 数据集来自公开语义分割基准。

本课设实现与改良：

- 使用 MindSpore 从零实现轻量版 SETR、SegFormer、Mask2Former 与 OneFormer，并在 Ascend 环境训练。
- 基于 SegFormer 增加浅层边界辅助分支，构造 `SegFormer-Edge`。
- 在 ADE20K 同一数据集、同一训练配置下对纯 Transformer、层次化 Transformer、边界辅助模型、mask query 模型和任务条件化 query 模型进行对比。
