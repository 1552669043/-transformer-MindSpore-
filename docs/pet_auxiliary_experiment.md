# Pet 辅助数据集实验说明

## 数据集定位

`data/pet` 是一个宠物前景分割数据集，目录结构为：

```text
data/pet/
  images/
  masks/
  splits/
    train.txt
    val.txt
```

该数据集包含 7390 张图片和 7390 张像素级 mask，并已经提供训练/验证划分。相比 ADE20K 的 150 类场景解析任务，该数据集类别更少、目标更明确，更适合在课程设计资源有限时观察不同 Transformer 分割结构的相对差异。

报告中建议将其定位为“辅助实验”：

- ADE20K：论文基准主实验，用于和 SETR、SegFormer、Mask2Former、OneFormer 官方结果做对照。
- Pet：简化场景辅助实验，用于展示模型结构差异和可视化结果。

不要把 Pet 结果写成论文官方基准复现结果。

## 训练命令

单独训练：

```bash
python train.py --config configs/setr_tiny_pet_fast.json --device-target Ascend
python train.py --config configs/segformer_tiny_pet_fast.json --device-target Ascend
python train.py --config configs/segformer_edge_pet_fast.json --device-target Ascend
python train.py --config configs/mask2former_tiny_pet_fast.json --device-target Ascend
python train.py --config configs/oneformer_tiny_pet_fast.json --device-target Ascend
```

后台连续训练：

```bash
nohup bash scripts/run_pet_background.sh pet_fast_run > outputs/pet_fast_run.nohup.log 2>&1 &
tail -f outputs/pet_fast_run.nohup.log
```

训练结果会输出到：

```text
outputs/pet_fast_run/outputs/<experiment_name>/
  metrics.csv
  best.ckpt
  last.ckpt
  resolved_config.json
```

## 华为云部署流程

1. 本地把整个 `semanticnew` 文件夹压缩为 `semanticnew_pet.zip`，其中应包含 `data/pet`。
2. 上传 `semanticnew_pet.zip` 到 OBS 或 ModelArts Notebook。
3. 在 Notebook 中复制到本地高速盘 `/cache` 后解压：

```bash
mkdir -p /cache/course
cp /home/ma-user/work/<你的OBS挂载目录>/semanticnew_pet.zip /cache/course/
cd /cache/course
unzip -q semanticnew_pet.zip
cd semanticnew
```

4. 检查数据：

```bash
ls data/pet/images | head
ls data/pet/masks | head
head data/pet/splits/train.txt
```

5. 后台启动训练：

```bash
mkdir -p outputs
nohup bash scripts/run_pet_background.sh pet_fast_run > outputs/pet_fast_run.nohup.log 2>&1 &
tail -f outputs/pet_fast_run.nohup.log
```

6. 训练结束后，把结果复制回 OBS：

```bash
cp -r outputs/pet_fast_run /home/ma-user/work/<你的OBS挂载目录>/runs/
```
