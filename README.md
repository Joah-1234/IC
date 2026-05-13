# SRTP 项目执行说明

本项目用于复现论文 `Token-Wise Asymmetric Contrastive Learning in Countering Unknown Attacks for Face Anti-Spoofing`，并为结项材料准备可追踪的实验产物。

## 一、当前目录结构

```text
srtp/
├── README.md
├── src/                  # 所有可执行代码
├── docs/
│   ├── paper/            # 论文原文
│   ├── materials/        # 立项、中期、答辩材料
│   └── notes/            # 过程记录
├── assets/
│   └── figures/          # 现有图片和可视化结果
├── checkpoints/          # 模型权重输出
├── results/              # 测试与汇总结果输出
└── data/                 # 建议的数据目录
```

## 二、核心文件位置

- `src/train.py`：训练主程序
- `src/test.py`：测试主程序
- `src/generate_report_assets.py`：生成结项常用图片材料
- `src/run_all_protocols.py`：依次跑四个跨域协议
- `src/summarize_results.py`：汇总四个协议的测试结果
- `src/config.py`：实验配置
- `src/model_loss.py`：模型和损失函数
- `src/ocim_dataset.py`：数据加载
- `src/preprocess_mtcnn_ocim.py`：人脸裁剪预处理
- `src/safe_sort.py`：将预处理后的数据整理为 `live/spoof`
- `docs/paper/Token-Wise_Asymmetric_Contrastive_Learning_in_Countering_Unknown_Attacks_for_Face_Anti-Spoofing.pdf`：目标论文
- `docs/materials/3230103107_运用非对称对比学习优化人脸反欺骗模型_立项PDF.pdf`
- `docs/materials/3230103107_运用非对称对比学习优化人脸反欺骗模型_中期检查表.pdf`
- `docs/materials/SRTP中期答辩(3).pptx`
- `docs/notes/1.md`

## 三、需要准备的数据集

论文复现默认使用 OCIM 四个子数据集：

1. `oulu-npu`
2. `CASIA-MFSD`
3. `replayattack`
4. `MSU-MFSD`

原始数据建议放成如下结构：

```text
data/raw/
├── oulu-npu/
├── CASIA-MFSD/
├── replayattack/
└── MSU-MFSD/
```

预处理后会生成两个阶段的数据：

```text
data/mtcnn_output/   # MTCNN 裁剪结果
data/processed/      # 最终训练/测试使用的数据
```

最终训练目录要求每个数据集都整理成：

```text
data/processed/
├── oulu-npu/
│   ├── live/
│   └── spoof/
├── CASIA-MFSD/
├── replayattack/
└── MSU-MFSD/
```

## 四、需要修改或确认的内容

路径配置和主要训练参数已经统一收口到 `src/config.py` 顶部的 `User Config` 区域。

默认情况下：

- 原始数据读取自 `data/raw/`
- MTCNN 中间结果输出到 `data/mtcnn_output/`
- 最终训练数据输出到 `data/processed/`

也就是说，原始数据不会被修改，所有新生成的数据都会保存在当前 `SRTP` 项目目录下。

通常只需要直接修改 `src/config.py` 前几十行，不需要再到多个脚本里分别改路径。

### 1. 你通常需要修改的内容

- `RAW_DATA_ROOT`
- `MTCNN_OUTPUT_ROOT`
- `PROCESSED_DATA_ROOT`
- `PRETRAINED_WEIGHT_PATH`
- `PROTOCOL_NAME`

### 2. 关于预训练权重

如果你没有预训练权重，可以直接把：

```python
PRETRAINED_WEIGHT_PATH = None
```

保持不变。

这种情况下代码仍然可以训练，但会有这些影响：

- 模型从随机初始化开始，收敛更慢
- 对数据量和训练轮数更敏感
- 最终指标通常会明显低于使用 ImageNet 预训练的版本
- 更难接近论文结果

对于 SRTP 结项来说，这不是“不能做”，但你需要在报告里明确写出这是复现偏差来源之一。

### 3. 协议切换

直接修改：

```python
PROTOCOL_NAME = "OCI_to_M"
```

支持四种协议：

- `OCI_to_M`
- `OMI_to_C`
- `OCM_to_I`
- `ICM_to_O`

## 五、推荐操作顺序

### 第一步：预处理原始数据

```bash
python3 src/preprocess_mtcnn_ocim.py
```

作用：

- 对原始图片做人脸裁剪
- 输出到 `SRTP_MTCNN_OUTPUT_ROOT`

### 第二步：整理 live/spoof 数据

```bash
python3 src/safe_sort.py
```

作用：

- 将 MTCNN 输出结果整理成训练所需目录结构
- 输出到 `SRTP_PROCESSED_ROOT`
- 自动生成数据集统计文件 `results/dataset_summary.json` 和 `results/dataset_summary.csv`

### 第三步：跑单个协议训练

```bash
export SRTP_PROTOCOL_NAME="OCI_to_M"
python3 src/train.py
```

作用：

- 训练模型
- 保存 `best/latest/epoch_x` checkpoint
- 保存实验配置和训练 loss 记录

### 第四步：测试单个协议

```bash
python3 src/test.py
```

作用：

- 在目标域上测试
- 生成 `AUC/EER/HTER`
- 保存逐样本预测表和特征文件

### 第五步：生成结项图片材料

```bash
python3 src/generate_report_assets.py
```

会生成：

- `reports/<protocol>/training_curves.png`
- `reports/<protocol>/roc_curve.png`
- `reports/<protocol>/confusion_matrix.png`
- `reports/<protocol>/tsne.png`
- `reports/<protocol>/sample_predictions.png`

### 第六步：批量跑四个协议

如果算力和时间允许：

```bash
python3 src/run_all_protocols.py
```

### 第七步：汇总结果

```bash
python3 src/summarize_results.py
```

会输出：

- `results/summary.csv`

## 六、操作后如何观察效果

### 1. 看训练是否正常

重点看：

- 终端是否出现 `Non-finite loss`
- `results/<protocol>/train_metrics.json` 中的 `avg_total_loss` 是否整体下降
- `checkpoints/<protocol>/` 是否生成了 `best.pth` 和 `latest.pth`

### 2. 看测试效果

重点看：

- `results/<protocol>/test_metrics.json`
- `results/<protocol>/test_predictions.csv`
- `reports/<protocol>/roc_curve.png`
- `reports/<protocol>/confusion_matrix.png`
- `reports/<protocol>/tsne.png`
- `reports/<protocol>/sample_predictions.png`
- 终端输出的 `AUC`、`EER`、`HTER`

一般来说：

- `AUC` 越高越好
- `EER` 越低越好
- `HTER` 越低越好

### 3. 看是否能直接用于结项

你最终至少要拿到这些文件：

- `results/<protocol>/experiment_config.json`
- `results/<protocol>/train_metrics.json`
- `results/<protocol>/test_metrics.json`
- `checkpoints/<protocol>/<protocol>_best.pth`
- `results/summary.csv`

这些文件可以直接支撑：

- 结项报告中的实验设置
- 结项报告中的结果表格
- 答辩时对训练过程和测试指标的说明

## 七、结项时建议展示的结果

建议至少展示：

1. 一个主协议的完整结果，例如 `OCI_to_M`
2. 四协议汇总表
3. 训练 loss 曲线
4. 典型可视化图，例如 t-SNE
5. 与论文结果的差异分析

## 八、当前评测说明

当前 `test.py` 中的 `HTER` 是按目标测试集上的 EER 阈值计算的近似结果。

如果论文原文使用开发集阈值，请在结项报告中明确写：

“本项目在复现中采用了测试集 EER 阈值近似计算 HTER，故与论文原始协议存在轻微差异。”
