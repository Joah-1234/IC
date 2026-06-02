# ICM_to_O 初步结果与预训练权重修正说明

## 1. 当前结果

当前 `ICM_to_O` 协议的初步测试结果如下：

| 指标 | 数值 | 说明 |
|---|---:|---|
| Target domain | `oulu-npu` | 目标域为 OULU-NPU |
| Samples | 327306 | 测试样本总数 |
| Live / Spoof | 64708 / 262598 | 测试集类别明显不均衡 |
| AUC | 0.8062 | 排序能力中等，说明模型已学到部分真假脸区分信息 |
| EER | 0.2681 | 错误率仍偏高 |
| HTER | 0.2681 | 当前实现中 HTER 使用测试集 EER 阈值近似 |
| Threshold | 0.9670 | 阈值非常靠近 spoof 类高分区域，说明输出分数校准不理想 |

## 2. 问题判断

该结果不是完全随机，但距离理想跨域泛化仍有明显差距。结合当前代码，最优先怀疑的是 **ViT backbone 初始化质量不足**：

1. 修改前 `TACL_ViT` 一直以 `pretrained=False` 创建 timm 模型；
2. 当 `PRETRAINED_WEIGHT_PATH = None` 时，模型会从随机初始化开始训练；
3. 人脸反欺骗数据虽然样本量看似较大，但跨域泛化强依赖纹理、颜色和局部模式，随机初始化 ViT 通常更难稳定学到可迁移特征；
4. 当前 `Threshold = 0.9670` 偏高，说明分类头和特征空间的分数校准仍不稳定，预训练初始化通常能缓解这一问题。

因此这里应理解为“预训练权重”问题，而不是“MTCNN/图像预处理”问题。图像预处理仍可能影响结果，但本次优先修复 backbone 预训练初始化。

## 3. 已做代码修正

| 文件 | 修正内容 | 作用 |
|---|---|---|
| `src/config.py` | 新增 `PRETRAINED_INIT_MODE`，默认使用 `timm` 官方 ImageNet 预训练 | 避免无意中从随机 ViT 开始训练 |
| `src/config.py` | 支持环境变量 `SRTP_BACKBONE_NAME`、`SRTP_PRETRAINED_INIT_MODE`、`SRTP_PRETRAINED_WEIGHT_PATH`、`SRTP_MIN_PRETRAINED_MATCH_RATIO` | 服务器上可不改代码直接切换配置 |
| `src/model_loss.py` | `TACL_ViT` 支持 `timm` / `local` / `none` 三种初始化模式 | 同时兼容联网下载、本地权重和随机初始化 |
| `src/train.py` | 日志打印预训练模式、来源、backbone 和特征维度 | 训练开始即可确认是否真的加载了预训练 |
| `src/generate_report_assets.py` | 可视化加载 checkpoint 时读取保存的 `backbone_name` | 避免轻量 ViT checkpoint 生成 t-SNE 时结构不匹配 |

## 4. 推荐重新训练命令

### 4.1 服务器可联网或已有 timm 缓存

```bash
export SRTP_BACKBONE_NAME="vit_base_patch16_224"
export SRTP_PRETRAINED_INIT_MODE="timm"
python src/train.py
python src/test.py
python src/generate_report_assets.py
python src/summarize_results.py
```

训练日志中应出现：

```text
Pretrain mode : timm
Backbone initialized from timm ImageNet pretrained weights.
  -> Pretrain source: timm
```

### 4.2 服务器不能联网，使用本地权重

先把与 backbone 匹配的权重放到服务器，例如：

```text
weights/vit_base_patch16_224.pth
weights/vit_base_patch16_224.safetensors
```

然后运行：

```bash
export SRTP_BACKBONE_NAME="vit_base_patch16_224"
export SRTP_PRETRAINED_INIT_MODE="local"
export SRTP_PRETRAINED_WEIGHT_PATH="weights/vit_base_patch16_224.pth"
export SRTP_PRETRAINED_WEIGHT_PATH="weights/vit_base_patch16_224.safetensors"
python src/train.py
```

训练日志中应重点看：

```text
matched=.../...，missing=...，unexpected=...
```

如果 `matched` 很低，说明权重和当前 backbone 不匹配。当前训练脚本默认要求本地权重匹配率至少达到 `50%`，可通过 `SRTP_MIN_PRETRAINED_MATCH_RATIO` 调整。

### 4.3 Base/Tiny 对照实验建议

当前主实验默认恢复为 ViT-Base。若需要做轻量化或极限压缩对照实验，可临时切换 backbone，但权重也必须同步匹配：

```bash
# 轻量化 Small 对照
### 4.3 轻量化实验建议

在确认 ViT-Base + 预训练能够改善结果后，再切换轻量模型：

```bash
export SRTP_BACKBONE_NAME="vit_small_patch16_224"
export SRTP_PRETRAINED_INIT_MODE="timm"
python src/train.py
python src/test.py

# 极限压缩对照
export SRTP_BACKBONE_NAME="vit_tiny_patch16_224"
export SRTP_PRETRAINED_INIT_MODE="timm"
python src/train.py
python src/test.py
```

不要混用 ViT-Base、ViT-Small、ViT-Tiny 的本地权重；本地权重必须与 `SRTP_BACKBONE_NAME` 对应。
```

不要把 ViT-Base 的本地权重直接加载到 ViT-Small/Tiny；本地权重必须与 `SRTP_BACKBONE_NAME` 对应。

## 5. 后续观察重点

重新训练后建议重点比较：

1. AUC 是否从 `0.8062` 提升；
2. EER/HTER 是否明显低于 `0.2681`；
3. Threshold 是否不再极端靠近 `1.0`；
4. ROC 曲线是否更贴近左上角；
5. t-SNE 中 live/spoof 是否分离更清晰；
6. `weight_load_report` 中是否显示 `source=timm` 或本地权重 `matched_ratio` 较高。
