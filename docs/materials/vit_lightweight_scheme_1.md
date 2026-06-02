# 方案一：ViT Backbone 轻量化替换实施说明

## 1. 修改目标

当前项目原始模型固定使用 `vit_base_patch16_224`。方案一的目标是在不改变现有 Token-Wise Asymmetric Contrastive Loss、ArcFace 分类头和训练/测试流程主体的前提下，将 Backbone 变成可配置项，使同一套代码可以直接切换：

- `vit_base_patch16_224`：原始基线；
- `vit_small_patch16_224`：推荐的首选轻量化版本；
- `vit_tiny_patch16_224`：极限压缩版本。

这种方案保留了 ViT 的 patch token 结构，因此对当前 `TACLoss` 的侵入最小，适合作为轻量化创新的第一阶段实验。

## 2. 已完成的代码改动

| 文件 | 修改内容 | 目的 |
|---|---|---|
| `src/config.py` | 新增 `BACKBONE_NAME`，并写入 `ExperimentConfig` 和 `Config.BACKBONE_NAME` | 将轻量化模型选择集中到配置文件中 |
| `src/model_loss.py` | `TACL_ViT` 新增 `backbone_name` 参数，使用 `timm.create_model(backbone_name, ...)` 创建主干 | 支持 ViT-Base/Small/Tiny 统一入口 |
| `src/model_loss.py` | 通过 `num_features` 或 `embed_dim` 自动推断特征维度 | 避免 ViT-Small/Tiny 仍使用固定 768 维 ArcFace 输入导致维度错误 |
| `src/train.py` | 创建模型时传入 `Config.BACKBONE_NAME`，并在日志中打印当前 backbone | 保证训练时配置可追踪 |
| `src/test.py` | 测试时优先读取 checkpoint 中保存的 `backbone_name` | 避免测试轻量模型 checkpoint 时误用默认 ViT-Base 结构 |
| `src/generate_vit_lightweight_table.py` | 新增轻量化对比表生成脚本 | 生成 CSV、Markdown 和 LaTeX 表格材料 |

## 3. 推荐实验配置

建议按以下顺序开展实验：

1. 当前默认恢复为 `BACKBONE_NAME = "vit_base_patch16_224"`，作为原始基线；
2. 如需轻量化主实验，再临时设置 `BACKBONE_NAME = "vit_small_patch16_224"`；
3. 如需极限压缩，再临时设置 `BACKBONE_NAME = "vit_tiny_patch16_224"`；
1. 保持 `BACKBONE_NAME = "vit_base_patch16_224"`，作为原始基线；
2. 修改为 `BACKBONE_NAME = "vit_small_patch16_224"`，作为主推轻量化模型；
3. 修改为 `BACKBONE_NAME = "vit_tiny_patch16_224"`，作为极限压缩对照；
4. 三组实验保持相同协议、输入尺寸、batch size、学习率、损失权重和训练轮数；
5. 汇总 AUC、EER、HTER、训练显存、单张推理耗时和参数量。

推荐优先报告 `ViT-Small/16`，因为它相比 ViT-Base 参数量下降明显，同时通常比 ViT-Tiny 更容易保留跨域泛化能力。

## 4. 需要执行的命令

```bash
# 生成轻量化方案表格
python src/generate_vit_lightweight_table.py

# 当前默认训练 ViT-Base；也可显式指定
SRTP_BACKBONE_NAME="vit_base_patch16_224" python src/train.py
# 训练当前 BACKBONE_NAME 指定的模型
python src/train.py

# 测试模型。脚本会优先使用 checkpoint 中保存的 backbone_name
python src/test.py
```

## 5. 输出表格位置

| 输出文件 | 用途 |
|---|---|
| `reports/paper_assets/vit_lightweight_backbone_table.csv` | 便于后续用表格软件编辑 |
| `reports/paper_assets/vit_lightweight_backbone_table.md` | 便于直接粘贴到 Markdown 报告 |
| `reports/paper_assets/vit_lightweight_backbone_table.tex` | 便于放入论文 LaTeX 表格 |

## 6. 注意事项

- 当前代码支持 `PRETRAINED_INIT_MODE = "timm"` / `"local"` / `"none"` 三种初始化方式；默认推荐 `"timm"`，即使用 timm 官方 ImageNet 预训练。
- 服务器上可用环境变量临时切换，例如 `SRTP_BACKBONE_NAME="vit_base_patch16_224" SRTP_PRETRAINED_INIT_MODE="timm" python src/train.py`。
- 服务器上可用环境变量临时切换，例如 `SRTP_BACKBONE_NAME="vit_small_patch16_224" SRTP_PRETRAINED_INIT_MODE="timm" python src/train.py`。
- 使用本地权重时，训练脚本会检查权重匹配率，默认低于 `50%` 会停止训练，避免继续使用错误初始化。
- 如果使用外部预训练权重，权重结构必须与当前 `BACKBONE_NAME` 匹配。例如 ViT-Base 权重不能直接完整加载到 ViT-Small。
- 当前表格中的参数量为常用近似值，适合方案设计和报告初稿；最终论文建议补充实际统计脚本或训练日志中的参数量。
- 若切换 backbone 后发现 checkpoint 加载失败，优先检查 checkpoint 中的 `config.backbone_name` 是否与当前模型结构一致。
- 若轻量模型精度下降明显，下一步建议加入知识蒸馏，用 ViT-Base checkpoint 作为教师模型指导 ViT-Small/Tiny。
