# ViT-Base 本地匹配权重准备说明

当前主实验默认恢复为 `vit_base_patch16_224`。因此本地预训练权重也必须是 ViT-Base 对应权重，不能把 ViT-Small 或 ViT-Tiny 权重直接加载到 ViT-Base。

## 1. 在本地电脑准备 Python 环境

建议在本地电脑新建一个轻量环境，只用于下载并保存 timm 预训练权重：

```bash
python -m venv .venv
source .venv/bin/activate  # Windows PowerShell 可用: .venv\Scripts\Activate.ps1
pip install torch timm
```

如果本地电脑没有 GPU 也没关系；下载和保存权重只需要 CPU 版 PyTorch 即可。

## 2. 在本地电脑下载并保存 Base 权重

在项目根目录运行：

```bash
python src/download_timm_backbone_weights.py \
  --backbone vit_base_patch16_224 \
  --output weights/vit_base_patch16_224.pth
```

该命令会创建：

```text
weights/vit_base_patch16_224.pth
```

文件中保存的是 `vit_base_patch16_224` 的 `state_dict`，与当前默认 `BACKBONE_NAME` 匹配。

## 3. 上传到服务器

如果是在本地电脑下载的，把权重上传到服务器项目目录：

```bash
scp weights/vit_base_patch16_224.pth 用户名@服务器IP:/服务器项目路径/weights/
```

如果使用 VSCode Remote SSH，也可以直接把该文件拖到远程项目的 `weights/` 目录。

## 4. 服务器训练时使用本地 Base 权重

在服务器项目根目录运行：

```bash
export SRTP_BACKBONE_NAME="vit_base_patch16_224"
export SRTP_PRETRAINED_INIT_MODE="local"
export SRTP_PRETRAINED_WEIGHT_PATH="weights/vit_base_patch16_224.pth"
python src/train.py
```

训练日志中应出现类似：

```text
Backbone      : vit_base_patch16_224
Pretrain mode : local
Backbone weights loaded with strict=False.
  -> Pretrain source: local
  -> Weight path: weights/vit_base_patch16_224.pth
  -> Load summary: matched=.../...
```

如果匹配率过低，训练脚本会停止并提示检查 `PRETRAINED_WEIGHT_PATH` 是否与 `BACKBONE_NAME` 匹配。

## 5. 服务器可联网时的更简单方式

如果服务器可以直接下载 timm 权重，不需要提前保存本地权重，直接运行：

```bash
export SRTP_BACKBONE_NAME="vit_base_patch16_224"
export SRTP_PRETRAINED_INIT_MODE="timm"
python src/train.py
```

这种方式会通过 `timm.create_model("vit_base_patch16_224", pretrained=True, num_classes=0)` 初始化 backbone。

## 6. 如需轻量化 Small 对照

只有在明确要跑轻量化对照实验时，才切换为 Small，并准备对应 Small 权重：

```bash
python src/download_timm_backbone_weights.py \
  --backbone vit_small_patch16_224 \
  --output weights/vit_small_patch16_224.pth

export SRTP_BACKBONE_NAME="vit_small_patch16_224"
export SRTP_PRETRAINED_INIT_MODE="local"
export SRTP_PRETRAINED_WEIGHT_PATH="weights/vit_small_patch16_224.pth"
python src/train.py
```

## 7. 不要混用 Base 和 Small 权重

错误示例：

```bash
export SRTP_BACKBONE_NAME="vit_base_patch16_224"
export SRTP_PRETRAINED_WEIGHT_PATH="weights/vit_small_patch16_224.pth"
```

这种配置会导致大量 key 或 tensor shape 不匹配，不能作为有效的 ViT-Base 初始化。
