import json
import os
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Optional

import torch


PROJECT_ROOT = Path(__file__).resolve().parent.parent


def _env_optional_str(name: str, default: Optional[str] = None) -> Optional[str]:
    value = os.environ.get(name)
    if value is None:
        return default
    value = value.strip()
    if value.lower() in {"", "none", "null"}:
        return None
    return value


def _env_str(name: str, default: str) -> str:
    value = os.environ.get(name)
    if value is None or value.strip() == "":
        return default
    return value.strip()


# ============================================================
# User Config
# 你平时主要只需要修改这一段
# ============================================================

# 数据目录：原始数据只读，处理结果全部保存在当前 SRTP 项目目录下
RAW_DATA_ROOT = "/homeC/Public/FAS_dataset"
MTCNN_OUTPUT_ROOT = PROJECT_ROOT / "data" / "mtcnn_output"
PROCESSED_DATA_ROOT = PROJECT_ROOT / "data" / "processed"

# 预训练权重：
# 1. 默认使用 timm 官方 ImageNet 预训练初始化，避免 ViT 从随机权重开始训练。
# 2. 如果服务器不能联网，可提前下载对应 backbone 的权重并设置 PRETRAINED_INIT_MODE = "local"。
# 3. 可选模式："timm"（官方预训练）、"local"（本地权重）、"none"（随机初始化）。
PRETRAINED_WEIGHT_PATH = _env_optional_str("SRTP_PRETRAINED_WEIGHT_PATH", None)
PRETRAINED_INIT_MODE = _env_str(
    "SRTP_PRETRAINED_INIT_MODE",
    "local" if PRETRAINED_WEIGHT_PATH else "timm",
).lower()
MIN_PRETRAINED_MATCH_RATIO = float(_env_str("SRTP_MIN_PRETRAINED_MATCH_RATIO", "0.5"))

# 使用哪些数据集
DATASET_NAMES = [
    "oulu-npu",
    "CASIA-MFSD",
    "replayattack",
    "MSU-MFSD",
]

# 数据集定制预处理规则
PREPROCESS_RULES = {
    "oulu-npu": {
        "allowed_top_level_dirs": ["trainset_total", "testset", "devset"],
        "skip_dir_keywords": ["depth", "ir", "infrared", "profile"],
        "skip_file_keywords": ["depth", "ir", "infrared", "profile", "mask"],
        "reject_near_grayscale": True,
    },
    "CASIA-MFSD": {
        "allowed_top_level_dirs": ["train_img", "test_img"],
        "skip_dir_keywords": ["depth", "ir", "infrared", "profile"],
        "skip_file_keywords": ["depth", "ir", "infrared", "profile", "mask"],
        "reject_near_grayscale": True,
    },
    "replayattack": {
        "allowed_top_level_dirs": ["trainset", "devset", "testset"],
        "skip_dir_keywords": ["depth", "ir", "infrared"],
        "skip_file_keywords": ["depth", "ir", "infrared", "mask"],
        "reject_near_grayscale": False,
    },
    "MSU-MFSD": {
        "allowed_top_level_dirs": ["trainset", "testset"],
        "skip_dir_keywords": ["depth", "ir", "infrared"],
        "skip_file_keywords": ["depth", "ir", "infrared", "mask"],
        "reject_near_grayscale": False,
    },
}

# 当前实验协议
PROTOCOL_NAME = "ICM_to_O"

# 预处理抽样策略
# 可选:
# - "full": 使用全部帧
# - "sampled": 使用数据集提供的 14/32 帧采样索引
PREPROCESS_MODE = "sampled"
SAMPLED_FRAME_COUNT = 32

# 模型配置
# 方案一轻量化入口：可在 vit_base_patch16_224 / vit_small_patch16_224 /
# vit_tiny_patch16_224 之间切换。ArcFace 输入维度会由 timm 模型自动推断。
BACKBONE_NAME = _env_str("SRTP_BACKBONE_NAME", "vit_base_patch16_224")

# 训练超参数
IMG_SIZE = 224
BATCH_SIZE = 64
ACCUMULATION_STEPS = 2
LR = 1e-5
WEIGHT_DECAY = 5e-2
EPOCHS = 30
NUM_WORKERS = 4
ALPHA1 = 0.7
ALPHA2 = 0.3
TEMP = 0.2
MARGIN = 0.5
SCALE = 30.0
GRADIENT_CLIP_NORM = 1.0
MAX_LIVE_TOKENS = 4096
RETRY_LIMIT = 10
USE_AMP = False
SEED = 42
LOG_INTERVAL = 50
SAVE_EVERY_EPOCH = True
TRAIN_MAX_SPOOF_RATIO = 1.0

# 输出目录
RESULTS_ROOT = PROJECT_ROOT / "results"
RUNS_ROOT = PROJECT_ROOT / "runs"
CHECKPOINTS_ROOT = PROJECT_ROOT / "checkpoints"
REPORTS_ROOT = PROJECT_ROOT / "reports"


@dataclass(frozen=True)
class ProjectPaths:
    raw_root: str
    mtcnn_output_root: str
    processed_root: str
    pretrained_weight_path: str
    results_root: str
    runs_root: str
    checkpoints_root: str
    reports_root: str


@dataclass(frozen=True)
class ExperimentConfig:
    protocol_name: str
    backbone_name: str
    pretrained_init_mode: str
    min_pretrained_match_ratio: float
    preprocess_mode: str
    sampled_frame_count: int
    img_size: int
    batch_size: int
    accumulation_steps: int
    lr: float
    weight_decay: float
    epochs: int
    num_workers: int
    alpha1: float
    alpha2: float
    temp: float
    margin: float
    scale: float
    gradient_clip_norm: float
    max_live_tokens: int
    retry_limit: int
    use_amp: bool
    seed: int
    log_interval: int
    save_every_epoch: bool
    train_max_spoof_ratio: Optional[float]


class Config:
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    PATHS = ProjectPaths(
        raw_root=str(RAW_DATA_ROOT),
        mtcnn_output_root=str(MTCNN_OUTPUT_ROOT),
        processed_root=str(PROCESSED_DATA_ROOT),
        pretrained_weight_path="" if PRETRAINED_WEIGHT_PATH is None else str(PRETRAINED_WEIGHT_PATH),
        results_root=str(RESULTS_ROOT),
        runs_root=str(RUNS_ROOT),
        checkpoints_root=str(CHECKPOINTS_ROOT),
        reports_root=str(REPORTS_ROOT),
    )

    EXP = ExperimentConfig(
        protocol_name=PROTOCOL_NAME,
        backbone_name=BACKBONE_NAME,
        pretrained_init_mode=PRETRAINED_INIT_MODE,
        min_pretrained_match_ratio=MIN_PRETRAINED_MATCH_RATIO,
        preprocess_mode=PREPROCESS_MODE,
        sampled_frame_count=SAMPLED_FRAME_COUNT,
        img_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        accumulation_steps=ACCUMULATION_STEPS,
        lr=LR,
        weight_decay=WEIGHT_DECAY,
        epochs=EPOCHS,
        num_workers=NUM_WORKERS,
        alpha1=ALPHA1,
        alpha2=ALPHA2,
        temp=TEMP,
        margin=MARGIN,
        scale=SCALE,
        gradient_clip_norm=GRADIENT_CLIP_NORM,
        max_live_tokens=MAX_LIVE_TOKENS,
        retry_limit=RETRY_LIMIT,
        use_amp=USE_AMP,
        seed=SEED,
        log_interval=LOG_INTERVAL,
        save_every_epoch=SAVE_EVERY_EPOCH,
        train_max_spoof_ratio=TRAIN_MAX_SPOOF_RATIO,
    )

    DATASET_NAMES = DATASET_NAMES
    PREPROCESS_RULES = PREPROCESS_RULES
    RAW_ROOT = PATHS.raw_root
    MTCNN_OUTPUT_ROOT = PATHS.mtcnn_output_root
    PROCESSED_ROOT = PATHS.processed_root
    PRETRAINED_WEIGHT_PATH = PATHS.pretrained_weight_path

    if PRETRAINED_INIT_MODE not in {"timm", "local", "none"}:
        raise ValueError(
            f"Unsupported PRETRAINED_INIT_MODE '{PRETRAINED_INIT_MODE}'. "
            "Available options: timm, local, none."
        )

    PATH_O = str(Path(PROCESSED_ROOT) / "oulu-npu")
    PATH_C = str(Path(PROCESSED_ROOT) / "CASIA-MFSD")
    PATH_I = str(Path(PROCESSED_ROOT) / "replayattack")
    PATH_M = str(Path(PROCESSED_ROOT) / "MSU-MFSD")

    PROTOCOLS = {
        "OCI_to_M": {
            "train_source_paths": [PATH_O, PATH_C, PATH_I],
            "test_target_path": PATH_M,
        },
        "OMI_to_C": {
            "train_source_paths": [PATH_O, PATH_M, PATH_I],
            "test_target_path": PATH_C,
        },
        "OCM_to_I": {
            "train_source_paths": [PATH_O, PATH_C, PATH_M],
            "test_target_path": PATH_I,
        },
        "ICM_to_O": {
            "train_source_paths": [PATH_I, PATH_C, PATH_M],
            "test_target_path": PATH_O,
        },
    }

    if EXP.protocol_name not in PROTOCOLS:
        raise ValueError(
            f"Unsupported protocol '{EXP.protocol_name}'. "
            f"Available options: {', '.join(PROTOCOLS)}"
        )

    PROTOCOL_NAME = EXP.protocol_name
    BACKBONE_NAME = EXP.backbone_name
    PRETRAINED_INIT_MODE = EXP.pretrained_init_mode
    MIN_PRETRAINED_MATCH_RATIO = EXP.min_pretrained_match_ratio
    PREPROCESS_MODE = EXP.preprocess_mode
    SAMPLED_FRAME_COUNT = EXP.sampled_frame_count
    TRAIN_SOURCE_PATHS = PROTOCOLS[PROTOCOL_NAME]["train_source_paths"]
    TEST_TARGET_PATH = PROTOCOLS[PROTOCOL_NAME]["test_target_path"]

    IMG_SIZE = EXP.img_size
    BATCH_SIZE = EXP.batch_size
    ACCUMULATION_STEPS = EXP.accumulation_steps
    LR = EXP.lr
    WEIGHT_DECAY = EXP.weight_decay
    EPOCHS = EXP.epochs
    NUM_WORKERS = EXP.num_workers
    ALPHA1 = EXP.alpha1
    ALPHA2 = EXP.alpha2
    TEMP = EXP.temp
    MARGIN = EXP.margin
    SCALE = EXP.scale
    GRADIENT_CLIP_NORM = EXP.gradient_clip_norm
    MAX_LIVE_TOKENS = EXP.max_live_tokens
    RETRY_LIMIT = EXP.retry_limit
    USE_AMP = EXP.use_amp
    SEED = EXP.seed
    LOG_INTERVAL = EXP.log_interval
    SAVE_EVERY_EPOCH = EXP.save_every_epoch
    TRAIN_MAX_SPOOF_RATIO = EXP.train_max_spoof_ratio

    LOG_DIR = str(Path(PATHS.runs_root) / f"tacl_cross_domain_{PROTOCOL_NAME}")
    CHECKPOINT_DIR = str(Path(PATHS.checkpoints_root) / PROTOCOL_NAME)
    RESULT_DIR = str(Path(PATHS.results_root) / PROTOCOL_NAME)
    REPORT_DIR = str(Path(PATHS.reports_root) / PROTOCOL_NAME)
    TRAIN_METRICS_PATH = str(Path(RESULT_DIR) / "train_metrics.json")
    TEST_METRICS_PATH = str(Path(RESULT_DIR) / "test_metrics.json")
    TEST_PREDICTIONS_PATH = str(Path(RESULT_DIR) / "test_predictions.csv")
    TEST_FEATURES_PATH = str(Path(RESULT_DIR) / "test_features.npz")
    TRAIN_CURVE_PATH = str(Path(REPORT_DIR) / "training_curves.png")
    ROC_CURVE_PATH = str(Path(REPORT_DIR) / "roc_curve.png")
    CONFUSION_MATRIX_PATH = str(Path(REPORT_DIR) / "confusion_matrix.png")
    TSNE_PATH = str(Path(REPORT_DIR) / "tsne.png")
    SAMPLE_GRID_PATH = str(Path(REPORT_DIR) / "sample_predictions.png")
    DATASET_SUMMARY_JSON_PATH = str(Path(PATHS.results_root) / "dataset_summary.json")
    DATASET_SUMMARY_CSV_PATH = str(Path(PATHS.results_root) / "dataset_summary.csv")

    @classmethod
    def ensure_runtime_dirs(cls) -> None:
        Path(cls.LOG_DIR).mkdir(parents=True, exist_ok=True)
        Path(cls.CHECKPOINT_DIR).mkdir(parents=True, exist_ok=True)
        Path(cls.RESULT_DIR).mkdir(parents=True, exist_ok=True)
        Path(cls.REPORT_DIR).mkdir(parents=True, exist_ok=True)
        Path(cls.RAW_ROOT).mkdir(parents=True, exist_ok=True)
        Path(cls.MTCNN_OUTPUT_ROOT).mkdir(parents=True, exist_ok=True)
        Path(cls.PROCESSED_ROOT).mkdir(parents=True, exist_ok=True)

    @classmethod
    def snapshot(cls) -> dict:
        snapshot = asdict(cls.EXP)
        snapshot.update(
            {
                "device": cls.DEVICE,
                "project_root": str(PROJECT_ROOT),
                "raw_root": cls.RAW_ROOT,
                "mtcnn_output_root": cls.MTCNN_OUTPUT_ROOT,
                "processed_root": cls.PROCESSED_ROOT,
                "pretrained_weight_path": cls.PRETRAINED_WEIGHT_PATH,
                "train_source_paths": cls.TRAIN_SOURCE_PATHS,
                "test_target_path": cls.TEST_TARGET_PATH,
                "protocol_name": cls.PROTOCOL_NAME,
                "dataset_names": cls.DATASET_NAMES,
                "generated_at": datetime.now().isoformat(timespec="seconds"),
            }
        )
        return snapshot

    @classmethod
    def dump_snapshot(cls, save_path: str) -> None:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        with open(save_path, "w", encoding="utf-8") as file_obj:
            json.dump(cls.snapshot(), file_obj, indent=2, ensure_ascii=False)
