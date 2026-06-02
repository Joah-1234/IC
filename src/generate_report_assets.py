import csv
import json
import os
from pathlib import Path
from typing import List, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics import confusion_matrix, roc_curve
from torch.utils.data import DataLoader, Dataset

from config import Config
from model_loss import TACL_ViT
from ocim_dataset import get_transforms

try:
    import matplotlib.pyplot as plt
except ImportError:
    plt = None


class PathImageDataset(Dataset):
    def __init__(self, image_infos: Sequence[Tuple[str, int]], transform):
        self.image_infos = list(image_infos)
        self.transform = transform

    def __len__(self) -> int:
        return len(self.image_infos)

    def __getitem__(self, index: int):
        image_path, label = self.image_infos[index]
        with Image.open(image_path) as image:
            pil_image = image.convert("RGB")
        tensor = self.transform(pil_image)
        return tensor, label, image_path


def load_json(path: str):
    with open(path, "r", encoding="utf-8") as file_obj:
        return json.load(file_obj)


def load_predictions(path: str):
    rows = []
    with open(path, "r", encoding="utf-8") as file_obj:
        reader = csv.DictReader(file_obj)
        for row in reader:
            row["label"] = int(row["label"])
            row["pred_label"] = int(row["pred_label"])
            row["spoof_score"] = float(row["spoof_score"])
            row["live_score"] = float(row["live_score"])
            rows.append(row)
    return rows


def plot_roc_curve(labels, scores, auc_value, save_path: str) -> None:
    if plt is None:
        return
    fpr, tpr, _ = roc_curve(labels, scores, pos_label=1)
    plt.figure(figsize=(6, 6))
    plt.plot(fpr, tpr, label=f"AUC = {auc_value:.4f}", linewidth=2)
    plt.plot([0, 1], [0, 1], linestyle="--", color="gray")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(f"ROC Curve - {Config.PROTOCOL_NAME}")
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=200)
    plt.close()


def plot_confusion_matrix(labels, preds, save_path: str) -> None:
    if plt is None:
        return
    matrix = confusion_matrix(labels, preds, labels=[0, 1])
    plt.figure(figsize=(6, 5))
    plt.imshow(matrix, cmap="Blues")
    plt.xticks([0, 1], ["Live", "Spoof"])
    plt.yticks([0, 1], ["Live", "Spoof"])
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title(f"Confusion Matrix - {Config.PROTOCOL_NAME}")

    for row in range(matrix.shape[0]):
        for col in range(matrix.shape[1]):
            plt.text(col, row, str(matrix[row, col]), ha="center", va="center", color="black")

    plt.tight_layout()
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=200)
    plt.close()


def _fit_tsne(features: np.ndarray) -> np.ndarray:
    if features.shape[0] == 0:
        return np.empty((0, 2), dtype=np.float32)

    feature_input = features
    if features.shape[1] > 50 and features.shape[0] > 50:
        pca_dim = min(50, features.shape[0] - 1, features.shape[1])
        feature_input = PCA(n_components=pca_dim, random_state=42).fit_transform(features)

    perplexity = min(30, max(5, feature_input.shape[0] // 12))
    return TSNE(
        n_components=2,
        init="pca",
        metric="cosine",
        random_state=42,
        perplexity=perplexity,
    ).fit_transform(feature_input)


def plot_target_tsne(features, labels, save_path: str, max_points: int = 1000) -> None:
    if plt is None or features.shape[0] == 0:
        return

    if features.shape[0] > max_points:
        indices = np.random.RandomState(42).choice(features.shape[0], size=max_points, replace=False)
        features = features[indices]
        labels = labels[indices]

    embedding = _fit_tsne(features)
    plt.figure(figsize=(8, 6))
    for label_value, label_name, color in [(0, "Target Live", "#1f77b4"), (1, "Target Spoof", "#d62728")]:
        mask = labels == label_value
        if np.any(mask):
            plt.scatter(embedding[mask, 0], embedding[mask, 1], s=14, alpha=0.72, label=label_name, c=color)
    plt.title(f"Target-only t-SNE - {Config.PROTOCOL_NAME}")
    plt.legend()
    plt.grid(alpha=0.2)
    plt.tight_layout()
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=200)
    plt.close()


def create_sample_grid(predictions, save_path: str, cell_size: int = 220) -> None:
    sorted_rows = sorted(predictions, key=lambda row: abs(row["spoof_score"] - 0.5))
    chosen = sorted_rows[:6]
    if not chosen:
        return

    canvas = Image.new("RGB", (cell_size * 3, cell_size * 2), color=(255, 255, 255))

    for index, row in enumerate(chosen):
        try:
            with Image.open(row["path"]) as image:
                panel = image.convert("RGB").resize((cell_size, cell_size - 40))
        except Exception:
            panel = Image.new("RGB", (cell_size, cell_size - 40), color=(230, 230, 230))

        label_name = "Live" if row["label"] == 0 else "Spoof"
        pred_name = "Live" if row["pred_label"] == 0 else "Spoof"
        caption = f"T:{label_name} P:{pred_name} S:{row['spoof_score']:.3f}"

        tile = Image.new("RGB", (cell_size, cell_size), color=(255, 255, 255))
        tile.paste(panel, (0, 0))
        tile_draw = ImageDraw.Draw(tile)
        tile_draw.text((8, cell_size - 32), caption, fill=(0, 0, 0))

        x = (index % 3) * cell_size
        y = (index // 3) * cell_size
        canvas.paste(tile, (x, y))

    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    canvas.save(save_path)


def resolve_model_path(metrics: dict) -> str:
    candidates = [
        metrics.get("checkpoint_path", ""),
        os.path.join(Config.CHECKPOINT_DIR, f"{Config.PROTOCOL_NAME}_best.pth"),
        os.path.join(Config.CHECKPOINT_DIR, f"{Config.PROTOCOL_NAME}_latest.pth"),
    ]
    for candidate in candidates:
        if candidate and os.path.exists(candidate):
            return candidate
    raise FileNotFoundError("No checkpoint found for feature visualization.")


def scan_label_dir(root_path: str, label_name: str, label_value: int) -> List[Tuple[str, int]]:
    image_infos: List[Tuple[str, int]] = []
    label_dir = Path(root_path) / label_name
    if not label_dir.exists():
        return image_infos

    for file_name in sorted(os.listdir(label_dir)):
        if file_name.lower().endswith((".jpg", ".png", ".jpeg", ".bmp")):
            image_infos.append((str(label_dir / file_name), label_value))
    return image_infos


def sample_group(root_paths, label_name: str, label_value: int, max_points: int) -> List[Tuple[str, int]]:
    if isinstance(root_paths, str):
        root_paths = [root_paths]

    samples: List[Tuple[str, int]] = []
    for root_path in root_paths:
        samples.extend(scan_label_dir(root_path, label_name, label_value))

    if len(samples) > max_points:
        indices = np.random.RandomState(42).choice(len(samples), size=max_points, replace=False)
        samples = [samples[index] for index in indices]
    return samples


def extract_features(model, image_infos: Sequence[Tuple[str, int]], device, batch_size: int) -> Tuple[np.ndarray, np.ndarray]:
    if not image_infos:
        feature_dim = int(getattr(model, "feature_dim", 0))
        return np.empty((0, feature_dim), dtype=np.float32), np.empty((0,), dtype=np.int64)

    _, test_transform = get_transforms(Config.IMG_SIZE)
    dataset = PathImageDataset(image_infos, test_transform)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=device.type == "cuda",
    )

    all_features = []
    all_labels = []
    model.eval()
    with torch.no_grad():
        for images, labels, _ in loader:
            images = images.to(device, non_blocking=True)
            tokens, _ = model(images, labels=None)
            features = torch.max(tokens, dim=1).values
            all_features.append(features.cpu().numpy())
            all_labels.append(labels.numpy())

    return np.concatenate(all_features, axis=0), np.concatenate(all_labels, axis=0)


def plot_protocol_tsne(metrics: dict, save_path: str, max_points_per_group: int = 250) -> None:
    if plt is None:
        return

    device = torch.device(Config.DEVICE)
    checkpoint = torch.load(resolve_model_path(metrics), map_location=device)
    checkpoint_config = checkpoint.get("config", {}) if isinstance(checkpoint, dict) else {}
    backbone_name = checkpoint_config.get("backbone_name", Config.BACKBONE_NAME)
    model = TACL_ViT(
        scale=Config.SCALE,
        margin=Config.MARGIN,
        pretrained_weight_path="",
        backbone_name=backbone_name,
        pretrained_init_mode="none",
    ).to(device)
    state_dict = checkpoint["model_state_dict"] if "model_state_dict" in checkpoint else checkpoint
    model.load_state_dict(state_dict)

    groups = [
        ("Train Live", Config.TRAIN_SOURCE_PATHS, "live", 0, "#1f77b4", "o"),
        ("Train Spoof", Config.TRAIN_SOURCE_PATHS, "spoof", 1, "#d62728", "o"),
        ("Test Live", Config.TEST_TARGET_PATH, "live", 0, "#17becf", "X"),
        ("Test Spoof", Config.TEST_TARGET_PATH, "spoof", 1, "#ff9896", "X"),
    ]

    feature_blocks = []
    group_assignments = []
    for group_name, root_paths, label_name, label_value, _, _ in groups:
        image_infos = sample_group(root_paths, label_name, label_value, max_points=max_points_per_group)
        features, labels = extract_features(model, image_infos, device, batch_size=Config.BATCH_SIZE * 2)
        if features.shape[0] == 0:
            continue
        feature_blocks.append(features)
        group_assignments.extend([group_name] * len(labels))

    if not feature_blocks:
        return

    features = np.concatenate(feature_blocks, axis=0)
    embedding = _fit_tsne(features)
    group_assignments = np.asarray(group_assignments)

    plt.figure(figsize=(9, 7))
    for group_name, _, _, _, color, marker in groups:
        mask = group_assignments == group_name
        if np.any(mask):
            plt.scatter(
                embedding[mask, 0],
                embedding[mask, 1],
                s=26,
                alpha=0.78,
                label=group_name,
                c=color,
                marker=marker,
                edgecolors="none",
            )
    plt.title(f"Protocol-style t-SNE - {Config.PROTOCOL_NAME}")
    plt.legend()
    plt.grid(alpha=0.2)
    plt.tight_layout()
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=220)
    plt.close()


def main():
    metrics = load_json(Config.TEST_METRICS_PATH)
    predictions = load_predictions(Config.TEST_PREDICTIONS_PATH)
    feature_pack = np.load(Config.TEST_FEATURES_PATH)
    labels = feature_pack["labels"]
    scores = feature_pack["scores"]
    features = feature_pack["features"]
    preds = np.array([row["pred_label"] for row in predictions], dtype=np.int64)

    tsne_target_only_path = str(Path(Config.REPORT_DIR) / "tsne_target_only.png")
    tsne_protocol_path = Config.TSNE_PATH

    plot_roc_curve(labels, scores, metrics["auc"], Config.ROC_CURVE_PATH)
    plot_confusion_matrix(labels, preds, Config.CONFUSION_MATRIX_PATH)
    plot_target_tsne(features, labels, tsne_target_only_path)
    plot_protocol_tsne(metrics, tsne_protocol_path)
    create_sample_grid(predictions, Config.SAMPLE_GRID_PATH)

    print(f"ROC curve saved to: {Config.ROC_CURVE_PATH}")
    print(f"Confusion matrix saved to: {Config.CONFUSION_MATRIX_PATH}")
    print(f"Target-only t-SNE saved to: {tsne_target_only_path}")
    print(f"Protocol-style t-SNE saved to: {Config.TSNE_PATH}")
    print(f"Sample grid saved to: {Config.SAMPLE_GRID_PATH}")


if __name__ == "__main__":
    main()
