import argparse
import math
import os
from pathlib import Path
from typing import List, Sequence, Tuple

import numpy as np
import torch
from PIL import Image, ImageDraw
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
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


def build_protocol_groups(max_points_per_group: int):
    return [
        ("Train Live", sample_group(Config.TRAIN_SOURCE_PATHS, "live", 0, max_points_per_group), "#1f77b4", "o"),
        ("Train Spoof", sample_group(Config.TRAIN_SOURCE_PATHS, "spoof", 1, max_points_per_group), "#d62728", "o"),
        ("Test Live", sample_group(Config.TEST_TARGET_PATH, "live", 0, max_points_per_group), "#17becf", "X"),
        ("Test Spoof", sample_group(Config.TEST_TARGET_PATH, "spoof", 1, max_points_per_group), "#ff9896", "X"),
    ]


def extract_features(model, image_infos: Sequence[Tuple[str, int]], device, batch_size: int) -> np.ndarray:
    if not image_infos:
        return np.empty((0, 768), dtype=np.float32)

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
    model.eval()
    with torch.no_grad():
        for images, _, _ in loader:
            images = images.to(device, non_blocking=True)
            tokens, _ = model(images, labels=None)
            features = torch.max(tokens, dim=1).values
            all_features.append(features.cpu().numpy())

    return np.concatenate(all_features, axis=0) if all_features else np.empty((0, 768), dtype=np.float32)


def fit_tsne(features: np.ndarray) -> np.ndarray:
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


def load_epoch_checkpoint(epoch: int, device):
    checkpoint_path = os.path.join(
        Config.CHECKPOINT_DIR,
        f"{Config.PROTOCOL_NAME}_epoch_{epoch}.pth",
    )
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model = TACL_ViT(
        scale=Config.SCALE,
        margin=Config.MARGIN,
        pretrained_weight_path="",
    ).to(device)
    state_dict = checkpoint["model_state_dict"] if "model_state_dict" in checkpoint else checkpoint
    model.load_state_dict(state_dict)
    return model, checkpoint_path


def render_epoch_tsne(
    epoch: int,
    model,
    groups,
    device,
    output_path: Path,
    batch_size: int,
) -> None:
    if plt is None:
        return

    feature_blocks = []
    group_assignments = []
    for group_name, image_infos, _, _ in groups:
        features = extract_features(model, image_infos, device, batch_size=batch_size)
        if features.shape[0] == 0:
            continue
        feature_blocks.append(features)
        group_assignments.extend([group_name] * features.shape[0])

    if not feature_blocks:
        return

    features = np.concatenate(feature_blocks, axis=0)
    group_assignments = np.asarray(group_assignments)
    embedding = fit_tsne(features)

    plt.figure(figsize=(9, 7))
    for group_name, _, color, marker in groups:
        mask = group_assignments == group_name
        if np.any(mask):
            plt.scatter(
                embedding[mask, 0],
                embedding[mask, 1],
                s=24,
                alpha=0.78,
                label=group_name,
                c=color,
                marker=marker,
                edgecolors="none",
            )
    plt.title(f"Epoch {epoch} t-SNE - {Config.PROTOCOL_NAME}")
    plt.legend()
    plt.grid(alpha=0.2)
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=220)
    plt.close()


def create_contact_sheet(image_paths: Sequence[Path], output_path: Path, cols: int = 5) -> None:
    valid_paths = [path for path in image_paths if path.exists()]
    if not valid_paths:
        return

    tiles = [Image.open(path).convert("RGB") for path in valid_paths]
    try:
        tile_width, tile_height = tiles[0].size
        rows = math.ceil(len(tiles) / cols)
        canvas = Image.new("RGB", (cols * tile_width, rows * tile_height), color=(255, 255, 255))

        for index, (tile, path) in enumerate(zip(tiles, valid_paths)):
            x = (index % cols) * tile_width
            y = (index // cols) * tile_height
            canvas.paste(tile, (x, y))
            draw = ImageDraw.Draw(canvas)
            draw.text((x + 12, y + 12), path.stem.replace("_", " "), fill=(0, 0, 0))

        output_path.parent.mkdir(parents=True, exist_ok=True)
        canvas.save(output_path)
    finally:
        for tile in tiles:
            tile.close()


def resolve_epochs(max_epoch: int) -> List[int]:
    epochs = []
    for epoch in range(1, max_epoch + 1):
        checkpoint_path = os.path.join(Config.CHECKPOINT_DIR, f"{Config.PROTOCOL_NAME}_epoch_{epoch}.pth")
        if os.path.exists(checkpoint_path):
            epochs.append(epoch)
    return epochs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-epoch", type=int, default=Config.EPOCHS, help="Maximum epoch to scan.")
    parser.add_argument("--max-points-per-group", type=int, default=250, help="Max sample count per group.")
    parser.add_argument("--batch-size", type=int, default=Config.BATCH_SIZE * 2, help="Feature extraction batch size.")
    args = parser.parse_args()

    if plt is None:
        raise RuntimeError("matplotlib is required to generate t-SNE figures.")

    output_dir = Path(Config.REPORT_DIR) / "epoch_tsne"
    contact_sheet_path = output_dir / "epoch_tsne_contact_sheet.png"
    groups = build_protocol_groups(args.max_points_per_group)
    epochs = resolve_epochs(args.max_epoch)
    if not epochs:
        raise FileNotFoundError(
            f"No epoch checkpoints found under {Config.CHECKPOINT_DIR}. "
            "Please enable SAVE_EVERY_EPOCH and finish training first."
        )

    device = torch.device(Config.DEVICE)
    generated_paths = []
    print(f"Generating epoch t-SNE series for {Config.PROTOCOL_NAME} ...")
    print(f"Epochs found: {epochs}")

    for epoch in epochs:
        model, checkpoint_path = load_epoch_checkpoint(epoch, device)
        output_path = output_dir / f"epoch_{epoch:02d}_tsne.png"
        print(f"[epoch {epoch}] checkpoint: {checkpoint_path}")
        render_epoch_tsne(
            epoch=epoch,
            model=model,
            groups=groups,
            device=device,
            output_path=output_path,
            batch_size=args.batch_size,
        )
        generated_paths.append(output_path)
        print(f"[epoch {epoch}] saved to: {output_path}")

    create_contact_sheet(generated_paths, contact_sheet_path)
    print(f"Contact sheet saved to: {contact_sheet_path}")


if __name__ == "__main__":
    main()
