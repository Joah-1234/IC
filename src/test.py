import argparse
import csv
import json
import os
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import auc, roc_curve
from torch.utils.data import DataLoader
from tqdm import tqdm

from config import Config
from model_loss import TACL_ViT
from ocim_dataset import OCIMDataset, get_transforms


def save_json(data: dict, save_path: str) -> None:
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    with open(save_path, "w", encoding="utf-8") as file_obj:
            json.dump(data, file_obj, indent=2, ensure_ascii=False)


def save_predictions(rows, save_path: str) -> None:
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    with open(save_path, "w", encoding="utf-8", newline="") as file_obj:
        writer = csv.DictWriter(
            file_obj,
            fieldnames=["path", "label", "pred_label", "spoof_score", "live_score"],
        )
        writer.writeheader()
        writer.writerows(rows)


def calculate_metrics(y_true: np.ndarray, y_scores: np.ndarray):
    fpr, tpr, thresholds = roc_curve(y_true, y_scores, pos_label=1)
    roc_auc = auc(fpr, tpr)
    fnr = 1 - tpr
    eer_index = np.nanargmin(np.absolute(fpr - fnr))
    eer = float(fpr[eer_index])
    eer_threshold = float(thresholds[eer_index])
    hter = float((fpr[eer_index] + fnr[eer_index]) / 2.0)
    return roc_auc, eer, hter, eer_threshold


def resolve_model_path(cli_path: str = "") -> str:
    candidates = []
    if cli_path:
        candidates.append(cli_path)
    candidates.extend(
        [
            os.path.join(Config.CHECKPOINT_DIR, f"{Config.PROTOCOL_NAME}_best.pth"),
            os.path.join(Config.CHECKPOINT_DIR, f"{Config.PROTOCOL_NAME}_latest.pth"),
        ]
    )

    for candidate in candidates:
        if candidate and os.path.exists(candidate):
            return candidate
    raise FileNotFoundError(
        "No available checkpoint found. Please train the model first or pass --model-path."
    )


def evaluate(model, dataloader, device):
    model.eval()
    all_scores = []
    all_labels = []
    all_preds = []
    all_paths = []
    all_live_scores = []
    all_features = []

    print(f"Evaluating target domain: {os.path.basename(Config.TEST_TARGET_PATH)}")
    with torch.no_grad():
        for images, labels, paths in tqdm(dataloader, desc="Testing"):
            images = images.to(device, non_blocking=True)
            tokens, arc_logits = model(images, labels=None)
            probs = F.softmax(arc_logits, dim=1)
            spoof_scores = probs[:, 1]
            live_scores = probs[:, 0]
            preds = torch.argmax(probs, dim=1)
            features = torch.max(tokens, dim=1).values
            all_scores.extend(spoof_scores.cpu().numpy())
            all_live_scores.extend(live_scores.cpu().numpy())
            all_labels.extend(labels.numpy())
            all_preds.extend(preds.cpu().numpy())
            all_paths.extend(paths)
            all_features.append(features.cpu().numpy())

    all_scores = np.asarray(all_scores)
    all_live_scores = np.asarray(all_live_scores)
    all_labels = np.asarray(all_labels)
    all_preds = np.asarray(all_preds)
    all_features = np.concatenate(all_features, axis=0) if all_features else np.empty((0, 768))

    if len(np.unique(all_labels)) < 2:
        raise RuntimeError("Test set must contain both live and spoof samples.")

    roc_auc, eer, hter, threshold = calculate_metrics(all_labels, all_scores)
    predictions = []
    for index, path in enumerate(all_paths):
        predictions.append(
            {
                "path": path,
                "label": int(all_labels[index]),
                "pred_label": int(all_preds[index]),
                "spoof_score": float(all_scores[index]),
                "live_score": float(all_live_scores[index]),
            }
        )

    return {
        "protocol": Config.PROTOCOL_NAME,
        "target_domain": os.path.basename(Config.TEST_TARGET_PATH),
        "num_samples": int(len(all_labels)),
        "num_live": int(np.sum(all_labels == 0)),
        "num_spoof": int(np.sum(all_labels == 1)),
        "auc": float(roc_auc),
        "eer": eer,
        "hter": hter,
        "threshold": threshold,
        "metric_note": (
            "HTER is computed at the EER threshold on the target test set. "
            "If the paper uses a development-set threshold, report this as an approximation."
        ),
        "labels": all_labels,
        "scores": all_scores,
        "predictions": predictions,
        "features": all_features,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", default="", help="Path to the checkpoint file.")
    args = parser.parse_args()

    Config.ensure_runtime_dirs()
    device = torch.device(Config.DEVICE)
    print(f"Test device: {device}")

    _, test_transform = get_transforms(Config.IMG_SIZE)
    test_dataset = OCIMDataset(
        root_paths=Config.TEST_TARGET_PATH,
        transform=test_transform,
        is_train=False,
        retry_limit=Config.RETRY_LIMIT,
        strict_mode=True,
        return_path=True,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=Config.BATCH_SIZE * 2,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=device.type == "cuda",
    )

    model_path = resolve_model_path(args.model_path)
    print(f"Loading checkpoint: {model_path}")

    checkpoint = torch.load(model_path, map_location=device)
    checkpoint_config = checkpoint.get("config", {}) if isinstance(checkpoint, dict) else {}
    backbone_name = checkpoint_config.get("backbone_name", Config.BACKBONE_NAME)
    print(f"Backbone      : {backbone_name}")
    model = TACL_ViT(
        scale=Config.SCALE,
        margin=Config.MARGIN,
        pretrained_weight_path="",
        backbone_name=backbone_name,
    ).to(device)

    state_dict = checkpoint["model_state_dict"] if "model_state_dict" in checkpoint else checkpoint
    model.load_state_dict(state_dict)
    metrics = evaluate(model, test_loader, device)
    metrics["checkpoint_path"] = model_path
    labels = metrics.pop("labels")
    scores = metrics.pop("scores")
    predictions = metrics.pop("predictions")
    features = metrics.pop("features")

    print("\n" + "=" * 60)
    print(f"Protocol      : {metrics['protocol']}")
    print(f"Target domain : {metrics['target_domain']}")
    print(f"Samples       : total={metrics['num_samples']}, live={metrics['num_live']}, spoof={metrics['num_spoof']}")
    print(f"AUC           : {metrics['auc']:.4f}")
    print(f"EER           : {metrics['eer']:.4f}")
    print(f"HTER          : {metrics['hter']:.4f}")
    print(f"Threshold     : {metrics['threshold']:.4f}")
    print("=" * 60 + "\n")

    save_json(metrics, Config.TEST_METRICS_PATH)
    save_predictions(predictions, Config.TEST_PREDICTIONS_PATH)
    np.savez(
        Config.TEST_FEATURES_PATH,
        features=features,
        labels=labels,
        scores=scores,
    )
    print(f"Metrics saved to: {Config.TEST_METRICS_PATH}")
    print(f"Predictions saved to: {Config.TEST_PREDICTIONS_PATH}")
    print(f"Features saved to: {Config.TEST_FEATURES_PATH}")


if __name__ == "__main__":
    main()
