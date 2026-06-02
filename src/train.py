import json
import os
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.cuda.amp import GradScaler, autocast
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
from tqdm import tqdm

try:
    from torch.utils.tensorboard import SummaryWriter
except ImportError:
    SummaryWriter = None

from config import Config
from model_loss import TACLoss, TACL_ViT
from ocim_dataset import OCIMDataset, get_transforms

try:
    import matplotlib.pyplot as plt
except ImportError:
    plt = None


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def save_json(data: dict, save_path: str) -> None:
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    with open(save_path, "w", encoding="utf-8") as file_obj:
            json.dump(data, file_obj, indent=2, ensure_ascii=False)


def save_training_curve(history, save_path: str) -> None:
    if plt is None or not history:
        return

    epochs = [record["epoch"] for record in history]
    total_loss = [record["avg_total_loss"] for record in history]
    tac_loss = [record["avg_tac_loss"] for record in history]
    am_loss = [record["avg_am_loss"] for record in history]

    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, total_loss, label="Total Loss", linewidth=2)
    plt.plot(epochs, tac_loss, label="TAC Loss", linewidth=2)
    plt.plot(epochs, am_loss, label="AM Loss", linewidth=2)
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title(f"Training Curves - {Config.PROTOCOL_NAME}")
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.close()


def main():
    Config.ensure_runtime_dirs()
    set_seed(Config.SEED)

    writer = SummaryWriter(log_dir=Config.LOG_DIR) if SummaryWriter is not None else None
    Config.dump_snapshot(os.path.join(Config.RESULT_DIR, "experiment_config.json"))

    device = torch.device(Config.DEVICE)
    use_amp = Config.USE_AMP and device.type == "cuda"

    print("=" * 60)
    print(f"Protocol      : {Config.PROTOCOL_NAME}")
    print(f"Train domains : {[os.path.basename(path) for path in Config.TRAIN_SOURCE_PATHS]}")
    print(f"Target domain : {os.path.basename(Config.TEST_TARGET_PATH)}")
    print(f"Backbone      : {Config.BACKBONE_NAME}")
    print(f"Pretrain mode : {Config.PRETRAINED_INIT_MODE}")
    print(f"Device        : {device}")
    print(f"AMP enabled   : {use_amp}")
    print(f"Spoof ratio cap: {Config.TRAIN_MAX_SPOOF_RATIO}")
    print("=" * 60)

    train_aug, _ = get_transforms(Config.IMG_SIZE)
    train_dataset = OCIMDataset(
        root_paths=Config.TRAIN_SOURCE_PATHS,
        transform=train_aug,
        is_train=True,
        max_spoof_ratio=Config.TRAIN_MAX_SPOOF_RATIO,
        retry_limit=Config.RETRY_LIMIT,
        strict_mode=True,
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=True,
        num_workers=Config.NUM_WORKERS,
        drop_last=True,
        pin_memory=device.type == "cuda",
    )

    model = TACL_ViT(
        scale=Config.SCALE,
        margin=Config.MARGIN,
        pretrained_weight_path=Config.PRETRAINED_WEIGHT_PATH,
        backbone_name=Config.BACKBONE_NAME,
        pretrained_init_mode=Config.PRETRAINED_INIT_MODE,
    ).to(device)
    report = model.weight_load_report
    print(report["message"])
    print(f"  -> Backbone: {report.get('backbone_name', Config.BACKBONE_NAME)}")
    print(f"  -> Feature dim: {report.get('feature_dim', 'unknown')}")
    print(f"  -> Pretrain source: {report.get('source', 'unknown')}")
    if report.get("weight_path"):
    ).to(device)
    report = model.weight_load_report
    print(report["message"])
    print(f"  -> Backbone: {report.get('backbone_name', Config.BACKBONE_NAME)}")
    print(f"  -> Feature dim: {report.get('feature_dim', 'unknown')}")
    print(f"  -> Pretrain source: {report.get('source', 'unknown')}")
    if report.get("weight_path"):
    ).to(device)
    report = model.weight_load_report
    print(report["message"])
    if report.get("loaded"):
        print(f"  -> Backbone: {report.get('backbone_name', Config.BACKBONE_NAME)}")
        print(f"  -> Feature dim: {report.get('feature_dim', 'unknown')}")
        print(f"  -> Weight path: {report.get('weight_path', '')}")
    if report.get("num_model_keys") is not None:
        print(
            "  -> Load summary: "
            f"matched={report.get('num_matched_keys', 0)}/"
            f"{report.get('num_model_keys', 0)}, "
            f"provided={report.get('num_provided_keys', 0)}, "
            f"missing={report.get('num_missing_keys', 0)}, "
            f"unexpected={report.get('num_unexpected_keys', 0)}"
        )
        if report.get("missing_key_examples"):
            print(f"  -> Missing key examples: {report['missing_key_examples'][:5]}")
        if report.get("unexpected_key_examples"):
            print(f"  -> Unexpected key examples: {report['unexpected_key_examples'][:5]}")

    if Config.PRETRAINED_INIT_MODE != "none" and not report.get("loaded"):
        raise RuntimeError(
            "Pretrained initialization was requested but no pretrained weights were loaded. "
            "Use PRETRAINED_INIT_MODE='timm' with network/cache, provide a valid local "
            "PRETRAINED_WEIGHT_PATH, or explicitly set PRETRAINED_INIT_MODE='none'."
        )
    matched_ratio = report.get("matched_ratio")
    if (
        Config.PRETRAINED_INIT_MODE == "local"
        and matched_ratio is not None
        and matched_ratio < Config.MIN_PRETRAINED_MATCH_RATIO
    ):
        raise RuntimeError(
            f"Local pretrained weight match ratio is too low: {matched_ratio:.1%}. "
            f"Expected at least {Config.MIN_PRETRAINED_MATCH_RATIO:.1%}. "
            "Please check that PRETRAINED_WEIGHT_PATH matches BACKBONE_NAME."
        )

    loss_tac_fn = TACLoss(temp=Config.TEMP, max_live_tokens=Config.MAX_LIVE_TOKENS).to(device)
    loss_am_fn = nn.CrossEntropyLoss().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=Config.LR, weight_decay=Config.WEIGHT_DECAY)
    scheduler = CosineAnnealingLR(optimizer, T_max=Config.EPOCHS, eta_min=1e-6)
    scaler = GradScaler(enabled=use_amp)

    optimizer.zero_grad(set_to_none=True)
    global_step = 0
    best_loss = float("inf")
    history = []

    for epoch in range(Config.EPOCHS):
        model.train()
        running_loss = 0.0
        running_tac = 0.0
        running_am = 0.0

        current_lr = optimizer.param_groups[0]["lr"]
        progress = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{Config.EPOCHS} [LR={current_lr:.2e}]")

        for step, (images, labels) in enumerate(progress, start=1):
            imgs = torch.cat(images, dim=0).to(device, non_blocking=True)
            targets = torch.cat([labels, labels], dim=0).to(device, non_blocking=True)

            with autocast(enabled=use_amp):
                tokens, arc_logits = model(imgs, targets)
                loss_tac = loss_tac_fn(tokens, targets)
                loss_am = loss_am_fn(arc_logits, targets)
                total_loss = Config.ALPHA1 * loss_tac + Config.ALPHA2 * loss_am
                scaled_loss = total_loss / Config.ACCUMULATION_STEPS

            if not torch.isfinite(total_loss):
                raise RuntimeError(
                    f"Non-finite loss detected at epoch={epoch + 1}, step={step}: {total_loss.item()}"
                )

            scaler.scale(scaled_loss).backward()

            if step % Config.ACCUMULATION_STEPS == 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=Config.GRADIENT_CLIP_NORM)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)

                if writer is not None and global_step % Config.LOG_INTERVAL == 0:
                    writer.add_scalar("train/total_loss", total_loss.item(), global_step)
                    writer.add_scalar("train/tac_loss", loss_tac.item(), global_step)
                    writer.add_scalar("train/am_loss", loss_am.item(), global_step)
                    writer.add_scalar("train/lr", optimizer.param_groups[0]["lr"], global_step)
                global_step += 1

            running_loss += total_loss.item()
            running_tac += loss_tac.item()
            running_am += loss_am.item()
            progress.set_postfix(
                loss=f"{total_loss.item():.4f}",
                tac=f"{loss_tac.item():.4f}",
                am=f"{loss_am.item():.4f}",
            )

        scheduler.step()

        epoch_record = {
            "epoch": epoch + 1,
            "avg_total_loss": running_loss / len(train_loader),
            "avg_tac_loss": running_tac / len(train_loader),
            "avg_am_loss": running_am / len(train_loader),
            "lr": optimizer.param_groups[0]["lr"],
        }
        history.append(epoch_record)

        print(
            f"Epoch {epoch + 1} finished: "
            f"loss={epoch_record['avg_total_loss']:.4f}, "
            f"tac={epoch_record['avg_tac_loss']:.4f}, "
            f"am={epoch_record['avg_am_loss']:.4f}, "
            f"next_lr={epoch_record['lr']:.2e}"
        )

        checkpoint = {
            "epoch": epoch + 1,
            "protocol": Config.PROTOCOL_NAME,
            "config": Config.snapshot(),
            "weight_load_report": model.weight_load_report,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "loss": epoch_record["avg_total_loss"],
        }

        if Config.SAVE_EVERY_EPOCH:
            epoch_path = os.path.join(
                Config.CHECKPOINT_DIR,
                f"{Config.PROTOCOL_NAME}_epoch_{epoch + 1}.pth",
            )
            torch.save(checkpoint, epoch_path)

        latest_path = os.path.join(Config.CHECKPOINT_DIR, f"{Config.PROTOCOL_NAME}_latest.pth")
        torch.save(checkpoint, latest_path)

        if epoch_record["avg_total_loss"] < best_loss:
            best_loss = epoch_record["avg_total_loss"]
            best_path = os.path.join(Config.CHECKPOINT_DIR, f"{Config.PROTOCOL_NAME}_best.pth")
            torch.save(checkpoint, best_path)

        save_json(
            {
                "protocol": Config.PROTOCOL_NAME,
                "weight_load_report": model.weight_load_report,
                "history": history,
            },
            Config.TRAIN_METRICS_PATH,
        )
        save_training_curve(history, Config.TRAIN_CURVE_PATH)

    if writer is not None:
        writer.close()
    print("Training finished.")


if __name__ == "__main__":
    main()
