import os
import random
from typing import List, Sequence, Tuple

import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms


class TwoCropTransform:
    def __init__(self, base_transform):
        self.base_transform = base_transform

    def __call__(self, image):
        return [self.base_transform(image), self.base_transform(image)]


class OCIMDataset(Dataset):
    def __init__(
        self,
        root_paths,
        transform=None,
        is_train: bool = True,
        max_spoof_ratio: float = None,
        retry_limit: int = 10,
        strict_mode: bool = True,
        return_path: bool = False,
    ):
        self.transform = transform
        self.is_train = is_train
        self.retry_limit = retry_limit
        self.strict_mode = strict_mode
        self.return_path = return_path
        self.image_infos: List[Tuple[str, int]] = []

        if isinstance(root_paths, str):
            root_paths = [root_paths]

        mode_name = "train" if is_train else "test"
        print(f"[{mode_name}] Initializing OCIM dataset...")

        live_list: List[Tuple[str, int]] = []
        spoof_list: List[Tuple[str, int]] = []

        for root_path in root_paths:
            if not os.path.exists(root_path):
                print(f"Warning: dataset path does not exist: {root_path}")
                continue

            live_list.extend(self._scan_label_dir(root_path, "live", 0))
            spoof_list.extend(self._scan_label_dir(root_path, "spoof", 1))

        print(f"  -> Raw samples: live={len(live_list)}, spoof={len(spoof_list)}")

        if self.is_train and max_spoof_ratio is not None and len(live_list) > 0:
            target_spoof_num = int(len(live_list) * max_spoof_ratio)
            if len(spoof_list) > target_spoof_num:
                print(f"  -> Applying spoof ratio cap: {len(spoof_list)} -> {target_spoof_num}")
                random.Random(42).shuffle(spoof_list)
                spoof_list = spoof_list[:target_spoof_num]

        self.image_infos = live_list + spoof_list
        if self.is_train:
            random.shuffle(self.image_infos)

        print(f"  -> Final samples: total={len(self.image_infos)}")
        if not self.image_infos:
            raise RuntimeError("No samples found. Please verify the processed dataset path.")

    @staticmethod
    def _scan_label_dir(root_path: str, label_name: str, label_value: int) -> List[Tuple[str, int]]:
        image_infos = []
        label_dir = os.path.join(root_path, label_name)
        if not os.path.exists(label_dir):
            return image_infos

        for file_name in sorted(os.listdir(label_dir)):
            if file_name.lower().endswith((".jpg", ".png", ".jpeg", ".bmp")):
                image_infos.append((os.path.join(label_dir, file_name), label_value))
        return image_infos

    def __len__(self) -> int:
        return len(self.image_infos)

    def __getitem__(self, idx: int):
        return self._safe_getitem(idx, retry_count=0, history=[])

    def _safe_getitem(self, idx: int, retry_count: int, history: Sequence[str]):
        if retry_count > self.retry_limit:
            visited = " | ".join(history[-3:])
            raise RuntimeError(
                f"Failed to load a valid sample after {self.retry_limit} retries. "
                f"Recent files: {visited}"
            )

        img_path, label = self.image_infos[idx]
        next_history = list(history) + [img_path]

        try:
            with Image.open(img_path) as image:
                pil_image = image.convert("RGB")
        except Exception as exc:
            return self._retry_or_raise(idx, retry_count, next_history, f"Image read failed: {exc}")

        try:
            image_output = self.transform(pil_image) if self.transform else pil_image
        except Exception as exc:
            return self._retry_or_raise(idx, retry_count, next_history, f"Transform failed: {exc}")

        if self._contains_nan(image_output):
            return self._retry_or_raise(idx, retry_count, next_history, "NaN detected after transform.")

        if self.return_path:
            return image_output, label, img_path
        return image_output, label

    def _retry_or_raise(self, idx: int, retry_count: int, history: Sequence[str], message: str):
        if self.strict_mode and retry_count >= self.retry_limit:
            raise RuntimeError(f"{message} Last file: {history[-1]}")

        new_idx = random.randint(0, len(self.image_infos) - 1)
        if retry_count < 3:
            print(f"Warning: {message} Retry with random sample.")
        return self._safe_getitem(new_idx, retry_count + 1, history)

    @staticmethod
    def _contains_nan(image_output) -> bool:
        if isinstance(image_output, list):
            return any(torch.isnan(view).any() for view in image_output if isinstance(view, torch.Tensor))
        if isinstance(image_output, torch.Tensor):
            return torch.isnan(image_output).any()
        return False


def get_transforms(img_size: int = 224):
    train_base = transforms.Compose(
        [
            transforms.RandomResizedCrop(img_size, scale=(0.8, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomApply(
                [transforms.ColorJitter(0.2, 0.2, 0.2, 0.05)],
                p=0.5,
            ),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ]
    )

    test_base = transforms.Compose(
        [
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ]
    )

    return TwoCropTransform(train_base), test_base
