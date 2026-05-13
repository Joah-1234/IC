import warnings
import os
from pathlib import Path

import numpy as np
import torch
from facenet_pytorch import MTCNN
from PIL import Image
from tqdm import tqdm
from config import Config

# 忽略PIL因为图片损坏可能发出的警告
warnings.filterwarnings("ignore")

RAW_ROOT = Config.RAW_ROOT
PROCESSED_ROOT = Config.MTCNN_OUTPUT_ROOT
DATASET_NAMES = Config.DATASET_NAMES
PREPROCESS_RULES = Config.PREPROCESS_RULES
IMG_SIZE = Config.IMG_SIZE
MARGIN = 0
DEVICE = Config.DEVICE

VALID_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp")


def get_rule(dataset_name: str) -> dict:
    return PREPROCESS_RULES.get(dataset_name, {})


def should_skip_directory(dataset_name: str, dir_path: str, src_dir: str) -> bool:
    rule = get_rule(dataset_name)
    skip_dir_keywords = set(rule.get("skip_dir_keywords", []))
    lowered_parts = {part.lower() for part in Path(dir_path).parts}
    if any(keyword in lowered_parts for keyword in skip_dir_keywords):
        return True

    allowed_top_level_dirs = rule.get("allowed_top_level_dirs")
    if not allowed_top_level_dirs:
        return False

    rel_dir = os.path.relpath(dir_path, src_dir)
    if rel_dir == ".":
        return False
    top_level = rel_dir.split(os.sep)[0]
    return top_level not in allowed_top_level_dirs


def should_skip_file(dataset_name: str, file_name: str) -> bool:
    rule = get_rule(dataset_name)
    skip_file_keywords = set(rule.get("skip_file_keywords", []))
    lower_name = file_name.lower()
    if not lower_name.endswith(VALID_EXTENSIONS):
        return True
    return any(keyword in lower_name for keyword in skip_file_keywords)


def detect_content_issue(image_np: np.ndarray, dataset_name: str):
    if image_np.ndim != 3 or image_np.shape[2] != 3:
        return "invalid_channels"

    img_float = image_np.astype(np.float32)
    gray = img_float.mean(axis=2)
    mean_val = float(gray.mean())
    std_val = float(gray.std())
    dynamic_range = float(np.percentile(gray, 95) - np.percentile(gray, 5))

    if mean_val < 8 and std_val < 8:
        return "near_black"
    if mean_val > 247 and std_val < 6:
        return "near_white"
    if dynamic_range < 10:
        return "low_dynamic_range"

    rule = get_rule(dataset_name)
    if rule.get("reject_near_grayscale", False):
        red = img_float[:, :, 0]
        green = img_float[:, :, 1]
        blue = img_float[:, :, 2]
        channel_gap = float(
            np.mean(np.abs(red - green)) +
            np.mean(np.abs(green - blue)) +
            np.mean(np.abs(red - blue))
        ) / 3.0
        max_channel_gap = float(
            max(
                np.max(np.abs(red - green)),
                np.max(np.abs(green - blue)),
                np.max(np.abs(red - blue)),
            )
        )
        if channel_gap < 1.5 and max_channel_gap < 8 and dynamic_range < 45:
            return "near_grayscale_depth_like"

    return None

def setup_mtcnn():
    print(f"正在初始化 MTCNN，使用设备: {DEVICE}...")
    # post_process=False 确保返回 uint8 图像以便保存为 jpg
    mtcnn = MTCNN(
        image_size=IMG_SIZE,
        margin=MARGIN,
        min_face_size=40,
        thresholds=[0.6, 0.7, 0.7],
        factor=0.709,
        post_process=False, 
        device=DEVICE,
        keep_all=False,
        select_largest=True
    )
    return mtcnn

def process_single_dataset(mtcnn, dataset_name):
    src_dir = os.path.join(RAW_ROOT, dataset_name)
    dst_dir = os.path.join(PROCESSED_ROOT, dataset_name)

    if not os.path.exists(src_dir):
        print(f"警告: 找不到源目录 {src_dir}，跳过该数据集。")
        return

    print(f"\n开始处理数据集: {dataset_name}")
    print(f"源目录: {src_dir}")
    print(f"目标目录: {dst_dir}")

    image_files = []
    skipped_roots = set()
    for root, dirs, files in os.walk(src_dir):
        dirs[:] = [
            dir_name
            for dir_name in dirs
            if not should_skip_directory(dataset_name, os.path.join(root, dir_name), src_dir)
        ]
        if should_skip_directory(dataset_name, root, src_dir):
            rel_root = os.path.relpath(root, src_dir)
            if rel_root not in skipped_roots:
                skipped_roots.add(rel_root)
            continue
        for file in files:
            if should_skip_file(dataset_name, file):
                continue
            image_files.append(os.path.join(root, file))
    
    print(f"找到 {len(image_files)} 张图片。")
    skipped_roots = sorted(path for path in skipped_roots if path not in (".", ""))
    if skipped_roots:
        preview = skipped_roots[:10]
        print(f"跳过的目录/路径样例 ({len(skipped_roots)}): {preview}")
    
    success_count = 0
    fail_count = 0
    skip_count = 0
    filtered_count = 0
    failure_log = os.path.join(PROCESSED_ROOT, f"{dataset_name}_failures.txt")
    os.makedirs(PROCESSED_ROOT, exist_ok=True)

    for img_path in tqdm(image_files, desc=f"Processing {dataset_name}"):
        # 计算相对路径，保持 live/spoof 结构
        rel_path = os.path.relpath(img_path, src_dir)
        save_path = os.path.join(dst_dir, rel_path)
        save_path_jpg = os.path.splitext(save_path)[0] + '.jpg'
        save_folder = os.path.dirname(save_path)

        if os.path.exists(save_path_jpg):
            skip_count += 1
            continue

        os.makedirs(save_folder, exist_ok=True)

        try:
            img = Image.open(img_path).convert('RGB')
            img_np = np.array(img)
            raw_issue = detect_content_issue(img_np, dataset_name)
            if raw_issue is not None:
                filtered_count += 1
                with open(failure_log, "a") as f:
                    f.write(f"FilteredRaw[{raw_issue}]: {img_path}\n")
                continue
            # MTCNN 返回裁剪后的 Tensor [3, H, W]
            img_cropped = mtcnn(img)

            if img_cropped is not None:
                # 转回 numpy 并保存
                img_np = img_cropped.permute(1, 2, 0).cpu().numpy().astype('uint8')
                crop_issue = detect_content_issue(img_np, dataset_name)
                if crop_issue is not None:
                    filtered_count += 1
                    with open(failure_log, "a") as f:
                        f.write(f"FilteredCrop[{crop_issue}]: {img_path}\n")
                    continue
                img_pil = Image.fromarray(img_np)
                # 统一保存为 jpg
                img_pil.save(save_path_jpg, quality=95)
                success_count += 1
            else:
                fail_count += 1
                with open(failure_log, "a") as f:
                    f.write(f"No face: {img_path}\n")
        except Exception as e:
            fail_count += 1
            with open(failure_log, "a") as f:
                f.write(f"Error: {img_path} | {e}\n")

    print(
        f"数据集 {dataset_name} 处理完成。"
        f" 成功: {success_count}, 过滤: {filtered_count}, 失败: {fail_count}, 跳过: {skip_count}"
    )

def main():
    if not os.path.exists(RAW_ROOT):
        print(f"错误: 原始数据根目录 {RAW_ROOT} 不存在！请先检查路径。")
        raise SystemExit(1)
        
    mtcnn = setup_mtcnn()
    
    # 循环处理每一个数据集
    for db_name in DATASET_NAMES:
        process_single_dataset(mtcnn, db_name)
        
    print("\n========================================")
    print("所有 OCIM 数据集预处理完毕！")
    print(f"处理后的数据位于: {PROCESSED_ROOT}")
    print(f"原始数据目录未被修改: {RAW_ROOT}")
    print(f"后续整理输出目录: {Config.PROCESSED_ROOT}")
    print("========================================")

if __name__ == '__main__':
    main()
