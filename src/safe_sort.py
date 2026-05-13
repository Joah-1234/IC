import os
import shutil
import csv
import json
from config import Config

SOURCE_ROOT = Config.MTCNN_OUTPUT_ROOT
TARGET_ROOT = Config.PROCESSED_ROOT
DATASET_NAMES = Config.DATASET_NAMES

def get_unique_filename(source_root, file_path):
    """生成唯一文件名，保留路径信息防止覆盖"""
    rel_path = os.path.relpath(file_path, source_root)
    base, ext = os.path.splitext(rel_path)
    new_name = base.replace(os.sep, '_') + ext
    return new_name

def safe_copy(src_path, dataset_name, label):
    dst_dir = os.path.join(TARGET_ROOT, dataset_name, label)
    os.makedirs(dst_dir, exist_ok=True)
    
    dataset_source_root = os.path.join(SOURCE_ROOT, dataset_name)
    new_filename = get_unique_filename(dataset_source_root, src_path)
    dst_path = os.path.join(dst_dir, new_filename)
    
    try:
        shutil.copy2(src_path, dst_path)
        return True
    except Exception as e:
        print(f"Copy Error: {src_path} -> {e}")
        return False


def save_dataset_summary(summary_rows):
    os.makedirs(os.path.dirname(Config.DATASET_SUMMARY_JSON_PATH), exist_ok=True)

    with open(Config.DATASET_SUMMARY_JSON_PATH, "w", encoding="utf-8") as file_obj:
        json.dump(summary_rows, file_obj, indent=2, ensure_ascii=False)

    with open(Config.DATASET_SUMMARY_CSV_PATH, "w", encoding="utf-8", newline="") as file_obj:
        writer = csv.DictWriter(
            file_obj,
            fieldnames=["dataset", "live", "spoof", "unknown", "total"],
        )
        writer.writeheader()
        writer.writerows(summary_rows)

def determine_label_generic(path_parts, filename_lower):
    """通用关键词判断逻辑"""
    # 强特征：攻击
    attack_keywords = ['attack', 'print', 'video', 'ipad', 'iphone', 'highdef', 'mobile', 'warped', 'cut']
    if any(k in path_parts for k in ['spoof', 'attack']) or \
       any(k in filename_lower for k in attack_keywords):
        return 'spoof'

    # 强特征：真脸 (包含 ReplayAttack 的 authenticate)
    live_keywords = ['real', 'live', 'authenticate', 'enroll', 'client'] 
    
    # 检查路径中的关键词 (精确匹配)
    if any(k in path_parts for k in ['live', 'real', 'true']):
        return 'live'
    
    # 检查文件名中的关键词 (模糊匹配)
    if 'real' in filename_lower or 'authenticate' in filename_lower:
        return 'live'

    return None


def infer_casia_label_from_text(text: str):
    normalized = text.lower().replace("-", "_")
    parts = [part for part in normalized.split("_") if part]

    if normalized.isdigit():
        value = int(normalized)
        if value in (1, 2):
            return "live"
        if 3 <= value <= 8:
            return "spoof"

    for index, part in enumerate(parts):
        if part in {"nm", "hr"} and index + 1 < len(parts) and parts[index + 1].isdigit():
            value = int(parts[index + 1])
            # 当前使用的数据版本中，只有 HR_1 被确认是真脸，
            # HR_2/HR_3/HR_4 均按假脸处理。
            if part == "hr" and value == 1:
                return "live"
            if part == "hr" and value >= 2:
                return "spoof"
            if part == "nm" and value in (1, 2):
                return "live"
            if part == "nm" and 3 <= value <= 8:
                return "spoof"

    return None


def infer_casia_label(rel_path_from_subset, files):
    parts = [part for part in rel_path_from_subset.split(os.sep) if part not in ("", ".")]

    # CASIA-MFSD 结构:
    # train_img/<subject_id>/<sample_type>/
    # 其中 <subject_id> 是受试者编号，不是类别标签
    # 真正决定真假的是最后一级目录，如 1..8 / HR_1..HR_4
    if len(parts) >= 2:
        label = infer_casia_label_from_text(parts[-1])
        if label is not None:
            return label

    for file_name in files:
        if not file_name.lower().endswith(".jpg"):
            continue
        stem = os.path.splitext(file_name)[0]
        label = infer_casia_label_from_text(stem)
        if label is not None:
            return label

    return None

def process_oulu():
    name = 'oulu-npu'
    src_root = os.path.join(SOURCE_ROOT, name)
    if not os.path.exists(src_root): return
    print(f"\n正在处理 {name} ...")
    
    # 指定只处理这些文件夹，忽略 depth
    target_subdirs = ['trainset_total', 'testset', 'devset']
    
    count = {'live': 0, 'spoof': 0, 'unknown': 0}
    
    for subdir in target_subdirs:
        current_dir = os.path.join(src_root, subdir)
        if not os.path.exists(current_dir): continue
        
        for root, dirs, files in os.walk(current_dir):
            # 再次保险：跳过 depth 文件夹
            if 'depth' in root.lower(): continue
            
            for f in files:
                if not f.lower().endswith(('.jpg', '.png')): continue
                src_path = os.path.join(root, f)
                
                try:
                    # OULU 规则应当来自父目录名，而不是帧文件名。
                    # 目录结构示例: devset/1_1_21_4/001.jpg
                    # 这里最后一位 "4" 才是 attack type，001.jpg 只是帧编号。
                    folder_name = os.path.basename(root)
                    parts = folder_name.split('_')
                    if parts[-1].isdigit():
                        type_code = int(parts[-1])
                        label = 'live' if type_code == 1 else 'spoof'
                        safe_copy(src_path, name, label)
                        count[label] += 1
                    else:
                        raise ValueError("Format error")
                except:
                    safe_copy(src_path, name, 'unknown')
                    count['unknown'] += 1
                    
    print(f"  -> {name} 结果: {count}")

def process_casia():
    name = 'CASIA-MFSD'
    src_root = os.path.join(SOURCE_ROOT, name)
    if not os.path.exists(src_root): return
    print(f"\n正在处理 {name} ...")

    # 指定只处理 img 文件夹，忽略 depth
    target_subdirs = ['train_img', 'test_img']
    
    count = {'live': 0, 'spoof': 0, 'unknown': 0}

    for subdir in target_subdirs:
        current_dir = os.path.join(src_root, subdir)
        if not os.path.exists(current_dir): continue

        for root, dirs, files in os.walk(current_dir):
            if 'depth' in root.lower(): continue
            
            # 这里必须相对于 train_img/test_img 判断。
            # current_dir 下第一层是 subject_id，第二层才是 live/spoof 类型目录。
            rel_path = os.path.relpath(root, current_dir)
            label = infer_casia_label(rel_path, files)
            
            if label:
                for f in files:
                    if f.lower().endswith('.jpg'):
                        safe_copy(os.path.join(root, f), name, label)
                        count[label] += 1
            else:
                # 确实无法分类 (Unknown)
                has_img = any(f.lower().endswith('.jpg') for f in files)
                if has_img:
                    for f in files:
                        if f.lower().endswith('.jpg'):
                            safe_copy(os.path.join(root, f), name, 'unknown')
                            count['unknown'] += 1

    print(f"  -> {name} 结果: {count}")

def process_replay_msu(name):
    src_root = os.path.join(SOURCE_ROOT, name)
    if not os.path.exists(src_root): return
    print(f"\n正在处理 {name} ...")
    
    # 只要不是 depth 文件夹就行
    # Replay: trainset, testset, devset
    # MSU: trainset, testset
    
    count = {'live': 0, 'spoof': 0, 'unknown': 0}

    for root, dirs, files in os.walk(src_root):
        # 1. 跳过 Depth 文件夹
        if 'depth' in root.lower(): continue
        
        rel_path = os.path.relpath(root, src_root)
        path_parts = [p.lower() for p in rel_path.split(os.sep)]
        
        # 2. 调用通用判断逻辑
        label = None
        
        # 优先看文件夹结构
        if 'attack' in path_parts:
            label = 'spoof'
        elif 'real' in path_parts:
            label = 'live'
        
        # 3. 如果文件夹没看出来，对每个文件单独判断
        if label:
            for f in files:
                if f.lower().endswith('.jpg'):
                    safe_copy(os.path.join(root, f), name, label)
                    count[label] += 1
        else:
            # 逐个文件判断
            for f in files:
                if not f.lower().endswith('.jpg'): continue
                
                file_label = determine_label_generic(path_parts, f.lower())
                
                if file_label:
                    safe_copy(os.path.join(root, f), name, file_label)
                    count[file_label] += 1
                else:
                    safe_copy(os.path.join(root, f), name, 'unknown')
                    count['unknown'] += 1
                    if count['unknown'] <= 3:
                        print(f"    [Warning] Uncategorized: {os.path.join(rel_path, f)}")

    print(f"  -> {name} 结果: {count}")

def verify_dataset():
    print("\n====== 最终自检 (V4) ======")
    summary_rows = []
    for ds in DATASET_NAMES:
        path = os.path.join(TARGET_ROOT, ds)
        if not os.path.exists(path): continue
        
        live = len(os.listdir(os.path.join(path, 'live'))) if os.path.exists(os.path.join(path, 'live')) else 0
        spoof = len(os.listdir(os.path.join(path, 'spoof'))) if os.path.exists(os.path.join(path, 'spoof')) else 0
        unknown = len(os.listdir(os.path.join(path, 'unknown'))) if os.path.exists(os.path.join(path, 'unknown')) else 0
        total = live + spoof + unknown
        
        print(f"数据集: {ds}")
        print(f"  Live : {live}")
        print(f"  Spoof: {spoof}")
        print(f"  Unknown: {unknown}")
        print(f"  Total: {total}")

        summary_rows.append(
            {
                "dataset": ds,
                "live": live,
                "spoof": spoof,
                "unknown": unknown,
                "total": total,
            }
        )

    if summary_rows:
        save_dataset_summary(summary_rows)
        print(f"\n统计已保存到: {Config.DATASET_SUMMARY_JSON_PATH}")
        print(f"统计已保存到: {Config.DATASET_SUMMARY_CSV_PATH}")

def main():
    if not os.path.exists(SOURCE_ROOT):
        print(f"错误: 源目录 {SOURCE_ROOT} 不存在！")
        return
        
    process_oulu()
    process_casia()
    process_replay_msu('replayattack')
    process_replay_msu('MSU-MFSD')
    
    verify_dataset()
    print(f"\n处理完成！原始数据未修改。")
    print(f"MTCNN 中间结果位于: {SOURCE_ROOT}")
    print(f"新数据集位于: {TARGET_ROOT}")

if __name__ == '__main__':
    main()
