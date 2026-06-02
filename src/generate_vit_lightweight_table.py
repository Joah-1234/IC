import csv
from pathlib import Path


LIGHTWEIGHT_BACKBONE_ROWS = [
    {
        "model": "ViT-Base/16",
        "timm_name": "vit_base_patch16_224",
        "embed_dim": 768,
        "depth": 12,
        "heads": 12,
        "patch_tokens_224": 196,
        "params_m": 85.8,
        "params_reduction": "baseline",
        "arcface_params": 1536,
        "suggested_stage": "原始基线",
        "risk": "精度稳定但训练/推理开销最大",
    },
    {
        "model": "ViT-Small/16",
        "timm_name": "vit_small_patch16_224",
        "embed_dim": 384,
        "depth": 12,
        "heads": 6,
        "patch_tokens_224": 196,
        "params_m": 21.7,
        "params_reduction": "约减少 75%",
        "arcface_params": 768,
        "suggested_stage": "首选轻量化实验",
        "risk": "通常速度/显存收益明显，精度下降相对可控",
    },
    {
        "model": "ViT-Tiny/16",
        "timm_name": "vit_tiny_patch16_224",
        "embed_dim": 192,
        "depth": 12,
        "heads": 3,
        "patch_tokens_224": 196,
        "params_m": 5.7,
        "params_reduction": "约减少 93%",
        "arcface_params": 384,
        "suggested_stage": "极限压缩实验",
        "risk": "速度最快但跨域泛化精度下降风险最高",
    },
]


PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUTPUT_DIR = PROJECT_ROOT / "reports" / "paper_assets"
CSV_PATH = OUTPUT_DIR / "vit_lightweight_backbone_table.csv"
TEX_PATH = OUTPUT_DIR / "vit_lightweight_backbone_table.tex"
MD_PATH = OUTPUT_DIR / "vit_lightweight_backbone_table.md"


def ensure_output_dir() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def write_csv(rows) -> None:
    fieldnames = list(rows[0].keys())
    with open(CSV_PATH, "w", newline="", encoding="utf-8") as file_obj:
        writer = csv.DictWriter(file_obj, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_markdown(rows) -> None:
    headers = [
        "模型",
        "timm 名称",
        "特征维度",
        "层数",
        "注意力头",
        "224输入Token数",
        "参数量(M)",
        "参数下降",
        "建议用途",
    ]
    lines = ["| " + " | ".join(headers) + " |", "|" + "---|" * len(headers)]
    for row in rows:
        lines.append(
            "| "
            + " | ".join(
                [
                    row["model"],
                    f"`{row['timm_name']}`",
                    str(row["embed_dim"]),
                    str(row["depth"]),
                    str(row["heads"]),
                    str(row["patch_tokens_224"]),
                    f"{row['params_m']:.1f}",
                    row["params_reduction"],
                    row["suggested_stage"],
                ]
            )
            + " |"
        )
    lines.append("")
    lines.append("注：参数量为 ViT backbone 常用近似值，用于方案筛选；最终论文表格建议以实际运行统计脚本输出为准。")
    MD_PATH.write_text("\n".join(lines), encoding="utf-8")


def escape_latex(value) -> str:
    return str(value).replace("_", "\\_").replace("%", "\\%")


def write_latex(rows) -> None:
    lines = [
        r"\begin{tabular}{llrrrrll}",
        r"\toprule",
        r"模型 & timm名称 & 维度 & 层数 & 头数 & Token数 & 参数量(M) & 定位 \\",
        r"\midrule",
    ]
    for row in rows:
        lines.append(
            f"{escape_latex(row['model'])} & "
            f"{escape_latex(row['timm_name'])} & "
            f"{row['embed_dim']} & {row['depth']} & {row['heads']} & "
            f"{row['patch_tokens_224']} & {row['params_m']:.1f} & "
            f"{escape_latex(row['suggested_stage'])} \\\\"
        )
    lines.extend([r"\bottomrule", r"\end{tabular}"])
    TEX_PATH.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    ensure_output_dir()
    write_csv(LIGHTWEIGHT_BACKBONE_ROWS)
    write_markdown(LIGHTWEIGHT_BACKBONE_ROWS)
    write_latex(LIGHTWEIGHT_BACKBONE_ROWS)
    print(f"Saved CSV: {CSV_PATH}")
    print(f"Saved TeX: {TEX_PATH}")
    print(f"Saved Markdown: {MD_PATH}")


if __name__ == "__main__":
    main()
