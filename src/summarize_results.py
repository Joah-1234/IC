import json
import os
from pathlib import Path

try:
    import matplotlib.pyplot as plt
except ImportError:
    plt = None

PROTOCOLS = ["OCI_to_M", "OMI_to_C", "OCM_to_I", "ICM_to_O"]
RESULT_ROOT = Path(os.environ.get("SRTP_RESULTS_ROOT", "./results"))


def load_metrics(protocol):
    metrics_path = RESULT_ROOT / protocol / "test_metrics.json"
    if not metrics_path.exists():
        return None
    with open(metrics_path, "r", encoding="utf-8") as file_obj:
        return json.load(file_obj)


def main():
    rows = []
    auc_values = []
    hter_values = []
    for protocol in PROTOCOLS:
        metrics = load_metrics(protocol)
        if metrics is None:
            rows.append((protocol, "N/A", "N/A", "N/A"))
            continue
        auc_values.append((protocol, metrics["auc"]))
        hter_values.append((protocol, metrics["hter"]))
        rows.append(
            (
                protocol,
                f"{metrics['auc']:.4f}",
                f"{metrics['eer']:.4f}",
                f"{metrics['hter']:.4f}",
            )
        )

    output_path = RESULT_ROOT / "summary.csv"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as file_obj:
        file_obj.write("protocol,auc,eer,hter\n")
        for row in rows:
            file_obj.write(",".join(row) + "\n")

    print("protocol,auc,eer,hter")
    for row in rows:
        print(",".join(row))
    print(f"Saved summary to: {output_path}")

    if plt is not None and auc_values:
        protocols = [item[0] for item in auc_values]
        auc_nums = [item[1] for item in auc_values]
        hter_nums = [dict(hter_values)[protocol] for protocol in protocols]
        plt.figure(figsize=(10, 5))
        plt.subplot(1, 2, 1)
        plt.bar(protocols, auc_nums, color="#1f77b4")
        plt.title("AUC by Protocol")
        plt.ylabel("AUC")
        plt.xticks(rotation=20)
        plt.subplot(1, 2, 2)
        plt.bar(protocols, hter_nums, color="#d62728")
        plt.title("HTER by Protocol")
        plt.ylabel("HTER")
        plt.xticks(rotation=20)
        plt.tight_layout()
        figure_path = RESULT_ROOT / "summary_overview.png"
        plt.savefig(figure_path, dpi=200)
        plt.close()
        print(f"Saved figure to: {figure_path}")


if __name__ == "__main__":
    main()
