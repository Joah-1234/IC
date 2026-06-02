import argparse
from pathlib import Path

import timm
import torch


DEFAULT_BACKBONE = "vit_base_patch16_224"
DEFAULT_OUTPUT = Path("weights") / f"{DEFAULT_BACKBONE}.pth"


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Download a timm ImageNet-pretrained backbone and save its state_dict "
            "as a local .pth file that matches BACKBONE_NAME."
        )
    )
    parser.add_argument(
        "--backbone",
        default=DEFAULT_BACKBONE,
        help="timm backbone name. Must match SRTP_BACKBONE_NAME/BACKBONE_NAME during training.",
    )
    parser.add_argument(
        "--output",
        default=str(DEFAULT_OUTPUT),
        help="Output .pth path for the backbone state_dict.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    model = timm.create_model(args.backbone, pretrained=True, num_classes=0)
    checkpoint = {
        "backbone_name": args.backbone,
        "state_dict": model.state_dict(),
    }
    torch.save(checkpoint, output_path)

    param_count = sum(parameter.numel() for parameter in model.parameters())
    print(f"Backbone     : {args.backbone}")
    print(f"Parameters   : {param_count / 1_000_000:.2f}M")
    print(f"Saved weights: {output_path}")
    print("Use with:")
    print(f"  export SRTP_BACKBONE_NAME=\"{args.backbone}\"")
    print("  export SRTP_PRETRAINED_INIT_MODE=\"local\"")
    print(f"  export SRTP_PRETRAINED_WEIGHT_PATH=\"{output_path}\"")
    print("  python src/train.py")


if __name__ == "__main__":
    main()
