import os
from typing import Dict, Tuple

import timm
import torch
import torch.nn as nn
import torch.nn.functional as F
from safetensors.torch import load_file


def _unwrap_state_dict(state_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    if "state_dict" in state_dict and isinstance(state_dict["state_dict"], dict):
        state_dict = state_dict["state_dict"]
    if "model" in state_dict and isinstance(state_dict["model"], dict):
        state_dict = state_dict["model"]
    return state_dict


def _adapt_backbone_keys(state_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    cleaned_state_dict = {}
    removable_prefixes = (
        "backbone.",
        "module.backbone.",
        "module.",
        "model.",
        "encoder.",
    )

    for key, value in state_dict.items():
        new_key = key
        for prefix in removable_prefixes:
            if new_key.startswith(prefix):
                new_key = new_key[len(prefix):]
                break
        cleaned_state_dict[new_key] = value
    return cleaned_state_dict


def _summarize_load_result(
    model_state_dict: Dict[str, torch.Tensor],
    provided_state_dict: Dict[str, torch.Tensor],
    missing_keys,
    unexpected_keys,
) -> Dict[str, object]:
    missing_keys = sorted(missing_keys)
    unexpected_keys = sorted(unexpected_keys)
    model_keys = set(model_state_dict.keys())
    provided_keys = set(provided_state_dict.keys())
    matched_keys = sorted(model_keys.intersection(provided_keys) - set(missing_keys))

    return {
        "missing_keys": missing_keys,
        "unexpected_keys": unexpected_keys,
        "num_model_keys": len(model_keys),
        "num_provided_keys": len(provided_keys),
        "num_missing_keys": len(missing_keys),
        "num_unexpected_keys": len(unexpected_keys),
        "num_matched_keys": len(matched_keys),
        "matched_ratio": len(matched_keys) / max(1, len(model_keys)),
        "matched_key_examples": matched_keys[:10],
        "missing_key_examples": missing_keys[:10],
        "unexpected_key_examples": unexpected_keys[:10],
    }


class ArcFaceLayer(nn.Module):
    def __init__(self, in_features: int, out_features: int, s: float = 30.0, m: float = 0.50):
        super().__init__()
        self.s = s
        self.m = m
        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        nn.init.xavier_uniform_(self.weight)

    def forward(self, features: torch.Tensor, label: torch.Tensor = None) -> torch.Tensor:
        cosine = F.linear(F.normalize(features), F.normalize(self.weight))
        if label is None:
            return cosine * self.s

        theta = torch.acos(torch.clamp(cosine, -1.0 + 1e-7, 1.0 - 1e-7))
        phi = torch.cos(theta + self.m)
        one_hot = torch.zeros_like(cosine)
        one_hot.scatter_(1, label.view(-1, 1).long(), 1)
        output = (one_hot * phi) + ((1.0 - one_hot) * cosine)
        return output * self.s


class TACL_ViT(nn.Module):
    def __init__(
        self,
        scale: float = 30.0,
        margin: float = 0.5,
        pretrained_weight_path: str = "",
        backbone_name: str = "vit_base_patch16_224",
        pretrained_init_mode: str = "none",
    ):
        super().__init__()
        self.backbone_name = backbone_name
        self.pretrained_init_mode = (pretrained_init_mode or "none").strip().lower()
        if self.pretrained_init_mode not in {"timm", "local", "none"}:
            raise ValueError(
                f"Unsupported pretrained_init_mode '{pretrained_init_mode}'. "
                "Available options: timm, local, none."
            )
        self.backbone = self._create_backbone()
    ):
        super().__init__()
        self.backbone_name = backbone_name
        self.backbone = timm.create_model(
            backbone_name,
            pretrained=False,
            num_classes=0,
        )
        self.feature_dim = self._resolve_feature_dim()
        self.arc_head = ArcFaceLayer(in_features=self.feature_dim, out_features=2, s=scale, m=margin)
        self.pretrained_weight_path = pretrained_weight_path.strip()
        self.weight_load_report = self._build_weight_load_report()

    def _create_backbone(self) -> nn.Module:
        use_timm_pretrained = self.pretrained_init_mode == "timm"
        try:
            return timm.create_model(
                self.backbone_name,
                pretrained=use_timm_pretrained,
                num_classes=0,
            )
        except Exception as exc:
            if use_timm_pretrained:
                raise RuntimeError(
                    "Failed to initialize timm pretrained backbone. "
                    "Check network/cache availability, or set PRETRAINED_INIT_MODE='local' "
                    "with PRETRAINED_WEIGHT_PATH, or PRETRAINED_INIT_MODE='none'."
                ) from exc
            raise

    def _base_weight_report(self, loaded: bool, source: str, message: str, weight_path: str = "") -> Dict[str, object]:
        return {
            "weight_path": weight_path,
            "loaded": loaded,
            "source": source,
            "pretrained_init_mode": self.pretrained_init_mode,
            "backbone_name": self.backbone_name,
            "feature_dim": self.feature_dim,
            "message": message,
        }

    def _build_weight_load_report(self) -> Dict[str, object]:
        if self.pretrained_init_mode == "timm":
            return self._base_weight_report(
                loaded=True,
                source="timm",
                message="Backbone initialized from timm ImageNet pretrained weights.",
            )
        if self.pretrained_init_mode == "none":
            return self._base_weight_report(
                loaded=False,
                source="none",
                message="Pretrained initialization disabled. Backbone uses random initialization.",
            )
        return self._load_backbone_weights(self.pretrained_weight_path)

    def _resolve_feature_dim(self) -> int:
        feature_dim = getattr(self.backbone, "num_features", None)
        if feature_dim is None:
            feature_dim = getattr(self.backbone, "embed_dim", None)
        if feature_dim is None:
            raise AttributeError(
                f"Cannot infer feature dimension for backbone '{self.backbone_name}'. "
                "Please choose a timm ViT backbone exposing num_features or embed_dim."
            )
        return int(feature_dim)

    def _resolve_feature_dim(self) -> int:
        feature_dim = getattr(self.backbone, "num_features", None)
        if feature_dim is None:
            feature_dim = getattr(self.backbone, "embed_dim", None)
        if feature_dim is None:
            raise AttributeError(
                f"Cannot infer feature dimension for backbone '{self.backbone_name}'. "
                "Please choose a timm ViT backbone exposing num_features or embed_dim."
            )
        return int(feature_dim)

    def _load_backbone_weights(self, weight_path: str) -> Dict[str, object]:
        if not weight_path:
            return self._base_weight_report(
                loaded=False,
                source="local",
                message="PRETRAINED_INIT_MODE is local, but no pretrained weight path is configured.",
            )

        if not os.path.exists(weight_path):
            return self._base_weight_report(
                loaded=False,
                source="local",
                weight_path=weight_path,
                message="Configured pretrained weight file does not exist.",
            )

        if not os.path.exists(weight_path):
            return self._base_weight_report(
                loaded=False,
                source="local",
                weight_path=weight_path,
                message="Configured pretrained weight file does not exist.",
            )
            return {
                "weight_path": "",
                "loaded": False,
                "backbone_name": self.backbone_name,
                "feature_dim": self.feature_dim,
                "message": "No pretrained weight path configured. Backbone uses random initialization.",
            }

        if not os.path.exists(weight_path):
            return {
                "weight_path": weight_path,
                "loaded": False,
                "backbone_name": self.backbone_name,
                "feature_dim": self.feature_dim,
                "message": "Configured pretrained weight file does not exist.",
            }

        try:
            if weight_path.endswith(".safetensors"):
                raw_state_dict = load_file(weight_path)
            else:
                checkpoint = torch.load(weight_path, map_location="cpu")
                raw_state_dict = checkpoint if isinstance(checkpoint, dict) else {}

            state_dict = _adapt_backbone_keys(_unwrap_state_dict(raw_state_dict))
            load_result = self.backbone.load_state_dict(state_dict, strict=False)
            summary = _summarize_load_result(
                self.backbone.state_dict(),
                state_dict,
                load_result.missing_keys,
                load_result.unexpected_keys,
            )
            return {
                **self._base_weight_report(
                    loaded=True,
                    source="local",
                    weight_path=weight_path,
                    message="",
                ),
                "weight_path": weight_path,
                "loaded": True,
                "backbone_name": self.backbone_name,
                "feature_dim": self.feature_dim,
                **summary,
                "message": (
                    "Backbone weights loaded with strict=False. "
                    f"matched={summary['num_matched_keys']}/{summary['num_model_keys']} "
                    f"({summary['matched_ratio']:.1%}), "
                    f"missing={summary['num_missing_keys']}, "
                    f"unexpected={summary['num_unexpected_keys']}."
                ),
            }
        except Exception as exc:
            return self._base_weight_report(
                loaded=False,
                source="local",
                weight_path=weight_path,
                message=f"Failed to load pretrained weights: {exc}",
            )
            return {
                "weight_path": weight_path,
                "loaded": False,
                "backbone_name": self.backbone_name,
                "feature_dim": self.feature_dim,
                "message": f"Failed to load pretrained weights: {exc}",
            }

    def forward(self, x: torch.Tensor, labels: torch.Tensor = None) -> Tuple[torch.Tensor, torch.Tensor]:
        features = self.backbone.forward_features(x)
        tokens = F.normalize(features[:, 1:, :], p=2, dim=-1)
        global_feat = torch.max(tokens, dim=1).values
        arc_logits = self.arc_head(global_feat, labels)
        return tokens, arc_logits


class TACLoss(nn.Module):
    def __init__(self, temp: float = 0.07, max_live_tokens: int = 4096):
        super().__init__()
        self.temp = temp
        self.max_live_tokens = max_live_tokens

    def forward(self, tokens: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        device = tokens.device
        tokens = tokens.float()

        live_mask = labels == 0
        spoof_mask = labels == 1
        live_tokens = tokens[live_mask]
        spoof_tokens = tokens[spoof_mask]

        if live_tokens.shape[0] == 0:
            return torch.tensor(0.0, device=device, requires_grad=True)

        flat_live = live_tokens.reshape(-1, live_tokens.shape[-1])
        if flat_live.shape[0] > self.max_live_tokens:
            indices = torch.randperm(flat_live.shape[0], device=device)[: self.max_live_tokens]
            flat_live = flat_live[indices]

        spoof_repr = None
        if spoof_tokens.shape[0] > 0:
            spoof_repr = torch.max(spoof_tokens, dim=1).values
            spoof_repr = F.normalize(spoof_repr, p=2, dim=-1)

        logits = torch.matmul(flat_live, flat_live.T) / self.temp
        logits_max = torch.max(logits, dim=1, keepdim=True).values
        logits = logits - logits_max.detach()

        sample_count = flat_live.shape[0]
        diag_mask = torch.eye(sample_count, device=device, dtype=logits.dtype)
        exp_logits = torch.exp(logits)
        live_denominator = exp_logits.sum(dim=1, keepdim=True) - (exp_logits * diag_mask).sum(
            dim=1, keepdim=True
        )

        if spoof_repr is not None:
            spoof_logits = torch.matmul(flat_live, spoof_repr.T) / self.temp
            spoof_logits = spoof_logits - logits_max.detach()
            spoof_denominator = torch.exp(spoof_logits).sum(dim=1, keepdim=True)
            gamma = float(flat_live.shape[0]) / float(spoof_repr.shape[0])
            denominator = live_denominator + gamma * spoof_denominator
        else:
            denominator = live_denominator + 1e-6

        log_prob = logits - torch.log(denominator + 1e-6)
        positive_mask = 1.0 - diag_mask
        positive_sum = (positive_mask * log_prob).sum(dim=1)
        positive_count = torch.clamp(positive_mask.sum(dim=1), min=1.0)
        loss = -(positive_sum / positive_count).mean()
        return loss
