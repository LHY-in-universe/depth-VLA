"""
Utility functions for LoRA management.
"""

from __future__ import annotations

import torch
import torch.nn as nn


def count_trainable_params(model: nn.Module) -> tuple[int, int]:
    """Returns (trainable_params, total_params)."""
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    return trainable, total


def print_trainable_params(model: nn.Module, label: str = "Model"):
    trainable, total = count_trainable_params(model)
    pct = 100 * trainable / max(total, 1)
    print(f"[{label}] Trainable params: {trainable:,} / {total:,} ({pct:.2f}%)")


def get_lora_state_dict(model: nn.Module) -> dict[str, torch.Tensor]:
    """Extract only LoRA weights (for checkpoint saving)."""
    return {k: v for k, v in model.state_dict().items() if "lora_" in k}


def get_non_lora_trainable_state_dict(model: nn.Module) -> dict[str, torch.Tensor]:
    """Extract trainable non-LoRA weights (fusion + region_pooler)."""
    return {
        k: v
        for k, v in model.state_dict().items()
        if "lora_" not in k
        and any(
            k.startswith(prefix)
            for prefix in ("fusion.", "region_pooler.")
        )
    }


def save_adapter_checkpoint(model: nn.Module, path: str):
    """Save only the trainable adapter weights (LoRA + fusion + proj + depth_head)."""
    state = {}
    state.update(get_lora_state_dict(model))
    state.update(get_non_lora_trainable_state_dict(model))
    torch.save(state, path)
    print(f"Saved adapter checkpoint ({len(state)} tensors) → {path}")


def load_adapter_checkpoint(model: nn.Module, path: str, strict: bool = False):
    state = torch.load(path, map_location="cpu")
    missing, unexpected = model.load_state_dict(state, strict=strict)
    if missing:
        print(f"[load_adapter] Missing keys: {missing}")
    if unexpected:
        print(f"[load_adapter] Unexpected keys: {unexpected}")
