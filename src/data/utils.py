"""
Data utilities for SpatialRGPT-style preprocessing.

Mirrors the behaviour of llava/data/utils.py in SpatialRGPT:
- process_depth: load 3-channel pseudo-RGB depth PNG
- process_masks: decode RLE / polygon masks referenced by <mask> tokens
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image


# ─────────────────────────────────────────────────────────────────────────
# Depth loading
# ─────────────────────────────────────────────────────────────────────────

def process_depth(filename: str, depth_root: str | Path) -> Image.Image:
    """
    Load a precomputed 3-channel pseudo-RGB depth PNG (uint8 0-255).

    The depth PNG is generated offline by `scripts/preprocess_depth.py`
    using DINOv3 Depther + min-max normalisation:
        depth_norm = (d - d.min()) / (d.max() - d.min()) * 255
        depth_3ch  = stack([depth_norm] * 3, axis=-1).astype(uint8)
    """
    depth_path = Path(depth_root) / f"{filename}.png"
    img = Image.open(depth_path).convert("RGB")  # ensure 3 channels
    return img


# ─────────────────────────────────────────────────────────────────────────
# Mask decoding
# ─────────────────────────────────────────────────────────────────────────

def decode_rle(rle: dict, height: int, width: int) -> np.ndarray:
    """
    Decode COCO-style RLE to a binary HxW uint8 mask.
    Uses pycocotools if available, else a pure-numpy fallback.
    """
    try:
        from pycocotools import mask as mask_utils
        if isinstance(rle.get("counts"), list):
            rle = mask_utils.frPyObjects(rle, height, width)
        m = mask_utils.decode(rle)
        if m.ndim == 3:
            m = m[..., 0]
        return m.astype(np.uint8)
    except ImportError:
        # Fallback: assume already-decoded list of integers
        counts = rle["counts"]
        flat = np.zeros(height * width, dtype=np.uint8)
        idx = 0
        val = 0
        for c in counts:
            flat[idx:idx + c] = val
            idx += c
            val = 1 - val
        return flat.reshape((height, width), order="F")


def process_masks(
    sources: list[dict],
    image_h: int,
    image_w: int,
    max_regions: int = 32,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Extract binary masks from a sample's `masks` / `seg` field.

    Expected sample-level field (added by SpatialRGPT data pipeline):
        sample["rle"]: list of RLE dicts, one per <mask> token in conversations
      OR
        sample["bboxes"]: list of [x1,y1,x2,y2] (used as fallback to build box masks)

    Returns:
        masks:       [R_max, H, W] uint8 tensor (zero-padded if R < R_max)
        mask_valid:  [R_max] bool tensor
    """
    masks_np = np.zeros((max_regions, image_h, image_w), dtype=np.uint8)
    valid = np.zeros(max_regions, dtype=bool)

    rles = sources.get("rle") or sources.get("masks") or []
    bboxes = sources.get("bboxes") or sources.get("boxes") or []

    if rles:
        for i, rle in enumerate(rles[:max_regions]):
            masks_np[i] = decode_rle(rle, image_h, image_w)
            valid[i] = True
    elif bboxes:
        for i, box in enumerate(bboxes[:max_regions]):
            x1, y1, x2, y2 = [int(round(v)) for v in box]
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(image_w, x2), min(image_h, y2)
            masks_np[i, y1:y2, x1:x2] = 1
            valid[i] = True

    return torch.from_numpy(masks_np), torch.from_numpy(valid)


# ─────────────────────────────────────────────────────────────────────────
# Conversation token preprocessing
# ─────────────────────────────────────────────────────────────────────────

DEFAULT_IMAGE_TOKEN = "<image>"
DEFAULT_DEPTH_TOKEN = "<depth>"
DEFAULT_MASK_TOKEN  = "<mask>"


def preprocess_multimodal_text(
    conversations: list[dict],
    inject_depth_token: bool = True,
) -> str:
    """
    Convert SpatialRGPT-style conversation list to a flat prompt string.
    Each turn has {"from": "human"/"gpt", "value": "..."}.

    If `inject_depth_token` is True and the first user turn contains <image>
    but no <depth>, append <depth> right after <image>.
    """
    parts: list[str] = []
    injected = False

    for turn in conversations:
        role = turn.get("from", "human")
        text = turn["value"]

        if not injected and inject_depth_token and DEFAULT_IMAGE_TOKEN in text \
                and DEFAULT_DEPTH_TOKEN not in text:
            text = text.replace(
                DEFAULT_IMAGE_TOKEN,
                f"{DEFAULT_IMAGE_TOKEN}\n{DEFAULT_DEPTH_TOKEN}",
                1,
            )
            injected = True

        speaker = "User" if role in ("human", "user") else "Assistant"
        parts.append(f"{speaker}: {text}")

    return "\n".join(parts)


def count_mask_tokens(conversations: list[dict]) -> int:
    return sum(turn["value"].count(DEFAULT_MASK_TOKEN) for turn in conversations)
