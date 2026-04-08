"""
SpatialRGPT-format dataset for depth-VLA.

JSON schema (mirrors a8cheng/OpenSpatialDataset, file: result_10_depth_convs.json):
[
  {
    "filename": "0a1b2c3d4e",                    # no extension
    "conversations": [
      {"from": "human", "value": "<image>\nWhat is the distance between <mask> and <mask>?"},
      {"from": "gpt",   "value": "Region 1 is roughly 1.2m closer to the camera than region 2."}
    ],
    "rle":    [ {"size": [H, W], "counts": "..."}, ... ],   # one per <mask>, COCO RLE
    "bboxes": [ [x1, y1, x2, y2], ... ]                     # optional fallback
  },
  ...
]

Files on disk:
    {image_root}/{filename}.jpg            — original RGB
    {depth_root}/{filename}.png            — 3-channel pseudo-RGB depth (uint8)
                                             precomputed by scripts/preprocess_depth.py
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch
from PIL import Image
from torch.utils.data import Dataset
from transformers import AutoProcessor

from .utils import (
    preprocess_multimodal_text,
    process_depth,
    process_masks,
)


class SpatialRGPTDataset(Dataset):
    def __init__(
        self,
        json_path: str,
        image_root: str,
        depth_root: str,
        qwen_processor: AutoProcessor,
        max_length: int = 2048,
        max_regions: int = 32,
        subset_size: int | None = None,
    ):
        self.image_root = Path(image_root)
        self.depth_root = Path(depth_root)
        self.qwen_processor = qwen_processor
        self.max_length = max_length
        self.max_regions = max_regions

        print(f"Loading dataset JSON from {json_path}...")
        with open(json_path) as f:
            self.samples = json.load(f)

        if subset_size is not None and subset_size < len(self.samples):
            self.samples = self.samples[:subset_size]
        print(f"Loaded {len(self.samples)} samples.")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        sample = self.samples[idx]
        filename = sample["filename"]

        # ── Load RGB ──────────────────────────────────────────────────────
        image_path = self.image_root / f"{filename}.jpg"
        rgb_image = Image.open(image_path).convert("RGB")
        W, H = rgb_image.size

        # ── Load 3-channel depth (precomputed offline by DINOv3 Depther) ──
        depth_image = process_depth(filename, self.depth_root)

        # ── Decode masks for <mask> tokens in conversation ────────────────
        masks, mask_valid = process_masks(
            sample, image_h=H, image_w=W, max_regions=self.max_regions
        )

        # ── Build prompt text (no <depth> token — depth fed as 2nd image) ─
        text = preprocess_multimodal_text(
            sample["conversations"], inject_depth_token=False
        )

        # ── Tokenise text + RGB through Qwen processor ────────────────────
        # We DO NOT pass the depth image to the processor — we process it
        # separately so that input_ids has only one <image> placeholder.
        # The depth tensor goes through the visual encoder in a second pass.
        rgb_inputs = self.qwen_processor(
            text=text,
            images=rgb_image,
            return_tensors="pt",
            max_length=self.max_length,
            truncation=True,
            padding="max_length",
        )

        # Process depth through Qwen's image processor only (no text)
        depth_inputs = self.qwen_processor.image_processor(
            images=depth_image, return_tensors="pt"
        )

        # ── Build labels (mask non-assistant tokens) ──────────────────────
        input_ids = rgb_inputs["input_ids"].squeeze(0)
        labels = self._build_labels(input_ids, sample["conversations"])

        out = {
            "input_ids":          input_ids,
            "attention_mask":     rgb_inputs["attention_mask"].squeeze(0),
            "rgb_pixel_values":   rgb_inputs["pixel_values"].squeeze(0),
            "depth_pixel_values": depth_inputs["pixel_values"].squeeze(0),
            "masks":              masks,
            "mask_valid":         mask_valid,
            "labels":             labels,
        }

        # Pass through Qwen extras (e.g. image_grid_thw)
        for k, v in rgb_inputs.items():
            if k not in ("input_ids", "attention_mask", "pixel_values") \
                    and isinstance(v, torch.Tensor):
                out[k] = v.squeeze(0)

        return out

    # ──────────────────────────────────────────────────────────────────────
    # Label masking
    # ──────────────────────────────────────────────────────────────────────

    def _build_labels(self, input_ids: torch.Tensor, conversations: list[dict]) -> torch.Tensor:
        """Mask everything except assistant ('gpt') response spans with -100."""
        labels = input_ids.clone()
        labels[:] = -100

        tokenizer = self.qwen_processor.tokenizer
        for turn in conversations:
            if turn["from"] not in ("gpt", "assistant"):
                continue
            assistant_ids = tokenizer(
                turn["value"], add_special_tokens=False
            )["input_ids"]
            if not assistant_ids:
                continue
            span = self._find_subseq(input_ids.tolist(), assistant_ids)
            if span is not None:
                start, end = span
                labels[start:end] = input_ids[start:end]

        return labels

    @staticmethod
    def _find_subseq(seq: list[int], sub: list[int]) -> tuple[int, int] | None:
        n, m = len(seq), len(sub)
        for i in range(n - m + 1):
            if seq[i:i + m] == sub:
                return i, i + m
        return None
