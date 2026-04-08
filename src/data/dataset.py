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
"""

from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any

import torch
from PIL import Image
from torch.utils.data import Dataset
from transformers import AutoImageProcessor, AutoProcessor

from .utils import (
    DEFAULT_DEPTH_TOKEN,
    DEFAULT_IMAGE_TOKEN,
    DEFAULT_MASK_TOKEN,
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
        dino_image_processor: AutoImageProcessor,
        max_length: int = 2048,
        max_regions: int = 32,
        subset_size: int | None = None,
    ):
        self.image_root = Path(image_root)
        self.depth_root = Path(depth_root)
        self.qwen_processor = qwen_processor
        self.dino_processor = dino_image_processor
        self.max_length = max_length
        self.max_regions = max_regions

        print(f"Loading dataset JSON from {json_path} (this may take a while for large files)...")
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
        image = Image.open(image_path).convert("RGB")
        W, H = image.size

        # ── Load 3-channel depth (precomputed) ────────────────────────────
        depth_image = process_depth(filename, self.depth_root)

        # ── Decode masks for <mask> tokens in conversation ────────────────
        masks, mask_valid = process_masks(sample, image_h=H, image_w=W, max_regions=self.max_regions)

        # ── Build prompt text ─────────────────────────────────────────────
        text = preprocess_multimodal_text(sample["conversations"], inject_depth_token=True)

        # ── Tokenise text + RGB through Qwen processor ────────────────────
        qwen_inputs = self.qwen_processor(
            text=text,
            images=[image, depth_image],   # pass two "images": rgb + depth
            return_tensors="pt",
            max_length=self.max_length,
            truncation=True,
            padding="max_length",
        )

        # ── DINOv3 preprocessing for both rgb and depth ───────────────────
        dino_rgb_inputs   = self.dino_processor(images=image,       return_tensors="pt")
        dino_depth_inputs = self.dino_processor(images=depth_image, return_tensors="pt")

        # ── Build labels (mask non-assistant tokens) ──────────────────────
        input_ids = qwen_inputs["input_ids"].squeeze(0)
        labels = self._build_labels(input_ids, sample["conversations"])

        out = {
            "input_ids":          input_ids,
            "attention_mask":     qwen_inputs["attention_mask"].squeeze(0),
            "pixel_values":       qwen_inputs["pixel_values"].squeeze(0),  # qwen processed (rgb+depth)
            "dino_rgb_pixel_values":   dino_rgb_inputs["pixel_values"].squeeze(0),
            "dino_depth_pixel_values": dino_depth_inputs["pixel_values"].squeeze(0),
            "masks":              masks,        # [R, H, W]
            "mask_valid":         mask_valid,   # [R]
            "labels":             labels,
        }

        # Pass through Qwen-specific extras (e.g. image_grid_thw)
        for k, v in qwen_inputs.items():
            if k not in out and isinstance(v, torch.Tensor):
                out[k] = v.squeeze(0)

        return out

    # ──────────────────────────────────────────────────────────────────────
    # Label masking
    # ──────────────────────────────────────────────────────────────────────

    def _build_labels(self, input_ids: torch.Tensor, conversations: list[dict]) -> torch.Tensor:
        """
        Mask everything that isn't an assistant ('gpt') response with -100.

        Heuristic: locate the assistant text spans by re-tokenising the
        assistant turns and matching against `input_ids`. For production,
        replace with proper turn-aligned masking using the tokenizer's
        chat template offsets.
        """
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
