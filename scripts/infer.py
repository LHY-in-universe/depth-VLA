#!/usr/bin/env python3
"""
Inference script for depth-VLA.

Usage:
    python scripts/infer.py \
        --model_config configs/model.yaml \
        --adapter_checkpoint checkpoints/stage3_epoch5.pt \
        --image path/to/image.jpg \
        --prompt "Describe the depth and layout of the scene."
"""

import argparse
import sys
from pathlib import Path

import torch
import yaml
from PIL import Image
from transformers import AutoProcessor

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data.utils import process_depth
from src.models.depth_vla import DepthVLA
from src.utils.lora import load_adapter_checkpoint


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_config",        required=True)
    parser.add_argument("--adapter_checkpoint",  required=True)
    parser.add_argument("--image",               required=True)
    parser.add_argument("--depth",               required=True, help="Path to precomputed 3-channel depth PNG")
    parser.add_argument("--prompt",              default="Describe the scene.")
    parser.add_argument("--max_new_tokens",      type=int, default=256)
    parser.add_argument("--device",              default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def main():
    args = parse_args()

    with open(args.model_config) as f:
        model_cfg = yaml.safe_load(f)

    # ── Model ─────────────────────────────────────────────────────────────
    print("Loading model...")
    model = DepthVLA.from_config(model_cfg)
    load_adapter_checkpoint(model, args.adapter_checkpoint)
    model.eval().to(args.device)

    # ── Processors ────────────────────────────────────────────────────────
    qwen_processor = AutoProcessor.from_pretrained(
        model_cfg["qwen"]["model_name"], trust_remote_code=True
    )

    # ── Prepare inputs ────────────────────────────────────────────────────
    rgb_image   = Image.open(args.image).convert("RGB")
    depth_image = Image.open(args.depth).convert("RGB")
    text  = f"User: <image>\n{args.prompt}\nAssistant:"

    rgb_inputs = qwen_processor(
        text=text, images=rgb_image, return_tensors="pt"
    ).to(args.device)
    depth_inputs = qwen_processor.image_processor(
        images=depth_image, return_tensors="pt"
    ).to(args.device)

    # Empty masks (no <mask> tokens in this prompt)
    import torch as _torch
    masks = _torch.zeros(1, model.max_regions, 1, 1, dtype=_torch.uint8, device=args.device)
    mask_valid = _torch.zeros(1, model.max_regions, dtype=_torch.bool, device=args.device)

    # ── Forward + decode ──────────────────────────────────────────────────
    with torch.inference_mode():
        outputs = model(
            input_ids=rgb_inputs["input_ids"],
            attention_mask=rgb_inputs["attention_mask"],
            rgb_pixel_values=rgb_inputs["pixel_values"],
            depth_pixel_values=depth_inputs["pixel_values"],
            masks=masks,
            mask_valid=mask_valid,
        )
    next_tokens = outputs["logits"][0, -1].argmax(-1, keepdim=True)
    print(f"\n[Next token id] {next_tokens.item()}")
    print(f"[Decoded] {qwen_processor.tokenizer.decode(next_tokens)}")
    print("(Note: full generation requires extending forward to support cache; "
          "for demo, only next-token prediction is shown.)")


if __name__ == "__main__":
    main()
