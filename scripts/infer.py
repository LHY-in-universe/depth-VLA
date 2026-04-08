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
from transformers import AutoImageProcessor, AutoProcessor

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.models.depth_vla import DepthVLA
from src.utils.lora import load_adapter_checkpoint


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_config",        required=True)
    parser.add_argument("--adapter_checkpoint",  required=True)
    parser.add_argument("--image",               required=True)
    parser.add_argument("--prompt",              default="Describe the scene.")
    parser.add_argument("--max_new_tokens",      type=int, default=256)
    parser.add_argument("--save_depth",          default=None, help="Save depth map to this PNG path")
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
    dino_processor = AutoImageProcessor.from_pretrained(
        model_cfg["dino"]["model_name"]
    )

    # ── Prepare inputs ────────────────────────────────────────────────────
    image = Image.open(args.image).convert("RGB")
    text  = f"User: <image>\n{args.prompt}\nAssistant:"

    qwen_inputs = qwen_processor(
        text=text, images=image, return_tensors="pt"
    ).to(args.device)

    dino_inputs = dino_processor(images=image, return_tensors="pt").to(args.device)

    # ── Forward (depth map) ───────────────────────────────────────────────
    with torch.inference_mode():
        outputs = model(
            input_ids=qwen_inputs["input_ids"],
            attention_mask=qwen_inputs["attention_mask"],
            pixel_values=qwen_inputs["pixel_values"],
            dino_pixel_values=dino_inputs["pixel_values"],
            **{k: v for k, v in qwen_inputs.items()
               if k not in ("input_ids", "attention_mask", "pixel_values")},
        )

    # ── Text generation ───────────────────────────────────────────────────
    with torch.inference_mode():
        generated_ids = model.qwen.generate(
            input_ids=qwen_inputs["input_ids"],
            attention_mask=qwen_inputs["attention_mask"],
            pixel_values=qwen_inputs["pixel_values"],
            max_new_tokens=args.max_new_tokens,
            do_sample=False,
        )

    response = qwen_processor.tokenizer.decode(
        generated_ids[0][qwen_inputs["input_ids"].shape[1]:],
        skip_special_tokens=True,
    )
    print(f"\n[Response]\n{response}")

    # ── Save depth map ────────────────────────────────────────────────────
    if args.save_depth and outputs["depth_map"] is not None:
        import numpy as np
        depth_np = outputs["depth_map"][0, 0].cpu().float().numpy()
        # Normalise to 16-bit PNG
        depth_mm = (depth_np * 1000).clip(0, 65535).astype("uint16")
        depth_img = Image.fromarray(depth_mm)
        depth_img.save(args.save_depth)
        print(f"Depth map saved → {args.save_depth}")


if __name__ == "__main__":
    main()
