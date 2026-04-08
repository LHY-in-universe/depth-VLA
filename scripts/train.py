#!/usr/bin/env python3
"""
Training entry point for depth-VLA (SpatialRGPT-format data).

Usage:
    python scripts/train.py \
        --model_config configs/model.yaml \
        --train_config configs/train.yaml
"""

import argparse
import sys
from pathlib import Path

import torch
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from transformers import AutoProcessor

from src.data.collator import DepthVLACollator
from src.data.dataset import SpatialRGPTDataset
from src.models.depth_vla import DepthVLA
from src.training.trainer import StagedTrainer


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model_config", default="configs/model.yaml")
    p.add_argument("--train_config", default="configs/train.yaml")
    p.add_argument("--resume",       default=None)
    p.add_argument("--device",       default="cuda" if torch.cuda.is_available() else "cpu")
    return p.parse_args()


def main():
    args = parse_args()

    with open(args.model_config) as f:
        model_cfg = yaml.safe_load(f)
    with open(args.train_config) as f:
        train_cfg = yaml.safe_load(f)

    # ── Model ─────────────────────────────────────────────────────────────
    print("Loading model...")
    model = DepthVLA.from_config(model_cfg)
    if args.resume:
        from src.utils.lora import load_adapter_checkpoint
        load_adapter_checkpoint(model, args.resume)

    # ── Processors ────────────────────────────────────────────────────────
    qwen_processor = AutoProcessor.from_pretrained(
        model_cfg["qwen"]["model_name"], trust_remote_code=True
    )

    # ── Datasets ──────────────────────────────────────────────────────────
    data_cfg = train_cfg["data"]
    common = dict(
        image_root=data_cfg["image_root"],
        depth_root=data_cfg["depth_root"],
        qwen_processor=qwen_processor,
        max_length=data_cfg.get("max_length", 2048),
        max_regions=data_cfg.get("max_regions", 32),
    )
    train_ds = SpatialRGPTDataset(
        json_path=data_cfg["train_json"],
        subset_size=data_cfg.get("subset_size"),
        **common,
    )
    val_ds = SpatialRGPTDataset(
        json_path=data_cfg["val_json"],
        subset_size=None,
        **common,
    )

    collator = DepthVLACollator(
        pad_token_id=qwen_processor.tokenizer.pad_token_id or 0
    )

    train_loader = torch.utils.data.DataLoader(
        train_ds,
        batch_size=train_cfg.get("per_device_train_batch_size", 2),
        shuffle=True,
        num_workers=train_cfg.get("dataloader_num_workers", 4),
        collate_fn=collator,
        pin_memory=True,
    )
    val_loader = torch.utils.data.DataLoader(
        val_ds,
        batch_size=train_cfg.get("per_device_eval_batch_size", 4),
        shuffle=False,
        num_workers=train_cfg.get("dataloader_num_workers", 4),
        collate_fn=collator,
        pin_memory=True,
    )

    trainer = StagedTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        train_cfg=train_cfg,
        device=args.device,
    )
    trainer.train()


if __name__ == "__main__":
    main()
