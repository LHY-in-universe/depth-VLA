#!/usr/bin/env python3
"""
Offline depth pre-generation using DINOv3 Depther.

For each image in --image_root, runs the official DINOv3 depth decoder
(`dinov3_vit7b16_dd`, trained on SYNTHMIX) and saves a 3-channel pseudo-RGB
PNG to --depth_root, following SpatialRGPT's normalisation:

    depth_norm = (d - d.min()) / (d.max() - d.min()) * 255
    depth_3ch  = stack([depth_norm] * 3, axis=-1).astype(uint8)

Usage:
    python scripts/preprocess_depth.py \
        --image_root  data/openimages/train \
        --depth_root  data/relative_depth/raw \
        --dinov3_repo /path/to/dinov3 \
        --depther_ckpt /path/to/dinov3_vit7b16_dd.pth \
        --backbone_ckpt /path/to/dinov3_vit7b16.pth \
        --batch_size 4 \
        --num_workers 8

Notes:
    - The 7B Depther needs an A100/H100 80G. For smaller GPUs, use a
      ConvNext-based DINOv3 depth variant or fall back to DepthAnythingV2.
    - We process in batches and skip images whose depth file already exists.
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import v2 as T
from tqdm import tqdm


# ─────────────────────────────────────────────────────────────────────────
# Image dataset
# ─────────────────────────────────────────────────────────────────────────

class ImageOnlyDataset(Dataset):
    def __init__(
        self,
        image_root: Path,
        depth_root: Path,
        transform,
        skip_existing: bool = True,
    ):
        all_imgs = sorted(image_root.glob("*.jpg")) + sorted(image_root.glob("*.jpeg"))
        if skip_existing:
            self.images = [
                p for p in all_imgs
                if not (depth_root / f"{p.stem}.png").exists()
            ]
        else:
            self.images = all_imgs
        self.transform = transform
        print(f"Found {len(all_imgs)} images, {len(self.images)} pending.")

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        path = self.images[idx]
        try:
            img = Image.open(path).convert("RGB")
            tensor = self.transform(img)
            return tensor, path.stem, img.size  # (W, H)
        except Exception as e:
            print(f"[skip] {path}: {e}")
            return None


def collate_skip_none(batch):
    batch = [b for b in batch if b is not None]
    if not batch:
        return None
    tensors = torch.stack([b[0] for b in batch])
    stems   = [b[1] for b in batch]
    sizes   = [b[2] for b in batch]
    return tensors, stems, sizes


# ─────────────────────────────────────────────────────────────────────────
# DINOv3 Depther loading
# ─────────────────────────────────────────────────────────────────────────

def load_dinov3_depther(repo_dir: str, depther_ckpt: str, backbone_ckpt: str, device: str):
    """Load the official DINOv3 depth decoder via torch.hub local source."""
    depther = torch.hub.load(
        repo_dir,
        "dinov3_vit7b16_dd",
        source="local",
        weights=depther_ckpt,
        backbone_weights=backbone_ckpt,
    )
    depther.eval().to(device)
    return depther


# ─────────────────────────────────────────────────────────────────────────
# Depth → 3-channel uint8 PNG
# ─────────────────────────────────────────────────────────────────────────

def depth_to_3channel_png(depth: torch.Tensor) -> np.ndarray:
    """
    Args:
        depth: [H, W] float tensor
    Returns:
        rgb: [H, W, 3] uint8 numpy
    """
    d = depth.float().cpu().numpy()
    lo, hi = d.min(), d.max()
    if hi - lo < 1e-6:
        norm = np.zeros_like(d, dtype=np.uint8)
    else:
        norm = ((d - lo) / (hi - lo) * 255.0).astype(np.uint8)
    return np.stack([norm, norm, norm], axis=-1)


# ─────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--image_root",     required=True, type=Path)
    p.add_argument("--depth_root",     required=True, type=Path)
    p.add_argument("--dinov3_repo",    required=True, help="Local path to facebookresearch/dinov3")
    p.add_argument("--depther_ckpt",   required=True)
    p.add_argument("--backbone_ckpt",  required=True)
    p.add_argument("--input_size",     type=int, default=1024)
    p.add_argument("--batch_size",     type=int, default=4)
    p.add_argument("--num_workers",    type=int, default=8)
    p.add_argument("--device",         default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--no_skip",        action="store_true", help="Reprocess existing depth files")
    return p.parse_args()


def main():
    args = parse_args()
    args.depth_root.mkdir(parents=True, exist_ok=True)

    transform = T.Compose([
        T.ToImage(),
        T.Resize((args.input_size, args.input_size), antialias=True),
        T.ToDtype(torch.float32, scale=True),
        T.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ])

    print("Loading DINOv3 Depther...")
    depther = load_dinov3_depther(
        args.dinov3_repo, args.depther_ckpt, args.backbone_ckpt, args.device
    )

    ds = ImageOnlyDataset(
        image_root=args.image_root,
        depth_root=args.depth_root,
        transform=transform,
        skip_existing=not args.no_skip,
    )
    loader = DataLoader(
        ds,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        collate_fn=collate_skip_none,
        pin_memory=True,
    )

    pbar = tqdm(loader, total=len(loader), desc="Depth")
    with torch.inference_mode():
        for batch in pbar:
            if batch is None:
                continue
            tensors, stems, sizes = batch
            tensors = tensors.to(args.device, non_blocking=True)

            # Forward
            depths = depther(tensors)  # [B, 1, H, W]

            # Save (resize back to original aspect ratio)
            for d, stem, (W, H) in zip(depths, stems, sizes):
                d_resized = torch.nn.functional.interpolate(
                    d.unsqueeze(0), size=(H, W), mode="bilinear", align_corners=False
                )[0, 0]
                rgb = depth_to_3channel_png(d_resized)
                Image.fromarray(rgb).save(args.depth_root / f"{stem}.png")


if __name__ == "__main__":
    main()
