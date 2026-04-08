#!/usr/bin/env python3
"""
Download OpenSpatialDataset (annotations) + sample 100K images from OpenImages V7.

Steps:
    1. Download `result_10_depth_convs.json` from a8cheng/OpenSpatialDataset (~32GB)
    2. Sub-sample to 100K entries
    3. Resolve which OpenImages V7 image IDs are referenced
    4. Download only those images via the OpenImages download tool

Usage:
    python scripts/download_data.py \
        --output_root data \
        --subset_size 100000

Requires: huggingface_hub, openimages (pip install openimages)
"""

import argparse
import json
import random
import subprocess
import sys
from pathlib import Path


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--output_root", type=Path, default=Path("data"))
    p.add_argument("--subset_size", type=int, default=100_000)
    p.add_argument("--seed",        type=int, default=42)
    p.add_argument("--skip_images", action="store_true",
                   help="Only download annotations, skip image download")
    return p.parse_args()


def download_annotations(output_root: Path) -> Path:
    from huggingface_hub import hf_hub_download
    print("Downloading OpenSpatialDataset annotations from HF...")
    path = hf_hub_download(
        repo_id="a8cheng/OpenSpatialDataset",
        filename="result_10_depth_convs.json",
        repo_type="dataset",
        local_dir=str(output_root / "annotations"),
    )
    return Path(path)


def subsample(json_path: Path, subset_size: int, seed: int, output_path: Path):
    print(f"Loading full annotations from {json_path}...")
    with open(json_path) as f:
        data = json.load(f)
    print(f"  total: {len(data)} samples")

    if subset_size >= len(data):
        subset = data
    else:
        random.seed(seed)
        subset = random.sample(data, subset_size)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(subset, f)
    print(f"  saved subset of {len(subset)} → {output_path}")
    return subset


def write_image_id_list(subset: list, output_path: Path):
    ids = sorted({s["filename"] for s in subset})
    with open(output_path, "w") as f:
        for img_id in ids:
            f.write(f"train/{img_id}\n")
    print(f"  wrote {len(ids)} image IDs → {output_path}")
    return output_path


def download_openimages(id_list_path: Path, image_root: Path):
    """Use the official openimages downloader CLI."""
    image_root.mkdir(parents=True, exist_ok=True)
    print("Downloading OpenImages V7 subset (this can take hours)...")
    cmd = [
        sys.executable, "-m", "openimages.download",
        "--image_list", str(id_list_path),
        "--download_folder", str(image_root),
        "--num_processes", "8",
    ]
    print("  $", " ".join(cmd))
    subprocess.run(cmd, check=True)


def main():
    args = parse_args()
    args.output_root.mkdir(parents=True, exist_ok=True)

    full_json = download_annotations(args.output_root)
    subset_json = args.output_root / "annotations" / f"osd_subset_{args.subset_size}.json"
    subset = subsample(full_json, args.subset_size, args.seed, subset_json)

    id_list = args.output_root / "annotations" / "image_ids.txt"
    write_image_id_list(subset, id_list)

    if not args.skip_images:
        download_openimages(id_list, args.output_root / "openimages" / "train")

    print("\nDone. Next steps:")
    print(f"  1. Generate depth maps:")
    print(f"     python scripts/preprocess_depth.py \\")
    print(f"         --image_root {args.output_root / 'openimages' / 'train'} \\")
    print(f"         --depth_root {args.output_root / 'relative_depth' / 'raw'} \\")
    print(f"         --dinov3_repo /path/to/dinov3 ...")
    print(f"  2. Train:")
    print(f"     python scripts/train.py \\")
    print(f"         --train_json {subset_json}")


if __name__ == "__main__":
    main()
