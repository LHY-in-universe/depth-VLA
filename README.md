# depth-VLA

DINOv3 + Qwen3.5-2B vision-language model with SpatialRGPT-style region prompts and precomputed pseudo-depth.

## Architecture

```
RGB image  ──► DINOv3 (LoRA) ──┐
Depth PNG  ──► DINOv3 (LoRA) ──┤── 3-way Conv fusion ──┐
RGB image  ──► Qwen visual ────┘                       │
                                                       ▼
                       region pooling on <mask> regions
                                                       │
                                                       ▼
                              Qwen3.5-2B LLM (LoRA) → output
```

- **DINOv3 ViT-Base** encodes RGB and 3-channel pseudo-depth (shared backbone, two passes)
- **Qwen3.5-2B**'s built-in visual encoder is intercepted via forward hook
- Three token streams fused via `Conv1d` (`token_fusion.py`)
- For each `<mask>` token in the prompt, `RegionPooler` mask-pools fused visual features and injects them at the corresponding embedding position

## Pipeline

1. **Download data** — OpenSpatialDataset annotations + OpenImages V7 image subset
   ```bash
   python scripts/download_data.py --output_root data --subset_size 100000
   ```

2. **Generate pseudo-depth** — DINOv3 Depther (`dinov3_vit7b16_dd`) → 3-channel uint8 PNG
   ```bash
   python scripts/preprocess_depth.py \
       --image_root data/openimages/train \
       --depth_root data/relative_depth/raw \
       --dinov3_repo /path/to/dinov3 \
       --depther_ckpt /path/to/dinov3_vit7b16_dd.pth \
       --backbone_ckpt /path/to/dinov3_vit7b16.pth
   ```

3. **Train**
   ```bash
   python scripts/train.py --model_config configs/model.yaml --train_config configs/train.yaml
   ```

## ⚠️ Known unverified assumptions

These were inferred from the SpatialRGPT source / DINOv3 docs but **must be verified before training**:

### 1. OpenSpatialDataset field names
`src/data/utils.py::process_masks` looks for sample-level fields named `rle` / `masks` / `bboxes` / `boxes`. The actual SpatialRGPT JSON (`result_10_depth_convs.json`, ~32 GB) may use different keys (e.g. `mask_path`, `seg`, `segmentations`).

**Action:** after downloading, inspect a sample:
```bash
python -c "import json; d=json.load(open('data/annotations/result_10_depth_convs.json')); print(json.dumps(d[0], indent=2)[:3000])"
```
Then edit `process_masks` keys accordingly.

### 2. Qwen3.5-2B special token IDs
`src/models/depth_vla.py::_resolve_token_ids` reads `image_token_id` / `depth_token_id` / `mask_token_id` from `qwen.config` via `getattr`. These attribute names are guesses — Qwen3.5-2B may store them differently (e.g. only `image_token_id` is exposed; `<mask>` / `<depth>` may need to be added to the tokenizer manually).

**Action:** after loading the model:
```python
print(model.qwen.config)
print(qwen_processor.tokenizer.convert_tokens_to_ids(["<image>", "<depth>", "<mask>"]))
```
If `<mask>` / `<depth>` aren't recognised tokens, add them via `tokenizer.add_special_tokens(...)` and resize the embedding layer (`model.qwen.resize_token_embeddings(len(tokenizer))`).

### 3. Hidden dim alignment
`configs/model.yaml` sets `dino.output_dim = 2048` to match Qwen3.5-2B's hidden size. The Qwen3.5-2B HF card lists hidden_size = 2048, but verify with:
```python
print(model.qwen.config.hidden_size)
```
If different, update `dino.output_dim` (and `fusion.dim` automatically follows).

### 4. Qwen visual encoder attribute name
`DepthVLA._find_visual_encoder` searches for child modules named `visual` / `vision_model` / `vision_encoder` / `img_encoder`. Qwen3.5-2B may use a different name.

**Action:** after loading:
```python
for name, _ in model.qwen.named_children():
    print(name)
```
Patch `_find_visual_encoder` if none of the candidates match.

### 5. DINOv3 Depther hardware
`dinov3_vit7b16_dd` is a 7 B parameter model — depth pre-generation needs ≥80 GB GPU. Alternatives:
- Use a smaller DINOv3 ConvNext depth variant once Meta releases one
- Fall back to DepthAnythingV2 (the original SpatialRGPT choice) — replace the model loading in `scripts/preprocess_depth.py::load_dinov3_depther`

### 6. Label masking
`SpatialRGPTDataset._build_labels` uses a substring search to locate assistant token spans. This is fragile when assistant text appears verbatim earlier in the prompt. For production, replace with proper turn-aligned masking using `tokenizer.apply_chat_template(..., return_assistant_tokens_mask=True)`.

## Project layout

```
depth-VLA/
├── configs/
│   ├── model.yaml          # DINOv3 / Qwen / LoRA config
│   └── train.yaml          # data paths, staged training schedule
├── src/
│   ├── models/
│   │   ├── dino_encoder.py     # DINOv3 + LoRA
│   │   ├── token_fusion.py     # Conv1d 3-way fusion
│   │   ├── region_pooler.py    # mask-guided RoI pooling
│   │   └── depth_vla.py        # full model
│   ├── data/
│   │   ├── dataset.py          # SpatialRGPTDataset (OSD format)
│   │   ├── collator.py
│   │   └── utils.py            # process_depth / process_masks / RLE
│   ├── training/trainer.py     # 3-stage StagedTrainer
│   └── utils/lora.py           # adapter checkpoint save/load
├── scripts/
│   ├── download_data.py        # OSD + OpenImages V7 subset download
│   ├── preprocess_depth.py     # offline DINOv3 Depther → 3ch PNG
│   ├── train.py
│   └── infer.py
└── requirements.txt
```

## Training stages

| Stage | Trainable params | Purpose |
|-------|------------------|---------|
| 1 | fusion + region_pooler + DINO proj | warm up the new modules against frozen backbones |
| 2 | + DINOv3 LoRA | let DINO adapt to spatial reasoning data |
| 3 | + Qwen LoRA | end-to-end joint optimisation |

Configured in `configs/train.yaml::stages`.
