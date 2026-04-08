# depth-VLA

Qwen3.5-2B-based VLM augmented with **dual-pass visual encoding**: the same Qwen visual encoder (with LoRA) processes both the original RGB image and a precomputed pseudo-depth map, then the two token streams are conv-fused before going into the LLM.

## Architecture

```
                           ┌─────────────────────────────┐
                           │  OFFLINE (preprocess once)  │
RGB image  ───► DINOv3 Depther ─► 3-channel depth PNG ───┘
                           
TRAIN / INFER:
RGB image    ──► Qwen visual encoder (LoRA) ──► rgb_tokens   ┐
Depth PNG    ──► Qwen visual encoder (LoRA, shared) ──► depth_tokens ┤
                                                                     │
                                                  Conv1d fusion ◄────┘
                                                       │
                                                  fused_tokens
                                                       │
                                  region_pooler(fused, mask_i) per <mask>
                                                       │
                  ┌──────────────────────────────────────┐
                  │ inputs_embeds:                       │
                  │   <image> positions ← fused_tokens   │
                  │   <mask>  positions ← region_tokens  │
                  └──────────────────────────────────────┘
                                                       │
                                              Qwen3.5-2B LLM (LoRA)
                                                       │
                                                    logits
```

Key points:
- **DINOv3** is used **only offline** in `preprocess_depth.py` to convert RGB → depth (`dinov3_vit7b16_dd`)
- **Single ViT** = Qwen3.5-2B's built-in visual encoder, **shared between RGB and depth**
- **LoRA on the visual encoder is active only during the depth pass.** The RGB pass runs under a `disable_adapter` context manager, so Qwen's pretrained RGB representation is preserved untouched. The same backbone weights are shared; only the adapter is toggled per pass.
- **Conv1d fusion** merges the two token streams (`token_fusion.py`)
- **Region pooling** mask-pools fused visual tokens for each `<mask>` placeholder

## Pipeline

1. **Download data** — OpenSpatialDataset annotations + OpenImages V7 image subset
   ```bash
   python scripts/download_data.py --output_root data --subset_size 100000
   ```

2. **Generate pseudo-depth** — DINOv3 Depther → 3-channel uint8 PNG
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

## Training stages

| Stage | Trainable params | Purpose |
|-------|------------------|---------|
| 1 | fusion + region_pooler | warm up new modules against frozen Qwen |
| 2 | + Qwen visual LoRA | adapt visual encoder to depth modality |
| 3 | + Qwen LLM LoRA | end-to-end joint optimisation |

## ⚠️ Known unverified assumptions

These were inferred from SpatialRGPT source / DINOv3 / Qwen docs and **must be verified before training**:

### 1. OpenSpatialDataset field names
`src/data/utils.py::process_masks` expects `rle` / `masks` / `bboxes` / `boxes` keys. Real OSD JSON may differ.

**Action:** after downloading, inspect a sample:
```bash
python -c "import json; d=json.load(open('data/annotations/result_10_depth_convs.json')); print(json.dumps(d[0], indent=2)[:3000])"
```
and patch `process_masks` keys.

### 2. Qwen3.5-2B special token IDs (`<image>`, `<mask>`)
`DepthVLA._resolve_token_ids` reads `image_token_id` / `mask_token_id` from `qwen.config` via `getattr`. Qwen3.5-2B exposes `<image>` natively, but `<mask>` is **not** a built-in special token and must be added.

**Action:** after loading the model:
```python
print(model.qwen.config)
print(qwen_processor.tokenizer.convert_tokens_to_ids(["<image>", "<mask>"]))
```
If `<mask>` returns the unk id, register it:
```python
qwen_processor.tokenizer.add_special_tokens({"additional_special_tokens": ["<mask>"]})
model.qwen.resize_token_embeddings(len(qwen_processor.tokenizer))
model._mask_token_id = qwen_processor.tokenizer.convert_tokens_to_ids("<mask>")
```

### 3. Qwen visual encoder attribute name
`DepthVLA._find_visual_encoder` searches for child modules named `visual` / `vision_model` / `vision_encoder` / `img_encoder`. Qwen3.5-2B may use a different name.

**Action:** after loading:
```python
for name, _ in model.qwen.named_children():
    print(name)
```
and patch `_find_visual_encoder` if none of the candidates match.

### 4. Visual encoder LoRA target module names
`configs/model.yaml::visual_lora.target_modules` defaults to `["q_proj", "v_proj", "qkv"]`. The actual layer names inside Qwen's visual encoder need verification. The model `__init__` will raise if 0 LoRA layers were injected.

**Action:** after loading:
```python
ve = model._find_visual_encoder()
for name, mod in ve.named_modules():
    if isinstance(mod, torch.nn.Linear):
        print(name)
```
and update `target_modules` to match.

> **Note on LoRA gradient flow:** the visual LoRA adapter only sees gradients from the **depth pass** (RGB runs under `_visual_lora_disabled()`). Effective batch size for the LoRA params equals the regular batch size — not double — even though the visual encoder is invoked twice per step.

### 5. Visual encoder forward signature
`DepthVLA._encode_visual` calls `visual_encoder(pixel_values, **extras)` where `extras` may include `grid_thw`. Different Qwen VL variants accept different kwargs (e.g. `image_grid_thw`, `pixel_values_videos`, etc.).

**Action:** check `inspect.signature(model._find_visual_encoder().forward)` and adjust `_encode_visual` accordingly.

### 6. Spatial layout from N tokens
`DepthVLA.forward` infers patch grid as `(√N, √N)`. If Qwen pads or uses non-square layouts, replace with `image_grid_thw`-derived dimensions.

### 7. DINOv3 Depther hardware
`dinov3_vit7b16_dd` is 7 B params — depth pre-generation needs ≥80 GB GPU. Alternatives:
- Smaller DINOv3 ConvNext depth variants when released
- Fall back to DepthAnythingV2 — replace `load_dinov3_depther` in `scripts/preprocess_depth.py`

### 8. Label masking heuristic
`SpatialRGPTDataset._build_labels` uses substring search to find assistant token spans. Fragile if assistant text appears verbatim earlier. For production, use `tokenizer.apply_chat_template(..., return_assistant_tokens_mask=True)`.

## Project layout

```
depth-VLA/
├── configs/
│   ├── model.yaml          # Qwen + visual_lora + llm_lora config
│   └── train.yaml          # data paths, staged training schedule
├── src/
│   ├── models/
│   │   ├── token_fusion.py     # Conv1d 2-stream fusion
│   │   ├── region_pooler.py    # mask-guided RoI pooling
│   │   └── depth_vla.py        # full model (Qwen + dual visual pass + fusion + region)
│   ├── data/
│   │   ├── dataset.py          # SpatialRGPTDataset
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
