# Qwen3-VL-4B Training Skeleton

This directory contains a text-generation fine-tuning scaffold built on top of `Qwen/Qwen3-VL-4B-Instruct`, with optional depth supervision aligned to the image tokens.

## Scope

- Image + prompt input
- Text response supervision
- Standard causal LM loss for text output
- Auxiliary depth alignment loss using frozen `MoGe + LingBot-Depth`
- No VLA action head

## Dataset format

Use `json` or `jsonl` records in one of these forms:

```json
{"image": "images/example.jpg", "prompt": "Describe the scene.", "response": "A robot arm is placing a cup on the table."}
```

```json
{"images": ["images/example.jpg"], "prompt": "What is in front of the bowl?", "response": "A mug is positioned in front of the bowl."}
```

## Notes

- The official model id is `Qwen/Qwen3-VL-4B-Instruct`.
- Official guidance currently recommends a recent `transformers` build with `Qwen3VLForConditionalGeneration`.
- The training entry point is [tasks/vl/train_qwen3_vl_4b.py](/Users/lhy/Desktop/lingbot-vla/tasks/vl/train_qwen3_vl_4b.py).
- Default depth backbones are:
  - `Ruicheng/moge-2-vitb-normal`
  - `robbyant/lingbot-depth-pretrain-vitl-14-v0.5`
- `LingBot-Depth-v0.5` is the currently recommended official release from Robbyant.

## Recommended command

```bash
python3 tasks/vl/train_qwen3_vl_4b.py \
  --config configs/vl/qwen3_vl_4b_sft.yaml
```

## Source links

- LingBot-Depth official repository: https://github.com/robbyant/lingbot-depth
- LingBot-Depth recommended checkpoint: https://huggingface.co/robbyant/lingbot-depth-pretrain-vitl-14-v0.5
- MoGe checkpoint: https://huggingface.co/Ruicheng/moge-2-vitb-normal
