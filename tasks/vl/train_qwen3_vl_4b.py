from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict

import yaml
from transformers import Trainer, TrainingArguments

from lingbotvla.models.vl.qwen3_vl_4b import (
    Qwen3VLDepthConfig,
    Qwen3VL4BTrainCollator,
    Qwen3VL4BTrainDataset,
    build_depth_aware_qwen3_vl_4b_model,
)


def _parse_args() -> argparse.Namespace:
    bootstrap = argparse.ArgumentParser(add_help=False)
    bootstrap.add_argument("--config", type=str, default=None)
    bootstrap_args, remaining = bootstrap.parse_known_args()

    defaults: Dict[str, Any] = {}
    if bootstrap_args.config:
        with open(bootstrap_args.config, "r", encoding="utf-8") as fh:
            defaults = yaml.safe_load(fh) or {}

    parser = argparse.ArgumentParser(description="Fine-tune Qwen3-VL-4B for image-to-text supervision.")
    parser.set_defaults(**defaults)

    parser.add_argument("--config", type=str, default=bootstrap_args.config)
    parser.add_argument("--model_name_or_path", type=str, default="Qwen/Qwen3-VL-4B-Instruct")
    parser.add_argument("--train_path", type=str, required="train_path" not in defaults)
    parser.add_argument("--image_root", type=str, default=None)
    parser.add_argument("--output_dir", type=str, required="output_dir" not in defaults)
    parser.add_argument("--image_column", type=str, default="image")
    parser.add_argument("--images_column", type=str, default="images")
    parser.add_argument("--prompt_column", type=str, default="prompt")
    parser.add_argument("--response_column", type=str, default="response")
    parser.add_argument("--max_length", type=int, default=2048)
    parser.add_argument("--depth_image_size", type=int, default=224)
    parser.add_argument("--learning_rate", type=float, default=1e-5)
    parser.add_argument("--num_train_epochs", type=float, default=1.0)
    parser.add_argument("--per_device_train_batch_size", type=int, default=1)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=16)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--warmup_ratio", type=float, default=0.03)
    parser.add_argument("--logging_steps", type=int, default=10)
    parser.add_argument("--save_steps", type=int, default=200)
    parser.add_argument("--save_total_limit", type=int, default=2)
    parser.add_argument("--dataloader_num_workers", type=int, default=4)
    parser.add_argument("--lr_scheduler_type", type=str, default="cosine")
    parser.add_argument("--torch_dtype", type=str, default="bfloat16")
    parser.add_argument("--attn_implementation", type=str, default="flash_attention_2")
    parser.add_argument("--gradient_checkpointing", action="store_true")
    parser.add_argument("--bf16", action="store_true")
    parser.add_argument("--tf32", action="store_true")
    parser.add_argument("--report_to", type=str, default="none")
    parser.add_argument("--moge_path", type=str, default="Ruicheng/moge-2-vitb-normal")
    parser.add_argument("--morgbd_path", type=str, default="robbyant/lingbot-depth-pretrain-vitl-14-v0.5")
    parser.add_argument("--depth_loss_weight", type=float, default=0.1)
    parser.add_argument("--depth_num_layers", type=int, default=1)
    parser.add_argument("--depth_num_heads", type=int, default=4)
    parser.add_argument("--depth_dim_head", type=int, default=32)
    parser.add_argument("--depth_ff_mult", type=int, default=1)
    parser.add_argument("--depth_num_backbone_tokens", type=int, default=256)
    parser.add_argument("--depth_token_size", type=int, default=16)
    parser.add_argument("--depth_dim_out", type=int, default=1024)

    return parser.parse_args(remaining)


def main() -> None:
    args = _parse_args()
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    model, processor = build_depth_aware_qwen3_vl_4b_model(
        args.model_name_or_path,
        attn_implementation=args.attn_implementation,
        torch_dtype=args.torch_dtype,
        gradient_checkpointing=args.gradient_checkpointing,
        depth_config=Qwen3VLDepthConfig(
            moge_path=args.moge_path,
            morgbd_path=args.morgbd_path,
            depth_loss_weight=args.depth_loss_weight,
            num_layers=args.depth_num_layers,
            num_heads=args.depth_num_heads,
            dim_head=args.depth_dim_head,
            ff_mult=args.depth_ff_mult,
            num_backbone_tokens=args.depth_num_backbone_tokens,
            token_size=args.depth_token_size,
            dim_out=args.depth_dim_out,
            input_size=args.depth_image_size,
        ),
    )
    total_params, trainable_params = model.trainable_parameter_summary()
    print(f"total parameters: {total_params:,}")
    print(f"trainable parameters: {trainable_params:,}")
    print(f"trainable ratio: {trainable_params / total_params:.6%}")

    dataset = Qwen3VL4BTrainDataset(
        args.train_path,
        image_root=args.image_root,
        image_column=args.image_column,
        images_column=args.images_column,
        prompt_column=args.prompt_column,
        response_column=args.response_column,
    )
    collator = Qwen3VL4BTrainCollator(
        processor,
        max_length=args.max_length,
        depth_image_size=args.depth_image_size,
    )

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        learning_rate=args.learning_rate,
        num_train_epochs=args.num_train_epochs,
        per_device_train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        weight_decay=args.weight_decay,
        warmup_ratio=args.warmup_ratio,
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        save_total_limit=args.save_total_limit,
        dataloader_num_workers=args.dataloader_num_workers,
        lr_scheduler_type=args.lr_scheduler_type,
        bf16=args.bf16,
        tf32=args.tf32,
        remove_unused_columns=False,
        report_to=args.report_to,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        data_collator=collator,
    )
    trainer.train()
    trainer.save_model(args.output_dir)
    processor.save_pretrained(args.output_dir)


if __name__ == "__main__":
    main()
