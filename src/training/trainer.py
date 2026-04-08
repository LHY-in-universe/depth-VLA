"""
Staged trainer for DepthVLA.

Stage 1 — warmup:   train fusion + DINOv3 proj + depth head only
Stage 2 — dino lora: + DINOv3 LoRA
Stage 3 — joint:    + Qwen LoRA
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from ..models.depth_vla import DepthVLA
from ..utils.lora import print_trainable_params, save_adapter_checkpoint


class StagedTrainer:
    def __init__(
        self,
        model: DepthVLA,
        train_loader: DataLoader,
        val_loader: DataLoader,
        train_cfg: dict,
        device: str = "cuda",
    ):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.cfg = train_cfg
        self.device = device
        self.output_dir = Path(train_cfg["output_dir"])
        self.output_dir.mkdir(parents=True, exist_ok=True)

    # ──────────────────────────────────────────────────────────────────────
    # Main entry
    # ──────────────────────────────────────────────────────────────────────

    def train(self):
        stages = self.cfg["stages"]
        for stage_idx, stage_cfg in enumerate(stages, start=1):
            print(f"\n{'='*60}")
            print(f"  Stage {stage_idx}: {stage_cfg['name']}")
            print(f"{'='*60}")

            # Enable Qwen LoRA at stage 3 if not already done
            if stage_idx == 3 and not self._qwen_has_lora():
                qwen_cfg = self.cfg.get("qwen_lora", {})
                self.model.enable_qwen_lora(
                    rank=qwen_cfg.get("rank", 16),
                    alpha=qwen_cfg.get("alpha", 32),
                    dropout=qwen_cfg.get("dropout", 0.05),
                    target_modules=qwen_cfg.get("target_modules", ["q_proj", "v_proj"]),
                )

            self.model.set_trainable_stage(stage_idx)
            print_trainable_params(self.model, label=f"Stage {stage_idx}")

            optimizer = self._build_optimizer(stage_cfg["lr"])
            scheduler = self._build_scheduler(optimizer, stage_cfg["epochs"])

            for epoch in range(1, stage_cfg["epochs"] + 1):
                train_loss = self._train_epoch(optimizer, scheduler, stage_idx, epoch)
                val_loss   = self._val_epoch(stage_idx, epoch)

                print(
                    f"  Epoch {epoch}/{stage_cfg['epochs']} | "
                    f"train_loss={train_loss:.4f}  val_loss={val_loss:.4f}"
                )

                ckpt_path = self.output_dir / f"stage{stage_idx}_epoch{epoch}.pt"
                save_adapter_checkpoint(self.model, str(ckpt_path))

    # ──────────────────────────────────────────────────────────────────────
    # Epoch loops
    # ──────────────────────────────────────────────────────────────────────

    def _train_epoch(self, optimizer, scheduler, stage: int, epoch: int) -> float:
        self.model.train()
        total_loss = 0.0
        steps = 0
        grad_accum = self.cfg.get("gradient_accumulation_steps", 1)
        optimizer.zero_grad()

        pbar = tqdm(self.train_loader, desc=f"Train S{stage}E{epoch}", leave=False)
        for step, batch in enumerate(pbar):
            batch = self._to_device(batch)
            outputs = self.model(**batch)
            loss = outputs["loss"]

            if loss is None:
                continue

            (loss / grad_accum).backward()

            if (step + 1) % grad_accum == 0:
                nn.utils.clip_grad_norm_(
                    [p for p in self.model.parameters() if p.requires_grad], 1.0
                )
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

            total_loss += loss.item()
            steps += 1
            pbar.set_postfix(loss=f"{loss.item():.4f}")

        return total_loss / max(steps, 1)

    @torch.no_grad()
    def _val_epoch(self, stage: int, epoch: int) -> float:
        self.model.eval()
        total_loss = 0.0
        steps = 0

        for batch in tqdm(self.val_loader, desc=f"Val S{stage}E{epoch}", leave=False):
            batch = self._to_device(batch)
            outputs = self.model(**batch)
            if outputs["loss"] is not None:
                total_loss += outputs["loss"].item()
                steps += 1

        return total_loss / max(steps, 1)

    # ──────────────────────────────────────────────────────────────────────
    # Helpers
    # ──────────────────────────────────────────────────────────────────────

    def _build_optimizer(self, lr: float) -> torch.optim.Optimizer:
        params = [p for p in self.model.parameters() if p.requires_grad]
        return torch.optim.AdamW(
            params,
            lr=lr,
            weight_decay=self.cfg.get("weight_decay", 0.01),
        )

    def _build_scheduler(self, optimizer, epochs: int):
        from torch.optim.lr_scheduler import CosineAnnealingLR
        total_steps = epochs * len(self.train_loader)
        return CosineAnnealingLR(optimizer, T_max=total_steps, eta_min=1e-7)

    def _to_device(self, batch: dict) -> dict:
        return {
            k: v.to(self.device) if isinstance(v, torch.Tensor) else v
            for k, v in batch.items()
        }

    def _qwen_has_lora(self) -> bool:
        return any("lora_" in n for n, _ in self.model.qwen.named_parameters())
